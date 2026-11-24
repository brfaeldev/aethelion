"""
Interface de Linha de Comando (cli.py)

Analisa argumentos de execução, arquivos de configuração JSON e presets de
desempenho/qualidade (`SVSettings`), orquestrando os modos de operação do
sistema: visualizador interativo em Pygame, suíte de conformidade OpenGL 1.4,
roteiro experimental de 8 fases, varreduras paramétricas e geração de relatórios.
"""
from __future__ import annotations

import argparse
import json

from .core.settings import BACKENDS, PRESETS, SVSettings, apply_preset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aethelion",
        description="Aethelion — sombras volumétricas por stencil (Eixo 7).",
    )

    m = p.add_argument_group("modo")
    m.add_argument("--experiment", "-experiment", action="store_true",
                   help="executa o roteiro de fases sem abrir janela")
    m.add_argument("--conformance", "-conformance", action="store_true",
                   help="executa apenas a suíte de conformidade")
    m.add_argument("--sv-sweeps", "-sv-sweeps", action="store_true",
                   help="executa varreduras de resolução, complexidade e extrusão P_inf")
    m.add_argument("--report", "-report", action="store_true",
                   help="gera gráficos e RELATORIO.md a partir dos CSVs")
    m.add_argument("-o", "--outdir", default="results",
                   help="diretório de saída dos artefatos (padrão: results)")
    m.add_argument("--measure-every", type=float, default=2.5,
                   help="intervalo (s) entre medições contra a referência (padrão: 2.5)")
    m.add_argument("--deterministic", "-deterministic", action="store_true",
                   help="usa passo temporal fixo por quadro em vez de relógio de parede")
    m.add_argument("--fixed-dt", type=float, default=0.5,
                   help="passo temporal fixo (s) quando --deterministic está ativo (padrão: 0.5)")
    m.add_argument("--config", type=str, default=None,
                   help="caminho de arquivo JSON com parâmetros de SVSettings")
    m.add_argument("--preset", choices=list(PRESETS), default=None,
                   help="perfil pronto de desempenho/qualidade")
    m.add_argument("--backend", choices=list(BACKENDS), default=None,
                   help="backend de renderização (padrão: cpu-sw)")

    w = p.add_argument_group("janela (visualizador)")
    w.add_argument("--fullscreen", action="store_true",
                   help="inicia a janela do visualizador em tela cheia (F11 alterna)")
    w.add_argument("--win-width", type=int, default=1280,
                   help="largura inicial da janela em pixels (padrão: 1280)")
    w.add_argument("--win-height", type=int, default=780,
                   help="altura inicial da janela em pixels (padrão: 780)")
    w.add_argument("--display-filter", choices=["nearest", "bilinear"], default=None,
                   help="filtro de escala na janela (padrão: nearest)")

    r = p.add_argument_group("renderizador (resolução interna e parâmetros)")
    r.add_argument("--width", type=int, default=None,
                   help="largura interna do framebuffer renderizado (padrão: 256)")
    r.add_argument("--height", type=int, default=None,
                   help="altura interna do framebuffer renderizado (padrão: 192)")
    r.add_argument("--detail", type=float, default=None,
                   help="densidade das malhas da cripta (padrão: 1.0)")
    r.add_argument("--algorithm", choices=["none", "zpass", "zfail"],
                   default=None, help="algoritmo de sombra (padrão: zfail)")
    r.add_argument("--stencil-bits", type=int, default=None,
                   choices=[1, 2, 3, 4, 5, 8],
                   help="largura do buffer de stencil em bits (padrão: 8)")
    r.add_argument("--no-wrap", action="store_true", default=None,
                   help="usa INCR/DECR saturados em vez de INCR_WRAP/DECR_WRAP")
    r.add_argument("--depth-bits", type=int, default=None,
                   help="0 = float32; senão largura do buffer de profundidade por fragmento")
    r.add_argument("--extrude-len", type=float, default=None,
                   help="comprimento da extrusão finita (padrão: 40.0)")
    r.add_argument("--infinite-extrude", action="store_true", default=None,
                   help="usa matriz P_inf e extrusão homogênea no infinito (w=0)")
    r.add_argument("--no-depth-clamp", action="store_true", default=None,
                   help="desativa depth clamping no plano distante em z-fail")
    r.add_argument("--fov", type=float, default=None,
                   help="campo de visão vertical em graus (padrão: 62.0)")
    r.add_argument("--near", type=float, default=None,
                   help="plano próximo da câmera (padrão: 0.15)")
    r.add_argument("--far", type=float, default=None,
                   help="plano distante da câmera (padrão: 60.0)")
    r.add_argument("--precision",
                   choices=["float32", "float16", "fixed16_16", "fixed12_4"],
                   default=None, help="precisão numérica (padrão: float32)")
    r.add_argument("--vertex-snap", type=int, default=None,
                   help="grade de snap de vértices estilo PS1 (0 = desativado)")
    r.add_argument("--shading", choices=["gouraud", "flat", "unlit"],
                   default=None, help="modelo de sombreamento (padrão: gouraud)")
    r.add_argument("--ambient", type=float, default=None,
                   help="intensidade da luz ambiente 0..1 (padrão: 0.17)")
    r.add_argument("--color-bits", type=int, default=None,
                   choices=[8, 5, 4, 3, 2, 1],
                   help="profundidade de cor por canal (padrão: 8)")
    r.add_argument("--dither", action="store_true", default=None,
                   help="ativa dithering ordenado Bayer 4x4")
    r.add_argument("--no-texture", action="store_true", default=None,
                   help="desativa texturas")
    r.add_argument("--texture-res", type=int, default=None,
                   help="resolução das texturas procedurais (padrão: 64)")
    r.add_argument("--texture-filter", choices=["nearest", "bilinear"],
                   default=None, help="filtro de amostragem de textura (padrão: nearest)")
    return p


def settings_from(a) -> SVSettings:
    if getattr(a, "config", None):
        with open(a.config, "r", encoding="utf-8") as f:
            s = SVSettings.from_dict(json.load(f))
    else:
        s = SVSettings()

    if getattr(a, "preset", None):
        apply_preset(s, a.preset)

    direct_map = {
        "backend": "backend",
        "display_filter": "display_filter",
        "width": "width",
        "height": "height",
        "detail": "detail",
        "algorithm": "algorithm",
        "stencil_bits": "stencil_bits",
        "depth_bits": "depth_bits",
        "extrude_len": "extrude_len",
        "fov": "fov",
        "near": "near",
        "far": "far",
        "precision": "precision",
        "vertex_snap": "vertex_snap",
        "shading": "shading",
        "ambient": "ambient",
        "color_bits": "color_bits",
        "texture_res": "texture_res",
        "texture_filter": "texture_filter",
    }
    for arg_k, attr_k in direct_map.items():
        val = getattr(a, arg_k, None)
        if val is not None:
            setattr(s, attr_k, val)

    if getattr(a, "no_wrap", None):
        s.stencil_wrap = False
    if getattr(a, "infinite_extrude", None):
        s.infinite_extrude = True
    if getattr(a, "no_depth_clamp", None):
        s.depth_clamp = False
    if getattr(a, "dither", None):
        s.dither = True
    if getattr(a, "no_texture", None):
        s.textured = False
    return s.validate()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.conformance:
        from .evaluation import conformance
        res = conformance.run(args.outdir)
        return 0 if res["passed"] == res["total"] else 2

    if args.sv_sweeps:
        from .evaluation.sweeps import run_sv_sweeps
        run_sv_sweeps(settings_from(args), args.outdir)
        return 0

    if args.report:
        from .evaluation import report
        report.build(args.outdir)
        return 0

    if args.experiment:
        from .evaluation import conformance, report
        from .evaluation.runner import run_headless
        from .evaluation.sweeps import run_sv_sweeps
        print("=== Conformidade do instrumento ===")
        conformance.run(args.outdir)
        print("\n=== Roteiro experimental (8 fases) ===")
        run_headless(settings_from(args), args.outdir, args.measure_every,
                     deterministic=args.deterministic, fixed_dt=args.fixed_dt)
        print("\n=== Varreduras paramétricas (Resolução, Complexidade e P_inf) ===")
        run_sv_sweeps(settings_from(args), args.outdir)
        print("\n=== Relatório ===")
        report.build(args.outdir)
        return 0

    from .ui.viewer import main as run_viewer
    return run_viewer(settings_from(args), args.outdir,
                      deterministic=args.deterministic,
                      win_width=args.win_width, win_height=args.win_height,
                      fullscreen=args.fullscreen,
                      config_path=args.config)
