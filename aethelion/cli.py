"""Interface de linha de comando (CLI) do pacote Aethelion.

Modos:
    python -m aethelion                 # aplicação de tela única (pygame)
    python -m aethelion --fullscreen    # abre diretamente em tela cheia
    python -m aethelion --experiment    # roteiro completo + varreduras + relatório
    python -m aethelion --conformance   # apenas a suíte de conformidade
    python -m aethelion --sv-sweeps     # varreduras de resolução, complexidade e P_inf
    python -m aethelion --report        # gráficos + RELATORIO.md dos CSVs
    python -m aethelion --help
"""
from __future__ import annotations

import argparse

from .core.sv_renderer import SVSettings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aethelion",
        description="Aethelion — sombras volumétricas por stencil (Eixo 7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    m = p.add_argument_group("modo")
    m.add_argument("--experiment", action="store_true",
                   help="executa o roteiro de fases sem abrir janela")
    m.add_argument("--conformance", action="store_true",
                   help="executa apenas a suíte de conformidade")
    m.add_argument("--sv-sweeps", action="store_true",
                   help="executa varreduras de resolução, complexidade e extrusão P_inf")
    m.add_argument("--report", action="store_true",
                   help="gera gráficos e RELATORIO.md a partir dos CSVs")
    m.add_argument("-o", "--outdir", default="results")
    m.add_argument("--measure-every", type=float, default=2.5,
                   help="intervalo (s) entre medições contra a referência")
    m.add_argument("--deterministic", action="store_true",
                   help="usa passo temporal fixo por quadro em vez de relógio de parede")
    m.add_argument("--fixed-dt", type=float, default=0.5,
                   help="passo temporal fixo (s) quando --deterministic está ativo")

    w = p.add_argument_group("janela (visualizador)")
    w.add_argument("--fullscreen", action="store_true",
                   help="inicia a janela do visualizador em tela cheia (F11 alterna)")
    w.add_argument("--win-width", type=int, default=1280,
                   help="largura inicial da janela em pixels")
    w.add_argument("--win-height", type=int, default=780,
                   help="altura inicial da janela em pixels")

    r = p.add_argument_group("renderizador (resolução interna e parâmetros)")
    r.add_argument("--width", type=int, default=256,
                   help="largura interna do framebuffer renderizado")
    r.add_argument("--height", type=int, default=192,
                   help="altura interna do framebuffer renderizado")
    r.add_argument("--detail", type=float, default=1.0,
                   help="densidade das malhas da cripta")
    r.add_argument("--algorithm", choices=["none", "zpass", "zfail"],
                   default="zfail")
    r.add_argument("--stencil-bits", type=int, default=8, choices=[1, 2, 3, 4, 5, 8])
    r.add_argument("--no-wrap", action="store_true",
                   help="usa INCR/DECR saturados em vez de INCR_WRAP/DECR_WRAP")
    r.add_argument("--depth-bits", type=int, default=0,
                   help="0 = float32; senão largura do buffer de profundidade por fragmento")
    r.add_argument("--extrude-len", type=float, default=40.0)
    r.add_argument("--infinite-extrude", action="store_true",
                   help="usa matriz P_inf e extrusão homogênea no infinito (w=0, Everitt & Kilgard)")
    r.add_argument("--no-depth-clamp", action="store_true",
                   help="desativa depth clamping no plano distante em z-fail")
    r.add_argument("--precision",
                   choices=["float32", "float16", "fixed16_16", "fixed12_4"],
                   default="float32")
    r.add_argument("--shading", choices=["gouraud", "flat", "unlit"],
                   default="gouraud")
    r.add_argument("--color-bits", type=int, default=8,
                   choices=[8, 5, 4, 3, 2, 1])
    r.add_argument("--dither", action="store_true")
    r.add_argument("--no-texture", action="store_true")
    return p


def settings_from(a) -> SVSettings:
    return SVSettings(
        width=a.width, height=a.height, detail=a.detail,
        algorithm=a.algorithm, stencil_bits=a.stencil_bits,
        stencil_wrap=not a.no_wrap, depth_bits=a.depth_bits,
        extrude_len=a.extrude_len, infinite_extrude=a.infinite_extrude,
        depth_clamp=not a.no_depth_clamp, precision=a.precision,
        shading=a.shading, color_bits=a.color_bits, dither=a.dither,
        textured=not a.no_texture,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.conformance:
        from .evaluation import conformance
        res = conformance.run(args.outdir)
        return 0 if res["passed"] == res["total"] else 2

    if args.sv_sweeps:
        from .evaluation.scenario import run_sv_sweeps
        run_sv_sweeps(settings_from(args), args.outdir)
        return 0

    if args.report:
        from .evaluation import report
        report.build(args.outdir)
        return 0

    if args.experiment:
        from .evaluation import conformance, report
        from .evaluation.scenario import run_headless, run_sv_sweeps
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
                      fullscreen=args.fullscreen)
