#!/usr/bin/env python3
"""Aethelion — estudo experimental de sombras volumétricas por stencil.

Modos:
    python main.py                       # aplicação de tela única (pygame)
    python main.py --experiment          # roteiro completo sem janela
    python main.py --conformance         # apenas a suíte de conformidade
    python main.py --report              # gráficos + RELATORIO.md dos CSVs
    python main.py --legacy-sweeps       # varreduras do estudo anterior
    python main.py --help
"""
from __future__ import annotations

import argparse

from aethelion.sv_renderer import SVSettings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aethelion — sombras volumétricas por stencil (Eixo 7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    m = p.add_argument_group("modo")
    m.add_argument("--experiment", action="store_true",
                   help="executa o roteiro de fases sem abrir janela")
    m.add_argument("--conformance", action="store_true",
                   help="executa apenas a suíte de conformidade")
    m.add_argument("--report", action="store_true",
                   help="gera gráficos e RELATORIO.md a partir dos CSVs")
    m.add_argument("--legacy-sweeps", action="store_true",
                   help="varreduras retrô do estudo anterior")
    m.add_argument("-o", "--outdir", default="results")
    m.add_argument("--measure-every", type=float, default=2.5,
                   help="intervalo (s) entre medições contra a referência")

    r = p.add_argument_group("renderizador")
    r.add_argument("--width", type=int, default=256)
    r.add_argument("--height", type=int, default=192)
    r.add_argument("--detail", type=float, default=1.0,
                   help="densidade das malhas da cripta")
    r.add_argument("--algorithm", choices=["none", "zpass", "zfail"],
                   default="zfail")
    r.add_argument("--stencil-bits", type=int, default=8, choices=[1, 2, 3, 4, 5, 8])
    r.add_argument("--no-wrap", action="store_true",
                   help="usa INCR/DECR saturados em vez de INCR_WRAP/DECR_WRAP")
    r.add_argument("--depth-bits", type=int, default=0,
                   help="0 = float32; senão largura do buffer de profundidade")
    r.add_argument("--extrude-len", type=float, default=40.0)
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
        extrude_len=a.extrude_len, precision=a.precision,
        shading=a.shading, color_bits=a.color_bits, dither=a.dither,
        textured=not a.no_texture)


def main() -> int:
    args = build_parser().parse_args()

    if args.conformance:
        from aethelion import conformance
        res = conformance.run(args.outdir)
        return 0 if res["passed"] == res["total"] else 2

    if args.report:
        from aethelion import report
        report.build(args.outdir)
        return 0

    if args.legacy_sweeps:
        from aethelion import experiments as ex
        from aethelion.pipeline import RenderSettings
        ex.run_all(RenderSettings(), args.outdir, 10)
        return 0

    if args.experiment:
        from aethelion import conformance, report
        from aethelion.scenario import run_headless
        print("=== Conformidade do instrumento ===")
        conformance.run(args.outdir)
        print("\n=== Roteiro experimental ===")
        run_headless(settings_from(args), args.outdir, args.measure_every)
        print("\n=== Relatório ===")
        report.build(args.outdir)
        return 0

    from aethelion.viewer import main as run_viewer
    return run_viewer(settings_from(args), args.outdir)


if __name__ == "__main__":
    raise SystemExit(main())
