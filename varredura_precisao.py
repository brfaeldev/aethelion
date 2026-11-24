#!/usr/bin/env python3
"""
Script Auxiliar de Varredura de Precisão Numérica (varredura_precisao.py)

Reproduz a Tabela VI do artigo ("Precisão Numérica, 256x192, z-fail, 4 quadros"),
avaliando o impacto dos operadores de quantização aritmética (`float32`,
`float16`, `fixed16_16`, `fixed12_4` e `fixed12_4` com `vertex_snap = 1 px`)
sobre o tempo de quadro e as métricas de concordância (Precisão, Revocação e F1)
contra a verdade de referência por ray casting (Möller-Trumbore).
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace

from aethelion.core.settings import SVSettings
from aethelion.evaluation.sweeps import _eval_sv_config

CONFIGS: list[tuple[str, str, int]] = [
    ("float32", "float32", 0),
    ("float16", "float16", 0),
    ("fixed16_16", "fixed16_16", 0),
    ("fixed12_4", "fixed12_4", 0),
    ("fixed12_4 + snap 1 px", "fixed12_4", 1),
]


def run_precision_sweep(
    base: SVSettings | None = None,
    outdir: str = "results",
    frames: int = 4,
) -> list[dict]:
    """Executa a varredura de precisão numérica (Tabela VI do artigo) e salva CSV."""
    base = base or SVSettings()
    os.makedirs(outdir, exist_ok=True)
    rows: list[dict] = []

    print("\n=== Varredura SV: Precisão Numérica (Tabela VI) ===")
    for label, prec, snap in CONFIGS:
        cfg = replace(
            base,
            width=256,
            height=192,
            algorithm="zfail",
            detail=1.0,
            extrude_len=40.0,
            infinite_extrude=False,
            precision=prec,
            vertex_snap=snap,
        )
        row = _eval_sv_config(cfg, label, frames=frames)
        row["precisao"] = prec
        row["vertex_snap"] = snap
        rows.append(row)
        print(
            f"  {label:<22} | {row['mediana_ms_total']:7.1f} ms "
            f"| P={row['media_precision']:.3f} R={row['media_recall']:.3f} "
            f"| F1={row['media_f1']:.4f}±{row['dp_f1']:.4f}"
        )

    out_csv = os.path.join(outdir, "varredura_sv_precisao.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"  -> {out_csv}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera a Tabela VI do artigo (varredura de precisão numérica)."
    )
    parser.add_argument("-o", "--outdir", default="results", help="diretório de saída")
    parser.add_argument("--frames", type=int, default=4, help="quadros por configuração (padrão: 4)")
    args = parser.parse_args(argv)
    run_precision_sweep(outdir=args.outdir, frames=args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
