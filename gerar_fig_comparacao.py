#!/usr/bin/env python3
"""
Script Auxiliar de Geração da Figura Comparativa (gerar_fig_comparacao.py)

Reproduz a Figura 3 do artigo: painel comparativo 2x5 mostrando a cena
renderizada (acima) e o mapa de diferença contra o ray casting em falsa cor
(abaixo: verde = acordo, vermelho = falso positivo, azul = falso negativo)
nas fases C, D, E, F e H no instante médio u = 0,5.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from aethelion.core.backend import create_renderer
from aethelion.core.settings import SVSettings
from aethelion.evaluation.phases import PHASES, Phase, camera_for
from aethelion.evaluation.truth import (
    compare_masks,
    diff_image,
    evaluate_renderer_ground_truth,
)

SELECTED_PHASES: tuple[str, ...] = ("C", "D", "E", "F", "H")


def generate_comparison_figure(
    outdir: str = "results",
    u: float = 0.5,
    scale: int = 2,
    border: int = 2,
) -> tuple[str, list[dict]]:
    """Renderiza as fases C, D, E, F e H em `u` e monta o painel 2x5 da Figura 3."""
    os.makedirs(outdir, exist_ok=True)
    phase_map: dict[str, Phase] = {p.key: p for p in PHASES}

    top_imgs: list[np.ndarray] = []
    bot_imgs: list[np.ndarray] = []
    metrics: list[dict] = []

    print(f"\n=== Gerando Figura Comparativa (Fases {', '.join(SELECTED_PHASES)} em u={u}) ===")
    for key in SELECTED_PHASES:
        p = phase_map[key]
        s = SVSettings()
        for k, v in p.config.items():
            setattr(s, k, v)
        s.validate()

        r = create_renderer("cpu-sw", s)
        r.capture_gbuffer = True
        eye, tgt = camera_for(p.camera, u)
        r.camera_eye = eye
        r.camera_target = tgt
        t_scene = p.t_offset + u * p.duracao

        scene_img = r.render(t_scene)
        ref, pred, vis = evaluate_renderer_ground_truth(r)
        ag = compare_masks(pred[vis], ref[vis])
        diff_img = diff_image(pred, ref, vis)

        if scale > 1:
            scene_up = np.array(
                Image.fromarray(scene_img).resize(
                    (scene_img.shape[1] * scale, scene_img.shape[0] * scale),
                    Image.NEAREST,
                )
            )
            diff_up = np.array(
                Image.fromarray(diff_img).resize(
                    (diff_img.shape[1] * scale, diff_img.shape[0] * scale),
                    Image.NEAREST,
                )
            )
        else:
            scene_up = scene_img
            diff_up = diff_img

        top_imgs.append(scene_up)
        bot_imgs.append(diff_up)
        info = {
            "fase": key,
            "titulo": p.titulo,
            "precision": ag.precision,
            "recall": ag.recall,
            "f1": ag.f1,
            "error_rate": ag.error_rate,
        }
        metrics.append(info)
        print(
            f"  Fase {key} ({p.titulo:<24}) | "
            f"P={ag.precision:.4f} R={ag.recall:.4f} F1={ag.f1:.4f}"
        )

    h, w, _ = top_imgs[0].shape
    n_cols = len(top_imgs)
    grid_w = n_cols * w + (n_cols - 1) * border
    grid_h = 2 * h + border
    canvas = np.full((grid_h, grid_w, 3), 20, dtype=np.uint8)

    for idx, (top, bot) in enumerate(zip(top_imgs, bot_imgs)):
        x0 = idx * (w + border)
        canvas[0:h, x0:x0 + w] = top
        canvas[h + border:2 * h + border, x0:x0 + w] = bot

    out_path = os.path.join(outdir, "fig_comparacao_fases.png")
    Image.fromarray(canvas).save(out_path)
    print(f"  -> {out_path}")
    return out_path, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera a Figura 3 do artigo (cena e diferença contra ray casting nas fases C, D, E, F e H)."
    )
    parser.add_argument("-o", "--outdir", default="results", help="diretório de saída")
    parser.add_argument("--u", type=float, default=0.5, help="instante normalizado na fase (padrão: 0.5)")
    args = parser.parse_args(argv)
    generate_comparison_figure(outdir=args.outdir, u=args.u)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
