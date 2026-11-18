"""Montagem da cena de ensaio: uma cripta procedural (two-manifold).

A cena foi construída para isolar as condições críticas do algoritmo de sombras:
* colunas cilíndricas -> silhuetas longas e curvas;
* grade de barras finas -> alta sobreposição de volumes no mesmo pixel (H3);
* arcos e corredor -> travessia da câmera por volumes de sombra (H1);
* chão e paredes amplos -> receptores de sombra de grande área (H2, H4, H5).
"""
from __future__ import annotations

import numpy as np

from ..core import mathutil as mu
from ..core.mesh import Prop
from .primitives import box, cylinder, torus_solid, uv_sphere

F = np.float32


def build_crypt(detail: float = 1.0) -> list[Prop]:
    """Monta a cripta. `detail` escala a densidade das malhas curvas."""
    d = float(np.clip(detail, 0.25, 4.0))
    seg = max(6, int(14 * d))

    col = cylinder(seg, radius=0.30, height=3.0)
    cap = box(0.80, 0.22, 0.80)
    base_blk = box(0.90, 0.26, 0.90)
    beam = box(9.2, 0.38, 0.52)
    bar = box(0.075, 2.1, 0.075)
    floor = box(14.0, 0.40, 14.0)
    wall = box(0.5, 5.0, 14.0)
    pedestal = box(1.1, 0.7, 1.1)
    head = uv_sphere(max(5, int(9 * d)), max(6, int(13 * d)), 0.33)
    ring = torus_solid(max(8, int(16 * d)), max(5, int(9 * d)), 0.52, 0.14)
    body = cylinder(max(6, int(10 * d)), radius=0.26, height=1.15)

    props: list[Prop] = [
        Prop(floor, mu.translate(0, -1.60, 0), (0.52, 0.50, 0.47),
             occluder=False, texture="checker", name="chao"),
        Prop(wall, mu.translate(-5.2, 1.0, 0), (0.46, 0.44, 0.42),
             occluder=False, texture="bricks", name="parede_oeste"),
        Prop(wall, mu.translate(5.2, 1.0, 0), (0.46, 0.44, 0.42),
             occluder=False, texture="bricks", name="parede_leste"),
    ]

    # Duas fileiras de colunas, com base e capitel identificados univocamente
    for k, z in enumerate(np.linspace(-4.0, 4.0, 5)):
        for x in (-2.7, 2.7):
            props.append(Prop(col, mu.translate(x, 0.0, float(z)),
                              (0.80, 0.78, 0.72), True, "bricks", f"coluna_{k}_{x:+.0f}"))
            props.append(Prop(base_blk, mu.translate(x, -1.26, float(z)),
                              (0.70, 0.68, 0.64), True, "bricks", f"base_{k}_{x:+.0f}"))
            props.append(Prop(cap, mu.translate(x, 1.58, float(z)),
                              (0.82, 0.80, 0.74), True, "bricks", f"capitel_{k}_{x:+.0f}"))

    # Arquitraves longitudinais
    for x in (-2.7, 2.7):
        props.append(Prop(beam,
                          mu.translate(x, 1.88, 0.0) @ mu.rotate_y(np.pi / 2),
                          (0.74, 0.72, 0.68), True, "bricks", f"viga_{x:+.0f}"))

    # Grade: barras finas e próximas -> alta sobreposição de volumes de sombra
    for i, x in enumerate(np.linspace(-1.9, 1.9, 11)):
        props.append(Prop(bar, mu.translate(float(x), 0.25, -4.9),
                          (0.38, 0.36, 0.34), True, "noise", f"grade_{i}"))

    # Estátua central
    props.append(Prop(pedestal, mu.translate(0, -1.05, 0.6),
                      (0.66, 0.64, 0.60), True, "gradient", "pedestal"))
    props.append(Prop(body, mu.translate(0, -0.12, 0.6),
                      (0.78, 0.76, 0.70), True, "gradient", "estatua_corpo"))
    props.append(Prop(head, mu.translate(0, 0.72, 0.6),
                      (0.84, 0.82, 0.76), True, "gradient", "estatua_cabeca"))
    props.append(Prop(ring, mu.translate(0, 0.62, 0.6) @ mu.rotate_x(0.9),
                      (0.88, 0.74, 0.42), True, "gradient", "halo"))
    return props


def light_position(t: float) -> np.ndarray:
    """Lâmpada que oscila no corredor central entre as colunas. Determinística em t."""
    return np.array([
        1.95 * np.sin(0.21 * t),
        2.40 + 0.35 * np.sin(0.37 * t),
        3.10 * np.cos(0.17 * t),
    ], dtype=F)
