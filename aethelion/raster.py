"""Varredura de triângulos por funções de aresta, com regra de preenchimento
exclusiva (top-left generalizada).

A regra de desempate é antissimétrica sobre a aresta orientada: para uma aresta
compartilhada entre dois triângulos, a direção é oposta em cada um, logo o
predicado vale para exatamente um deles. Isso garante, por construção, que um
pixel sobre a aresta seja coberto UMA vez — sem furos e sem dupla cobertura.

A propriedade é essencial para shadow volumes: dupla cobertura de uma aresta
incrementa o contador de stencil duas vezes e produz sombra espúria.
"""
from __future__ import annotations

import numpy as np

F = np.float32


def _top_left(dx: float, dy: float) -> bool:
    """Predicado antissimétrico sobre a aresta orientada (dx, dy)."""
    if dy < 0.0:
        return True
    if dy == 0.0 and dx > 0.0:
        return True
    return False


def scan_triangle(sx, sy, width: int, height: int, min_area: float = 1e-9):
    """Rasteriza um triângulo em espaço de tela.

    sx, sy : (3,) coordenadas de tela dos vértices (origem no topo-esquerda).
    Retorna (yy, xx, l0, l1, l2, area) ou None se o triângulo não cobre pixels.
    l0, l1, l2 são coordenadas baricêntricas normalizadas (somam 1).
    """
    sx = np.asarray(sx, dtype=F)
    sy = np.asarray(sy, dtype=F)
    if not (np.all(np.isfinite(sx)) and np.all(np.isfinite(sy))):
        return None

    area = ((sx[1] - sx[0]) * (sy[2] - sy[0])
            - (sx[2] - sx[0]) * (sy[1] - sy[0]))
    if not np.isfinite(area) or abs(area) < min_area:
        return None

    x0 = max(int(np.ceil(sx.min() - 0.5)), 0)
    x1 = min(int(np.floor(sx.max() - 0.5)), width - 1)
    y0 = max(int(np.ceil(sy.min() - 0.5)), 0)
    y1 = min(int(np.floor(sy.max() - 0.5)), height - 1)
    if x0 > x1 or y0 > y1:
        return None

    px = np.arange(x0, x1 + 1, dtype=F)[None, :] + F(0.5)
    py = np.arange(y0, y1 + 1, dtype=F)[:, None] + F(0.5)
    inv = F(1.0) / area

    # l_i = area(p, v_j, v_k) / area(v0, v1, v2)
    l0 = ((sx[1] - px) * (sy[2] - py) - (sx[2] - px) * (sy[1] - py)) * inv
    l1 = ((sx[2] - px) * (sy[0] - py) - (sx[0] - px) * (sy[2] - py)) * inv
    l2 = F(1.0) - l0 - l1

    # Arestas orientadas associadas a cada baricêntrica
    tl0 = _top_left(float(sx[2] - sx[1]), float(sy[2] - sy[1]))
    tl1 = _top_left(float(sx[0] - sx[2]), float(sy[0] - sy[2]))
    tl2 = _top_left(float(sx[1] - sx[0]), float(sy[1] - sy[0]))

    m0 = (l0 > 0) | ((l0 == 0) & tl0)
    m1 = (l1 > 0) | ((l1 == 0) & tl1)
    m2 = (l2 > 0) | ((l2 == 0) & tl2)
    mask = m0 & m1 & m2
    if not mask.any():
        return None

    yy, xx = np.nonzero(mask)
    return (yy + y0, xx + x0, l0[mask], l1[mask], l2[mask], float(area))
