"""Varredura de triângulos por funções de aresta, com regra de preenchimento
exclusiva (top-left generalizada).

A regra de desempate é antissimétrica sobre a aresta orientada: para uma aresta
compartilhada entre dois triângulos adjacentes, a direção orientada em relação
ao interior de cada triângulo é oposta, logo o predicado vale para exatamente
um deles. Isso garante, por construção, que um pixel sobre a aresta seja
coberto UMA vez — sem furos e sem dupla cobertura.

A propriedade é essencial para shadow volumes: dupla cobertura de uma aresta
incrementa o contador de stencil duas vezes e produz sombra espúria; furos
sobre a aresta deixam escapar contagens de entrada/saída do volume.
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


def scan_triangle(sx, sy, width: int, height: int, min_area: float = 1e-9,
                  cull_backface: bool = False):
    """Rasteriza um triângulo em espaço de tela.

    sx, sy : (3,) coordenadas de tela dos vértices (origem no topo-esquerda).
    cull_backface : se True, descarta triângulos com area >= 0.0 antes de varrer pixels.
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
    if cull_backface and area >= 0.0:
        return None

    x0 = max(int(np.ceil(sx.min() - 0.5)), 0)
    x1 = min(int(np.floor(sx.max() - 0.5)), width - 1)
    y0 = max(int(np.ceil(sy.min() - 0.5)), 0)
    y1 = min(int(np.floor(sy.max() - 0.5)), height - 1)
    if x0 > x1 or y0 > y1:
        return None

    px = np.arange(x0, x1 + 1, dtype=F)[None, :] + F(0.5)
    py = np.arange(y0, y1 + 1, dtype=F)[:, None] + F(0.5)

    # Orienta funções de aresta e vetores diretores pelo sinal de area,
    # evitando resíduos de ponto flutuante na 3ª aresta e garantindo que
    # _top_left respeite o sentido ciclicamente orientado do interior.
    s = F(1.0) if area > 0.0 else F(-1.0)
    e0 = ((sx[1] - px) * (sy[2] - py) - (sx[2] - px) * (sy[1] - py)) * s
    e1 = ((sx[2] - px) * (sy[0] - py) - (sx[0] - px) * (sy[2] - py)) * s
    e2 = ((sx[0] - px) * (sy[1] - py) - (sx[1] - px) * (sy[0] - py)) * s

    tl0 = _top_left(float((sx[2] - sx[1]) * s), float((sy[2] - sy[1]) * s))
    tl1 = _top_left(float((sx[0] - sx[2]) * s), float((sy[0] - sy[2]) * s))
    tl2 = _top_left(float((sx[1] - sx[0]) * s), float((sy[1] - sy[0]) * s))

    m0 = (e0 > 0.0) | ((e0 == 0.0) & tl0)
    m1 = (e1 > 0.0) | ((e1 == 0.0) & tl1)
    m2 = (e2 > 0.0) | ((e2 == 0.0) & tl2)
    mask = m0 & m1 & m2
    if not mask.any():
        return None

    inv_abs = F(1.0) / abs(area)
    l0 = e0[mask] * inv_abs
    l1 = e1[mask] * inv_abs
    l2 = F(1.0) - l0 - l1

    yy, xx = np.nonzero(mask)
    return (yy + y0, xx + x0, l0, l1, l2, float(area))
