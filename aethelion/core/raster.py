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
    sx64 = np.asarray(sx, dtype=np.float64)
    sy64 = np.asarray(sy, dtype=np.float64)
    if not (np.all(np.isfinite(sx64)) and np.all(np.isfinite(sy64))):
        return None

    area = float((sx64[1] - sx64[0]) * (sy64[2] - sy64[0])
                 - (sx64[2] - sx64[0]) * (sy64[1] - sy64[0]))
    if not np.isfinite(area) or abs(area) < min_area:
        return None
    if cull_backface and area >= 0.0:
        return None

    x0 = max(int(np.ceil(sx64.min() - 0.5)), 0)
    x1 = min(int(np.floor(sx64.max() - 0.5)), width - 1)
    y0 = max(int(np.ceil(sy64.min() - 0.5)), 0)
    y1 = min(int(np.floor(sy64.max() - 0.5)), height - 1)
    if x0 > x1 or y0 > y1:
        return None

    px = np.arange(x0, x1 + 1, dtype=np.float64)[None, :] + 0.5
    py = np.arange(y0, y1 + 1, dtype=np.float64)[:, None] + 0.5

    # Orienta funções de aresta e vetores diretores pelo sinal de area,
    # evitando resíduos de ponto flutuante na 3ª aresta e garantindo que
    # _top_left respeite o sentido ciclicamente orientado do interior.
    s = 1.0 if area > 0.0 else -1.0
    e0 = ((sx64[1] - px) * (sy64[2] - py) - (sx64[2] - px) * (sy64[1] - py)) * s
    e1 = ((sx64[2] - px) * (sy64[0] - py) - (sx64[0] - px) * (sy64[2] - py)) * s
    e2 = ((sx64[0] - px) * (sy64[1] - py) - (sx64[1] - px) * (sy64[0] - py)) * s

    tl0 = _top_left(float((sx64[2] - sx64[1]) * s), float((sy64[2] - sy64[1]) * s))
    tl1 = _top_left(float((sx64[0] - sx64[2]) * s), float((sy64[0] - sy64[2]) * s))
    tl2 = _top_left(float((sx64[1] - sx64[0]) * s), float((sy64[1] - sy64[0]) * s))

    m0 = (e0 > 0.0) | ((e0 == 0.0) & tl0)
    m1 = (e1 > 0.0) | ((e1 == 0.0) & tl1)
    m2 = (e2 > 0.0) | ((e2 == 0.0) & tl2)
    mask = m0 & m1 & m2
    if not mask.any():
        return None

    inv_abs = 1.0 / abs(area)
    l0 = (e0[mask] * inv_abs).astype(F)
    l1 = (e1[mask] * inv_abs).astype(F)
    l2 = F(1.0) - l0 - l1

    yy, xx = np.nonzero(mask)
    return (yy + y0, xx + x0, l0, l1, l2, float(area))
