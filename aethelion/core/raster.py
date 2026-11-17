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

import math
import numpy as np

F = np.float32

_GRID_CACHE: dict[int, np.ndarray] = {}


def _get_grid(n: int) -> np.ndarray:
    arr = _GRID_CACHE.get(n)
    if arr is None:
        arr = np.arange(n, dtype=np.float64) + 0.5
        _GRID_CACHE[n] = arr
    return arr


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
    sx0, sx1, sx2 = float(sx[0]), float(sx[1]), float(sx[2])
    sy0, sy1, sy2 = float(sy[0]), float(sy[1]), float(sy[2])
    if not (math.isfinite(sx0) and math.isfinite(sx1) and math.isfinite(sx2)
            and math.isfinite(sy0) and math.isfinite(sy1) and math.isfinite(sy2)):
        return None

    area = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0)
    if not math.isfinite(area) or abs(area) < min_area:
        return None
    if cull_backface and area >= 0.0:
        return None

    min_x = sx0 if sx0 < sx1 else sx1
    if sx2 < min_x:
        min_x = sx2
    max_x = sx0 if sx0 > sx1 else sx1
    if sx2 > max_x:
        max_x = sx2
    min_y = sy0 if sy0 < sy1 else sy1
    if sy2 < min_y:
        min_y = sy2
    max_y = sy0 if sy0 > sy1 else sy1
    if sy2 > max_y:
        max_y = sy2

    x0 = int(math.ceil(min_x - 0.5))
    if x0 < 0:
        x0 = 0
    x1 = int(math.floor(max_x - 0.5))
    if x1 >= width:
        x1 = width - 1
    if x0 > x1:
        return None

    y0 = int(math.ceil(min_y - 0.5))
    if y0 < 0:
        y0 = 0
    y1 = int(math.floor(max_y - 0.5))
    if y1 >= height:
        y1 = height - 1
    if y0 > y1:
        return None

    # Orienta funções de aresta e vetores diretores pelo sinal de area,
    # evitando resíduos de ponto flutuante na 3ª aresta e garantindo que
    # _top_left respeite o sentido ciclicamente orientado do interior.
    s = 1.0 if area > 0.0 else -1.0
    tl0 = _top_left((sx2 - sx1) * s, (sy2 - sy1) * s)
    tl1 = _top_left((sx0 - sx2) * s, (sy0 - sy2) * s)
    tl2 = _top_left((sx1 - sx0) * s, (sy1 - sy0) * s)

    # Caminho rápido para bounding box 1x1
    if x0 == x1 and y0 == y1:
        pxs = x0 + 0.5
        pys = y0 + 0.5
        e0s = ((sx1 - pxs) * (sy2 - pys) - (sx2 - pxs) * (sy1 - pys)) * s
        if not (e0s > 0.0 or (e0s == 0.0 and tl0)):
            return None
        e1s = ((sx2 - pxs) * (sy0 - pys) - (sx0 - pxs) * (sy2 - pys)) * s
        if not (e1s > 0.0 or (e1s == 0.0 and tl1)):
            return None
        e2s = ((sx0 - pxs) * (sy1 - pys) - (sx1 - pxs) * (sy0 - pys)) * s
        if not (e2s > 0.0 or (e2s == 0.0 and tl2)):
            return None
        inv_abs = 1.0 / abs(area)
        l0v = F(e0s * inv_abs)
        l1v = F(e1s * inv_abs)
        l2v = F(1.0) - l0v - l1v
        return (
            np.array([y0], dtype=np.intp),
            np.array([x0], dtype=np.intp),
            np.array([l0v], dtype=F),
            np.array([l1v], dtype=F),
            np.array([l2v], dtype=F),
            area,
        )

    px = _get_grid(width)[x0:x1 + 1]
    py = _get_grid(height)[y0:y1 + 1]

    # Forma separável 1D + 1D -> 2D: e_k(x, y) = a_k * px + (b_k * py + c_k)
    a0 = (sy1 - sy2) * s
    b0 = (sx2 - sx1) * s
    c0 = (sx1 * sy2 - sx2 * sy1) * s
    e0 = (a0 * px)[None, :] + (b0 * py + c0)[:, None]

    a1 = (sy2 - sy0) * s
    b1 = (sx0 - sx2) * s
    c1 = (sx2 * sy0 - sx0 * sy2) * s
    e1 = (a1 * px)[None, :] + (b1 * py + c1)[:, None]

    a2 = (sy0 - sy1) * s
    b2 = (sx1 - sx0) * s
    c2 = (sx0 * sy1 - sx1 * sy0) * s
    e2 = (a2 * px)[None, :] + (b2 * py + c2)[:, None]

    m0 = (e0 >= 0.0) if tl0 else (e0 > 0.0)
    m1 = (e1 >= 0.0) if tl1 else (e1 > 0.0)
    m2 = (e2 >= 0.0) if tl2 else (e2 > 0.0)
    mask = m0 & m1 & m2
    if not mask.any():
        return None

    inv_abs = 1.0 / abs(area)
    l0 = (e0[mask] * inv_abs).astype(F)
    l1 = (e1[mask] * inv_abs).astype(F)
    l2 = F(1.0) - l0 - l1

    yy, xx = np.nonzero(mask)
    return (yy + y0, xx + x0, l0, l1, l2, area)
