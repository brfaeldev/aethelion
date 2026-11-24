"""
Rasterizador de Triângulos com Regra Top-Left (raster.py)

Converte triângulos projetados em fragmentos 2D e coordenadas baricêntricas
utilizando funções de aresta (Pineda, 1988; Akenine-Möller et al., 2018) com
regra de desempate antissimétrica (top-left, OpenGL 1.4 Seção 3.5). Garante
cobertura única em arestas compartilhadas (sem furos nem dupla contagem no
stencil), com suporte opcional a compilação JIT via Numba.
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
    m0 &= (e1 >= 0.0) if tl1 else (e1 > 0.0)
    if not m0.any():
        return None
    m0 &= (e2 >= 0.0) if tl2 else (e2 > 0.0)
    idx = np.flatnonzero(m0)
    if idx.size == 0:
        return None

    bw = x1 - x0 + 1
    yy, xx = np.divmod(idx, bw)
    inv_abs = 1.0 / abs(area)
    l0 = (e0.ravel()[idx] * inv_abs).astype(F)
    l1 = (e1.ravel()[idx] * inv_abs).astype(F)
    l2 = F(1.0) - l0 - l1

    return (yy + y0, xx + x0, l0, l1, l2, area)


try:
    from numba import njit as _njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _njit = None
    HAS_NUMBA = False


if HAS_NUMBA:  # pragma: no cover
    @_njit(cache=True)
    def _scan_numba(sx0: float, sy0: float, sx1: float, sy1: float,
                    sx2: float, sy2: float, x0: int, x1: int, y0: int, y1: int,
                    s: float, tl0: bool, tl1: bool, tl2: bool, inv_abs: float):
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        cap = bw * bh
        yy = np.empty(cap, dtype=np.intp)
        xx = np.empty(cap, dtype=np.intp)
        l0 = np.empty(cap, dtype=np.float32)
        l1 = np.empty(cap, dtype=np.float32)
        l2 = np.empty(cap, dtype=np.float32)

        a0 = (sy1 - sy2) * s
        b0 = (sx2 - sx1) * s
        c0 = (sx1 * sy2 - sx2 * sy1) * s
        a1 = (sy2 - sy0) * s
        b1 = (sx0 - sx2) * s
        c1 = (sx2 * sy0 - sx0 * sy2) * s
        a2 = (sy0 - sy1) * s
        b2 = (sx1 - sx0) * s
        c2 = (sx0 * sy1 - sx1 * sy0) * s

        n = 0
        for y in range(y0, y1 + 1):
            py = y + 0.5
            row0 = b0 * py + c0
            row1 = b1 * py + c1
            row2 = b2 * py + c2
            for x in range(x0, x1 + 1):
                px = x + 0.5
                e0 = a0 * px + row0
                if not (e0 >= 0.0 if tl0 else e0 > 0.0):
                    continue
                e1 = a1 * px + row1
                if not (e1 >= 0.0 if tl1 else e1 > 0.0):
                    continue
                e2 = a2 * px + row2
                if not (e2 >= 0.0 if tl2 else e2 > 0.0):
                    continue
                yy[n] = y
                xx[n] = x
                v0 = np.float32(e0 * inv_abs)
                v1 = np.float32(e1 * inv_abs)
                l0[n] = v0
                l1[n] = v1
                l2[n] = np.float32(1.0) - v0 - v1
                n += 1
        return yy[:n], xx[:n], l0[:n], l1[:n], l2[:n], n


def scan_triangle_fast(sx, sy, width: int, height: int, min_area: float = 1e-9,
                       cull_backface: bool = False):
    """Variante acelerada de `scan_triangle` (usa Numba se disponível)."""
    if not HAS_NUMBA:
        return scan_triangle(sx, sy, width, height, min_area=min_area,
                             cull_backface=cull_backface)
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

    min_x = min(sx0, sx1, sx2)
    max_x = max(sx0, sx1, sx2)
    min_y = min(sy0, sy1, sy2)
    max_y = max(sy0, sy1, sy2)
    x0 = max(0, int(math.ceil(min_x - 0.5)))
    x1 = min(width - 1, int(math.floor(max_x - 0.5)))
    if x0 > x1:
        return None
    y0 = max(0, int(math.ceil(min_y - 0.5)))
    y1 = min(height - 1, int(math.floor(max_y - 0.5)))
    if y0 > y1:
        return None

    s = 1.0 if area > 0.0 else -1.0
    tl0 = _top_left((sx2 - sx1) * s, (sy2 - sy1) * s)
    tl1 = _top_left((sx0 - sx2) * s, (sy0 - sy2) * s)
    tl2 = _top_left((sx1 - sx0) * s, (sy1 - sy0) * s)
    yy, xx, l0, l1, l2, n = _scan_numba(
        sx0, sy0, sx1, sy1, sx2, sy2, x0, x1, y0, y1,
        s, tl0, tl1, tl2, 1.0 / abs(area),
    )
    if n == 0:
        return None
    return (yy, xx, l0, l1, l2, area)
