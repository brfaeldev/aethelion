"""Recorte homogêneo contra o plano próximo em espaço de clip (Sutherland-Hodgman)."""
from __future__ import annotations

import numpy as np

F = np.float32


def _clip_halfspace(poly: np.ndarray, attrs: list[np.ndarray],
                    dist: np.ndarray) -> tuple[np.ndarray | None, list[np.ndarray] | None]:
    """Recorta polígono convexo contra o semiespaço dist >= 0."""
    n = poly.shape[0]
    if n < 3:
        return None, None
    inside = dist >= 0.0
    if np.all(inside):
        return poly, attrs
    if not np.any(inside):
        return None, None

    out_p = []
    out_a = [[] for _ in attrs]
    for i in range(n):
        j = (i + 1) % n
        di, dj = float(dist[i]), float(dist[j])
        ini, inj = di >= 0.0, dj >= 0.0
        if ini:
            out_p.append(poly[i])
            for k, a in enumerate(attrs):
                out_a[k].append(a[i])
        if ini != inj:
            denom = di - dj
            if abs(denom) < 1e-12:
                t = 0.5
            else:
                t = float(np.clip(di / denom, 0.0, 1.0))
            out_p.append(poly[i] + (poly[j] - poly[i]) * F(t))
            for k, a in enumerate(attrs):
                out_a[k].append(a[i] + (a[j] - a[i]) * F(t))
    if len(out_p) < 3:
        return None, None
    return np.array(out_p, dtype=F), [np.array(a, dtype=F) for a in out_a]


def clip_near(poly_clip: np.ndarray, attrs: list[np.ndarray],
              eps: float = 1e-5, near_plane: bool = True):
    """Sutherland-Hodgman contra o plano near em espaço de clip.

    Garante simultaneamente:
      1. w >= eps (coordenada homogênea estritamente positiva, evitando divisão por zero);
      2. z_clip + w_clip >= 0 quando near_plane=True (plano próximo real do frustum
         z_ndc >= -1, evitando projeções extremas próximas a w = 0).

    poly_clip: (N, 4) posições homogêneas; attrs: lista de (N, k) atributos.
    Retorna (poly, attrs) recortados — polígono convexo com N' vértices, ou (None, None).
    """
    poly = np.asarray(poly_clip, dtype=F)
    if not np.all(np.isfinite(poly)):
        return None, None

    cur_p = poly
    cur_a = [np.asarray(a, dtype=F) for a in attrs]

    # 1. Semiespaço w >= eps
    cur_p, cur_a = _clip_halfspace(cur_p, cur_a, cur_p[:, 3] - F(eps))
    if cur_p is None:
        return None, None

    # 2. Plano próximo canônico z_clip + w_clip >= 0 (z_ndc >= -1)
    if near_plane:
        cur_p, cur_a = _clip_halfspace(cur_p, cur_a, cur_p[:, 2] + cur_p[:, 3])
        if cur_p is None:
            return None, None

    return cur_p, cur_a
