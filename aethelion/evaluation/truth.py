"""Verdade de referência por ray casting e métricas de concordância.

A referência NÃO usa stencil nem volumes: para cada pixel visível, lança-se um
raio do ponto de superfície até a fonte de luz e testa-se oclusão contra a
geometria da cena. É um método independente do algoritmo em avaliação, o que é
a condição para que precisão, revocação e F1 tenham significado.

Custo elevado: executa apenas nos quadros de medição e seu tempo de CPU é
descontado do relógio de duração das fases para não interferir na amostragem.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

F = np.float32


def occluder_soup(props, world_fn: Callable | None = None) -> np.ndarray:
    """Concatena os triângulos oclusores em espaço de mundo.

    Se `world_fn` for fornecido (ex.: `SVRenderer._world`), reutiliza as posições
    de mundo em cache em vez de recomputar a transformação matricial.
    """
    out = []
    for p in props:
        if not p.occluder:
            continue
        if world_fn is not None:
            pw, _ = world_fn(p)
        else:
            pos = p.solid.positions
            ph = np.concatenate([pos, np.ones((pos.shape[0], 1), dtype=F)], axis=1)
            pw = (p.model @ ph.T).T[:, :3].astype(F)
        idx = p.solid.indices
        out.append(np.stack([pw[idx[:, 0]], pw[idx[:, 1]], pw[idx[:, 2]]], axis=1))
    if not out:
        return np.zeros((0, 3, 3), dtype=F)
    return np.concatenate(out, axis=0).astype(F)


def shadow_mask_raycast(points: np.ndarray, light: np.ndarray,
                        tris: np.ndarray, bias: float = 2.5e-3,
                        chunk: int = 1024) -> np.ndarray:
    """True onde o ponto está em sombra. points: (N,3) em espaço de mundo."""
    n = points.shape[0]
    out = np.zeros(n, dtype=bool)
    if n == 0 or tris.shape[0] == 0:
        return out

    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    tri_min = tris.min(axis=1)
    tri_max = tris.max(axis=1)

    for start in range(0, n, chunk):
        pts = points[start:start + chunk]
        # Poda exata por AABB do feixe de segmentos [pts -> light]
        rmin = np.minimum(pts.min(axis=0), light) - F(1e-2)
        rmax = np.maximum(pts.max(axis=0), light) + F(1e-2)
        cand = np.all((tri_max >= rmin[None, :]) & (tri_min <= rmax[None, :]), axis=1)
        if not cand.any():
            continue

        v0_c = v0[cand]
        e1_c = e1[cand]
        e2_c = e2[cand]

        d = light[None, :] - pts
        dist = np.linalg.norm(d, axis=1, keepdims=True)
        dirs = d / np.maximum(dist, 1e-9)
        origin = pts + dirs * bias
        limit = (dist[:, 0] - 2.0 * bias)

        # (P, T_cand) por broadcasting
        pv = np.cross(dirs[:, None, :], e2_c[None, :, :])
        det = np.einsum("ptj,tj->pt", pv, e1_c)
        ok = np.abs(det) > 1e-9
        inv = np.zeros_like(det)
        np.divide(1.0, det, out=inv, where=ok)

        tvec = origin[:, None, :] - v0_c[None, :, :]
        u = np.einsum("ptj,ptj->pt", tvec, pv) * inv
        ok &= (u >= 0.0) & (u <= 1.0)

        qv = np.cross(tvec, e1_c[None, :, :])
        v = np.einsum("pj,ptj->pt", dirs, qv) * inv
        ok &= (v >= 0.0) & ((u + v) <= 1.0)

        t = np.einsum("tj,ptj->pt", e2_c, qv) * inv
        ok &= (t > 0.0) & (t < limit[:, None])

        out[start:start + chunk] = ok.any(axis=1)
    return out


@dataclass
class Agreement:
    """Concordância entre a máscara prevista e a de referência."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return float(self.tp / d) if d else 1.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return float(self.tp / d) if d else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @property
    def error_rate(self) -> float:
        tot = self.tp + self.fp + self.fn + self.tn
        return float((self.fp + self.fn) / tot) if tot else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(precision=self.precision, recall=self.recall,
                 f1=self.f1, error_rate=self.error_rate)
        return d


def compare_masks(pred: np.ndarray, truth: np.ndarray) -> Agreement:
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(truth, dtype=bool)
    return Agreement(tp=int(np.count_nonzero(p & g)),
                     fp=int(np.count_nonzero(p & ~g)),
                     fn=int(np.count_nonzero(~p & g)),
                     tn=int(np.count_nonzero(~p & ~g)))


def diff_image(pred: np.ndarray, truth: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Imagem de falsa cor: verde=acerto, vermelho=falso positivo, azul=falso negativo."""
    h, w = pred.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[...] = 24
    agree = visible & (pred == truth)
    fp = visible & pred & ~truth
    fn = visible & ~pred & truth
    img[agree] = (40, 150, 60)
    img[fp] = (210, 50, 40)
    img[fn] = (50, 90, 220)
    return img


def evaluate_renderer_ground_truth(renderer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executa a referência por ray casting para o quadro atual de um `SVRenderer`.

    Retorna `(mask_ref, mask_pred, visible)`. Exige que o quadro anterior tenha sido
    renderizado com `capture_gbuffer=True`. Reutiliza o cache `renderer._world(prop)`.
    """
    vis = renderer.visible_mask
    ref = np.zeros_like(vis)
    if not vis.any():
        return ref, renderer.shadow_mask, vis
    pts = renderer.gbuf[vis]
    tris = occluder_soup(renderer.scene, world_fn=renderer._world)
    ref[vis] = shadow_mask_raycast(pts, renderer.light, tris)
    return ref, renderer.shadow_mask, vis
