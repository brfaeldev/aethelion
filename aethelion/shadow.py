"""Construção de volumes de sombra a partir da silhueta.

Segue a formulação de Everitt e Kilgard (2002): para uma malha fechada e uma
fonte pontual, as arestas de silhueta são aquelas cujas duas faces adjacentes
discordam quanto a estarem voltadas para a luz. Cada aresta de silhueta é
extrudada no sentido oposto à luz, formando um quadrilátero; as tampas frontal
e traseira fecham o volume e são obrigatórias na variante z-fail.

DESVIO DOCUMENTADO: a extrusão aqui é FINITA (comprimento configurável), não
infinita com plano distante no infinito. O comprimento de extrusão é tratado
como fator experimental, justamente porque extrusão insuficiente ou recorte
pelo plano distante são fontes conhecidas de artefato em z-fail.
"""
from __future__ import annotations

import numpy as np

F = np.float32


def face_facing(pw: np.ndarray, indices: np.ndarray, light: np.ndarray) -> np.ndarray:
    """True para faces voltadas para a luz (fonte pontual)."""
    v0 = pw[indices[:, 0]]
    v1 = pw[indices[:, 1]]
    v2 = pw[indices[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    centroid = (v0 + v1 + v2) * F(1.0 / 3.0)
    return np.einsum("ij,ij->i", n, light[None, :] - centroid) > 0.0


def _extrude(points: np.ndarray, light: np.ndarray, length: float) -> np.ndarray:
    d = points - light[None, :]
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return (points + d / np.maximum(n, 1e-9) * F(length)).astype(F)


def build_volume(pw, indices, dir_edges, edge_face, edge_opp,
                 light, length: float, caps: bool = True):
    """Gera o volume de sombra como sopa de triângulos em espaço de mundo.

    Retorna (sides (A,3,3), front_cap (B,3,3), back_cap (C,3,3)).
    `sides` são os quadriláteros extrudados; as tampas vêm separadas porque
    z-pass pode descartá-las e z-fail não.
    """
    facing = face_facing(pw, indices, light)
    if not facing.any():
        z = np.zeros((0, 3, 3), dtype=F)
        return z, z, z

    opp = edge_opp
    is_sil = facing[edge_face] & ((opp < 0) | ~facing[np.maximum(opp, 0)])
    sel = np.nonzero(is_sil)[0]

    if sel.size:
        a = pw[dir_edges[sel, 0]]
        b = pw[dir_edges[sel, 1]]
        a2 = _extrude(a, light, length)
        b2 = _extrude(b, light, length)
        sides = np.concatenate([
            np.stack([a, b, b2], axis=1),
            np.stack([a, b2, a2], axis=1),
        ], axis=0).astype(F)
    else:
        sides = np.zeros((0, 3, 3), dtype=F)

    if not caps:
        z = np.zeros((0, 3, 3), dtype=F)
        return sides, z, z

    lit = indices[facing]
    front = np.stack([pw[lit[:, 0]], pw[lit[:, 1]], pw[lit[:, 2]]], axis=1).astype(F)
    ext = _extrude(pw, light, length)
    back = np.stack([ext[lit[:, 0]], ext[lit[:, 2]], ext[lit[:, 1]]], axis=1).astype(F)
    return sides, front, back


def point_in_volume(point: np.ndarray, tris: np.ndarray) -> bool:
    """Teste de ponto dentro de um volume fechado por paridade de interseções.

    Usado para contar quantos volumes de sombra contêm a câmera — a variável
    que explica a divergência entre z-pass e z-fail (hipótese H1).
    """
    if tris.shape[0] == 0:
        return False
    direction = np.array([0.5773503, 0.5773503, 0.5773503], dtype=F)
    hits = _ray_hits(point.astype(F), direction, tris)
    return bool(hits % 2 == 1)


def _ray_hits(origin: np.ndarray, direction: np.ndarray, tris: np.ndarray) -> int:
    """Möller-Trumbore vetorizado; devolve o número de interseções com t > eps."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(direction[None, :], e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-9
    if not ok.any():
        return 0
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    tvec = origin[None, :] - v0
    u = np.einsum("ij,ij->i", tvec, p) * inv
    ok &= (u >= 0.0) & (u <= 1.0)
    q = np.cross(tvec, e1)
    v = np.einsum("j,ij->i", direction, q) * inv
    ok &= (v >= 0.0) & (u + v <= 1.0)
    t = np.einsum("ij,ij->i", e2, q) * inv
    ok &= t > 1e-5
    return int(np.count_nonzero(ok))
