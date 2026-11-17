"""Construção de volumes de sombra a partir da silhueta (finita e homogênea no infinito).

Segue a formulação de Everitt e Kilgard (2002): para uma malha fechada (2-manifold)
e uma fonte pontual L = (L_x, L_y, L_z, 1), as arestas de silhueta são aquelas cujas
duas faces adjacentes discordam quanto a estarem voltadas para a luz. Cada aresta
de silhueta é extrudada no sentido oposto à luz, formando um quadrilátero; as tampas
frontal e traseira fecham o volume e são obrigatórias na variante z-fail.

Suporta dois regimes de extrusão:
1. Extrusão finita por distância L (`extrude_len`), usada para investigar o
   truncamento de volumes (Hipótese H5) e o comportamento clássico de Carmack;
2. Extrusão homogênea até o infinito com w = 0:
       V_inf = <V_x - L_x, V_y - L_y, V_z - L_z, 0>
   combinada com a matriz P_inf (Everitt & Kilgard, 2002, Seções 3.2 e 4.2).
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


def _extrude_homogeneous(points: np.ndarray, light: np.ndarray,
                         length: float, infinite: bool = False) -> np.ndarray:
    """Devolve vértices extrudados em coordenadas homogêneas (N, 4).

    Quando infinite=True, retorna pontos direcionais no infinito com w = 0:
        V_inf = [ (V - L) / ||V - L||, 0.0 ]
    Quando infinite=False, retorna pontos finitos com w = 1:
        V_fin = [ V + ((V - L) / ||V - L||) * length, 1.0 ]
    """
    d = points - light[None, :]
    n = np.linalg.norm(d, axis=1, keepdims=True)
    unit_d = (d / np.maximum(n, 1e-9)).astype(F)
    if infinite:
        return np.concatenate([unit_d, np.zeros((points.shape[0], 1), dtype=F)], axis=1)
    fin = (points + unit_d * F(length)).astype(F)
    return np.concatenate([fin, np.ones((points.shape[0], 1), dtype=F)], axis=1)


def extract_silhouette_segments(pw: np.ndarray, indices: np.ndarray,
                                dir_edges: np.ndarray, edge_face: np.ndarray,
                                edge_opp: np.ndarray, light: np.ndarray) -> np.ndarray:
    """Retorna os segmentos 3D (S, 2, 3) das arestas de silhueta para inspeção visual."""
    facing = face_facing(pw, indices, light)
    if not facing.any():
        return np.zeros((0, 2, 3), dtype=F)
    opp = edge_opp
    is_sil = facing[edge_face] & ((opp < 0) | ~facing[np.maximum(opp, 0)])
    sel = np.nonzero(is_sil)[0]
    if not sel.size:
        return np.zeros((0, 2, 3), dtype=F)
    a = pw[dir_edges[sel, 0]]
    b = pw[dir_edges[sel, 1]]
    return np.stack([a, b], axis=1).astype(F)


def build_volume(pw, indices, dir_edges, edge_face, edge_opp,
                 light, length: float, caps: bool = True):
    """Gera o volume de sombra como sopa de triângulos em espaço de mundo (3D).

    Retorna (sides (A,3,3), front_cap (B,3,3), back_cap (C,3,3)).
    `sides` são os quadriláteros extrudados; as tampas vêm separadas porque
    z-pass pode descartá-las (`caps=False`) e z-fail exige ambas (`caps=True`).
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


def build_volume_homogeneous(pw, indices, dir_edges, edge_face, edge_opp,
                             light, length: float = 40.0,
                             infinite: bool = False, caps: bool = True):
    """Gera o volume de sombra em coordenadas homogêneas de mundo (N, 3, 4).

    Quando infinite=True, os vértices extrudados das laterais e da tampa
    traseira possuem w = 0 (direções puras no infinito segundo Everitt & Kilgard, 2002).
    """
    facing = face_facing(pw, indices, light)
    if not facing.any():
        z = np.zeros((0, 3, 4), dtype=F)
        return z, z, z

    opp = edge_opp
    is_sil = facing[edge_face] & ((opp < 0) | ~facing[np.maximum(opp, 0)])
    sel = np.nonzero(is_sil)[0]

    pw_h = np.concatenate([pw, np.ones((pw.shape[0], 1), dtype=F)], axis=1)
    ext_h = _extrude_homogeneous(pw, light, length, infinite=infinite)

    if sel.size:
        ia, ib = dir_edges[sel, 0], dir_edges[sel, 1]
        a, b = pw_h[ia], pw_h[ib]
        a2, b2 = ext_h[ia], ext_h[ib]
        sides = np.concatenate([
            np.stack([a, b, b2], axis=1),
            np.stack([a, b2, a2], axis=1),
        ], axis=0).astype(F)
    else:
        sides = np.zeros((0, 3, 4), dtype=F)

    if not caps:
        z = np.zeros((0, 3, 4), dtype=F)
        return sides, z, z

    lit = indices[facing]
    front = np.stack([pw_h[lit[:, 0]], pw_h[lit[:, 1]], pw_h[lit[:, 2]]], axis=1).astype(F)
    back = np.stack([ext_h[lit[:, 0]], ext_h[lit[:, 2]], ext_h[lit[:, 1]]], axis=1).astype(F)
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
