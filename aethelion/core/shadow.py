"""
Construtor de Volumes de Sombra e Extrator de Silhuetas (shadow.py)

Identifica faces voltadas para a luz e arestas de silhueta em malhas fechadas
(2-manifold, Crow, 1977; Heidmann, 1991), extrudando quadriláteros laterais e
tampas frontal/traseira tanto em distância finita (Carmack) quanto em
coordenadas homogêneas no infinito (`w = 0`, Everitt & Kilgard, 2002), além de
testar inclusão da câmera nos volumes por ray casting (Möller & Trumbore, 1997).
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


def _select_silhouette(pw: np.ndarray, indices: np.ndarray,
                       edge_face: np.ndarray, edge_opp: np.ndarray,
                       light: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula a máscara de faces iluminadas e os índices das arestas de silhueta."""
    facing = face_facing(pw, indices, light)
    if not facing.any():
        return facing, np.empty(0, dtype=np.intp)
    is_sil = facing[edge_face] & ((edge_opp < 0) | ~facing[np.maximum(edge_opp, 0)])
    return facing, np.nonzero(is_sil)[0]


def extract_silhouette_segments(pw: np.ndarray, indices: np.ndarray,
                                dir_edges: np.ndarray, edge_face: np.ndarray,
                                edge_opp: np.ndarray, light: np.ndarray) -> np.ndarray:
    """Retorna os segmentos 3D (S, 2, 3) das arestas de silhueta para inspeção visual."""
    _, sel = _select_silhouette(pw, indices, edge_face, edge_opp, light)
    if not sel.size:
        return np.zeros((0, 2, 3), dtype=F)
    a = pw[dir_edges[sel, 0]]
    b = pw[dir_edges[sel, 1]]
    return np.stack([a, b], axis=1).astype(F)


def build_volume_homogeneous(pw, indices, dir_edges, edge_face, edge_opp,
                             light, length: float = 40.0,
                             infinite: bool = False, caps: bool = True):
    """Gera o volume de sombra em coordenadas homogêneas de mundo (N, 3, 4).

    Quando infinite=True, os vértices extrudados das laterais e da tampa
    traseira possuem w = 0 (direções puras no infinito segundo Everitt & Kilgard, 2002).
    """
    facing, sel = _select_silhouette(pw, indices, edge_face, edge_opp, light)
    if not facing.any():
        z = np.zeros((0, 3, 4), dtype=F)
        return z, z, z

    pw_h = np.concatenate([pw, np.ones((pw.shape[0], 1), dtype=F)], axis=1)
    ext_h = _extrude_homogeneous(pw, light, length, infinite=infinite)

    if sel.size:
        ia, ib = dir_edges[sel, 0], dir_edges[sel, 1]
        a, b = pw_h[ia], pw_h[ib]
        a2, b2 = ext_h[ia], ext_h[ib]
        # A tampa frontal percorre a aresta de silhueta como (a -> b).
        # Para que o volume fechado seja 2-manifold orientável com normais
        # externas consistentes em todas as faces, o quadrilátero lateral
        # deve percorrer a aresta compartilhada no sentido oposto (b -> a).
        sides = np.concatenate([
            np.stack([b, a, a2], axis=1),
            np.stack([b, a2, b2], axis=1),
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


def build_volume(pw, indices, dir_edges, edge_face, edge_opp,
                 light, length: float, caps: bool = True):
    """Gera o volume de sombra finito como sopa de triângulos em espaço de mundo 3D (N, 3, 3).

    Retorna (sides (A,3,3), front_cap (B,3,3), back_cap (C,3,3)).
    Reutiliza `build_volume_homogeneous(..., infinite=False)` projetando para 3D.
    """
    sides_h, front_h, back_h = build_volume_homogeneous(
        pw, indices, dir_edges, edge_face, edge_opp,
        light, length=length, infinite=False, caps=caps,
    )
    return sides_h[..., :3], front_h[..., :3], back_h[..., :3]


def point_in_volume(point: np.ndarray, tris: np.ndarray) -> bool:
    """Teste de ponto dentro de um volume fechado por paridade de interseções.

    Usado para contar quantos volumes de sombra contêm a câmera — a variável
    que explica a divergência entre z-pass e z-fail (hipótese H1).
    """
    if tris.shape[0] == 0:
        return False
    pt = np.asarray(point, dtype=F)
    vmin = tris.min(axis=(0, 1))
    vmax = tris.max(axis=(0, 1))
    if np.any(pt < vmin - 1e-4) or np.any(pt > vmax + 1e-4):
        return False
    # Direção unitária assimétrica para evitar planos de simetria (x=z, x=y) de malhas alinhadas aos eixos
    direction = np.array([0.43643578, 0.87287156, 0.21821789], dtype=F)
    hits = _ray_hits(pt, direction, tris)
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
