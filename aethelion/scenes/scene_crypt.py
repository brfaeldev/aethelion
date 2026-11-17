"""Cena de ensaio: uma cripta procedural.

A cena não é decorativa. Ela foi construída para ser o PIOR CASO controlado do
algoritmo em estudo:

* colunas cilíndricas -> silhuetas longas e curvas, muitas arestas de silhueta;
* grade de barras finas -> muitos volumes de sombra SOBREPOSTOS no mesmo pixel,
  que é a condição que satura contadores de stencil estreitos;
* arcos e nichos -> a câmera consegue atravessar volumes de sombra, que é a
  condição que separa z-pass de z-fail;
* chão e paredes amplos -> receptores de sombra com grande área, dominando o
  custo de preenchimento.

Todas as malhas oclusoras são FECHADAS (two-manifold) após soldagem de
vértices: é pré-requisito para extração de silhueta correta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import mathutil as mu

F = np.float32


# --------------------------------------------------------------------------- #
# Topologia
# --------------------------------------------------------------------------- #
def weld(positions: np.ndarray, indices: np.ndarray, eps: float = 1e-4):
    """Solda vértices coincidentes. Sem isso as costuras quebram a adjacência."""
    key = np.round(positions.astype(np.float64) / eps).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    newpos = positions[first].astype(F)
    newidx = inv[indices].astype(np.int32)
    ok = ((newidx[:, 0] != newidx[:, 1])
          & (newidx[:, 1] != newidx[:, 2])
          & (newidx[:, 0] != newidx[:, 2]))
    return newpos, newidx[ok]


def topology(indices: np.ndarray):
    """Arestas orientadas e face oposta de cada uma.

    Retorna (dir_edges (3T,2), edge_face (3T,), edge_opp (3T,)).
    edge_opp[e] = -1 quando a aresta é de borda (malha aberta).
    """
    t = int(indices.shape[0])
    de = np.concatenate([indices[:, [0, 1]], indices[:, [1, 2]],
                         indices[:, [2, 0]]], axis=0).astype(np.int32)
    face = np.concatenate([np.arange(t), np.arange(t), np.arange(t)]).astype(np.int32)
    key = np.sort(de, axis=1)
    opp = np.full(de.shape[0], -1, dtype=np.int32)
    seen: dict[tuple, int] = {}
    for e in range(de.shape[0]):
        k = (int(key[e, 0]), int(key[e, 1]))
        j = seen.pop(k, None)
        if j is None:
            seen[k] = e
        else:
            opp[e] = face[j]
            opp[j] = face[e]
    return de, face, opp


def vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    v0, v1, v2 = (positions[indices[:, i]] for i in range(3))
    fn = np.cross(v1 - v0, v2 - v0)
    acc = np.zeros_like(positions)
    for i in range(3):
        np.add.at(acc, indices[:, i], fn)
    n = np.linalg.norm(acc, axis=1, keepdims=True)
    return (acc / np.maximum(n, 1e-9)).astype(F)


@dataclass
class Solid:
    """Malha fechada com topologia pré-computada."""
    positions: np.ndarray
    indices: np.ndarray
    normals: np.ndarray
    dir_edges: np.ndarray
    edge_face: np.ndarray
    edge_opp: np.ndarray

    @property
    def tri_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def closed(self) -> bool:
        return bool(np.all(self.edge_opp >= 0))


def make_solid(positions, indices) -> Solid:
    p, i = weld(np.asarray(positions, F), np.asarray(indices, np.int32))
    de, ef, eo = topology(i)
    return Solid(p, i, vertex_normals(p, i), de, ef, eo)


@dataclass
class Prop:
    """Instância posicionada de um sólido."""
    solid: Solid
    model: np.ndarray = field(default_factory=mu.identity)
    tint: tuple = (0.75, 0.73, 0.70)
    occluder: bool = True
    texture: str = "bricks"
    name: str = "prop"


# --------------------------------------------------------------------------- #
# Primitivas fechadas
# --------------------------------------------------------------------------- #
def box(sx=1.0, sy=1.0, sz=1.0) -> Solid:
    h = np.array([sx, sy, sz], dtype=F) * 0.5
    p = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=F) * h
    idx = np.array([
        [0, 2, 1], [0, 3, 2],   # -Z
        [4, 5, 6], [4, 6, 7],   # +Z
        [0, 1, 5], [0, 5, 4],   # -Y
        [3, 7, 6], [3, 6, 2],   # +Y
        [0, 4, 7], [0, 7, 3],   # -X
        [1, 2, 6], [1, 6, 5],   # +X
    ], dtype=np.int32)
    return make_solid(p, idx)


def cylinder(segments: int = 14, radius: float = 0.5, height: float = 1.0) -> Solid:
    segments = max(4, int(segments))
    ang = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=F)
    cx, cz = np.cos(ang) * radius, np.sin(ang) * radius
    hy = height * 0.5
    low = np.stack([cx, np.full_like(cx, -hy), cz], axis=1)
    up = np.stack([cx, np.full_like(cx, hy), cz], axis=1)
    p = np.concatenate([low, up,
                        np.array([[0, -hy, 0], [0, hy, 0]], dtype=F)], axis=0)
    cb, ct = 2 * segments, 2 * segments + 1
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        a, b, c, d = i, j, segments + i, segments + j
        tris += [[a, c, b], [b, c, d]]      # lateral (normal externa radial)
        tris += [[cb, a, b]]                # tampa inferior (normal -Y)
        tris += [[ct, d, c]]                # tampa superior (normal +Y)
    return make_solid(p, np.array(tris, dtype=np.int32))


def uv_sphere(stacks: int = 10, slices: int = 14, radius: float = 0.5) -> Solid:
    stacks, slices = max(3, int(stacks)), max(4, int(slices))
    pts = [[0.0, radius, 0.0]]
    for i in range(1, stacks):
        phi = np.pi * i / stacks
        for j in range(slices):
            th = 2.0 * np.pi * j / slices
            pts.append([np.sin(phi) * np.cos(th) * radius,
                        np.cos(phi) * radius,
                        np.sin(phi) * np.sin(th) * radius])
    pts.append([0.0, -radius, 0.0])
    p = np.array(pts, dtype=F)
    south = len(pts) - 1
    tris = []
    for j in range(slices):
        tris.append([0, 1 + (j + 1) % slices, 1 + j])
    for i in range(stacks - 2):
        base = 1 + i * slices
        nxt = base + slices
        for j in range(slices):
            j2 = (j + 1) % slices
            tris += [[base + j, base + j2, nxt + j],
                     [base + j2, nxt + j2, nxt + j]]
    last = 1 + (stacks - 2) * slices
    for j in range(slices):
        tris.append([south, last + j, last + (j + 1) % slices])
    return make_solid(p, np.array(tris, dtype=np.int32))


def torus_solid(seg_u: int = 16, seg_v: int = 9, r_major=0.5, r_minor=0.17) -> Solid:
    seg_u, seg_v = max(4, int(seg_u)), max(4, int(seg_v))
    pts, tris = [], []
    for i in range(seg_u):
        u = 2.0 * np.pi * i / seg_u
        cu, su = np.cos(u), np.sin(u)
        for j in range(seg_v):
            v = 2.0 * np.pi * j / seg_v
            cv, sv = np.cos(v), np.sin(v)
            pts.append([cu * (r_major + r_minor * cv), r_minor * sv,
                        su * (r_major + r_minor * cv)])
    for i in range(seg_u):
        for j in range(seg_v):
            a = i * seg_v + j
            b = i * seg_v + (j + 1) % seg_v
            c = ((i + 1) % seg_u) * seg_v + j
            d = ((i + 1) % seg_u) * seg_v + (j + 1) % seg_v
            tris += [[a, b, c], [b, d, c]]
    return make_solid(np.array(pts, dtype=F),
                      np.array(tris, dtype=np.int32))


# --------------------------------------------------------------------------- #
# Montagem da cripta
# --------------------------------------------------------------------------- #
def build_crypt(detail: float = 1.0) -> list[Prop]:
    """Monta a cripta. `detail` escala a densidade das malhas curvas."""
    d = float(np.clip(detail, 0.35, 2.5))
    seg = max(6, int(14 * d))

    col = cylinder(seg, radius=0.30, height=3.0)
    cap = box(0.80, 0.22, 0.80)
    base_blk = box(0.90, 0.26, 0.90)
    beam = box(9.2, 0.38, 0.52)
    bar = box(0.075, 2.1, 0.075)
    floor = box(14.0, 0.40, 14.0)
    wall = box(0.5, 5.0, 14.0)
    pedestal = box(1.1, 0.7, 1.1)
    head = uv_sphere(max(5, int(9 * d)), max(6, int(13 * d)), 0.33)
    ring = torus_solid(max(8, int(16 * d)), max(5, int(9 * d)), 0.52, 0.14)
    body = cylinder(max(6, int(10 * d)), radius=0.26, height=1.15)

    props: list[Prop] = [
        Prop(floor, mu.translate(0, -1.60, 0), (0.52, 0.50, 0.47),
             occluder=False, texture="checker", name="chao"),
        Prop(wall, mu.translate(-5.2, 1.0, 0), (0.46, 0.44, 0.42),
             occluder=False, texture="bricks", name="parede_oeste"),
        Prop(wall, mu.translate(5.2, 1.0, 0), (0.46, 0.44, 0.42),
             occluder=False, texture="bricks", name="parede_leste"),
    ]

    # Duas fileiras de colunas, com base e capitel identificados univocamente
    for k, z in enumerate(np.linspace(-4.0, 4.0, 5)):
        for x in (-2.7, 2.7):
            props.append(Prop(col, mu.translate(x, 0.0, float(z)),
                              (0.80, 0.78, 0.72), True, "bricks", f"coluna_{k}_{x:+.0f}"))
            props.append(Prop(base_blk, mu.translate(x, -1.26, float(z)),
                              (0.70, 0.68, 0.64), True, "bricks", f"base_{k}_{x:+.0f}"))
            props.append(Prop(cap, mu.translate(x, 1.58, float(z)),
                              (0.82, 0.80, 0.74), True, "bricks", f"capitel_{k}_{x:+.0f}"))

    # Arquitraves longitudinais
    for x in (-2.7, 2.7):
        props.append(Prop(beam,
                          mu.translate(x, 1.88, 0.0) @ mu.rotate_y(np.pi / 2),
                          (0.74, 0.72, 0.68), True, "bricks", f"viga_{x:+.0f}"))

    # Grade: barras finas e próximas -> alta sobreposição de volumes de sombra
    for i, x in enumerate(np.linspace(-1.9, 1.9, 11)):
        props.append(Prop(bar, mu.translate(float(x), 0.25, -4.9),
                          (0.38, 0.36, 0.34), True, "noise", f"grade_{i}"))

    # Estátua central
    props.append(Prop(pedestal, mu.translate(0, -1.05, 0.6),
                      (0.66, 0.64, 0.60), True, "gradient", "pedestal"))
    props.append(Prop(body, mu.translate(0, -0.12, 0.6),
                      (0.78, 0.76, 0.70), True, "gradient", "estatua_corpo"))
    props.append(Prop(head, mu.translate(0, 0.72, 0.6),
                      (0.84, 0.82, 0.76), True, "gradient", "estatua_cabeca"))
    props.append(Prop(ring, mu.translate(0, 0.62, 0.6) @ mu.rotate_x(0.9),
                      (0.88, 0.74, 0.42), True, "gradient", "halo"))
    return props


def light_position(t: float) -> np.ndarray:
    """Lâmpada que oscila no corredor central entre as colunas. Determinística em t.

    As fileiras de colunas/capitéis começam em |x| = 2.30 e as vigas ocupam
    |x| in [2.44, 2.96], y in [1.69, 2.07]. A amplitude horizontal (1.95) e a
    cota vertical (2.40 ± 0.35) mantêm a luz estritamente fora das malhas sólidas.
    """
    return np.array([1.95 * np.sin(0.21 * t),
                     2.40 + 0.35 * np.sin(0.37 * t),
                     3.10 * np.cos(0.17 * t)], dtype=F)


def scene_tri_count(props: list[Prop]) -> int:
    return int(sum(p.solid.tri_count for p in props))
