"""
Geradores de Primitivas Geométricas 3D (primitives.py)

Sintetiza malhas poligonais fechadas (two-manifold) com orientação consistente
de normais externas — paralelepípedo (`box`), cilindro (`cylinder`), esfera UV
(`uv_sphere`) e toro (`torus_solid`) — prontas para extração de silhuetas e
extrusão de volumes de sombra sem arestas de borda abertas.
"""
from __future__ import annotations

import numpy as np

from ..core.mesh import Solid, make_solid

F = np.float32


def box(sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> Solid:
    """Cubo/paralelepípedo centrado na origem com normais externas consistentes."""
    h = np.array([sx, sy, sz], dtype=F) * 0.5
    p = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=F,
    ) * h
    idx = np.array(
        [
            [0, 2, 1], [0, 3, 2],   # -Z
            [4, 5, 6], [4, 6, 7],   # +Z
            [0, 1, 5], [0, 5, 4],   # -Y
            [3, 7, 6], [3, 6, 2],   # +Y
            [0, 4, 7], [0, 7, 3],   # -X
            [1, 2, 6], [1, 6, 5],   # +X
        ],
        dtype=np.int32,
    )
    return make_solid(p, idx)


def cylinder(segments: int = 14, radius: float = 0.5, height: float = 1.0) -> Solid:
    """Cilindro fechado (two-manifold) alinhado ao eixo Y."""
    segments = max(4, int(segments))
    ang = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=F)
    cx, cz = np.cos(ang) * radius, np.sin(ang) * radius
    hy = height * 0.5
    low = np.stack([cx, np.full_like(cx, -hy), cz], axis=1)
    up = np.stack([cx, np.full_like(cx, hy), cz], axis=1)
    p = np.concatenate(
        [low, up, np.array([[0, -hy, 0], [0, hy, 0]], dtype=F)],
        axis=0,
    )
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
    """Esfera UV fechada (two-manifold) centrada na origem."""
    stacks, slices = max(3, int(stacks)), max(4, int(slices))
    pts = [[0.0, radius, 0.0]]
    for i in range(1, stacks):
        phi = np.pi * i / stacks
        for j in range(slices):
            th = 2.0 * np.pi * j / slices
            pts.append([
                np.sin(phi) * np.cos(th) * radius,
                np.cos(phi) * radius,
                np.sin(phi) * np.sin(th) * radius,
            ])
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
            tris += [
                [base + j, base + j2, nxt + j],
                [base + j2, nxt + j2, nxt + j],
            ]
    last = 1 + (stacks - 2) * slices
    for j in range(slices):
        tris.append([south, last + j, last + (j + 1) % slices])
    return make_solid(p, np.array(tris, dtype=np.int32))


def torus_solid(
    seg_u: int = 16,
    seg_v: int = 9,
    r_major: float = 0.5,
    r_minor: float = 0.17,
) -> Solid:
    """Toro fechado (two-manifold) no plano XZ."""
    seg_u, seg_v = max(4, int(seg_u)), max(4, int(seg_v))
    pts, tris = [], []
    for i in range(seg_u):
        u = 2.0 * np.pi * i / seg_u
        cu, su = np.cos(u), np.sin(u)
        for j in range(seg_v):
            v = 2.0 * np.pi * j / seg_v
            cv, sv = np.cos(v), np.sin(v)
            pts.append([
                cu * (r_major + r_minor * cv),
                r_minor * sv,
                su * (r_major + r_minor * cv),
            ])
    for i in range(seg_u):
        for j in range(seg_v):
            a = i * seg_v + j
            b = i * seg_v + (j + 1) % seg_v
            c = ((i + 1) % seg_u) * seg_v + j
            d = ((i + 1) % seg_u) * seg_v + (j + 1) % seg_v
            tris += [[a, b, c], [b, d, c]]
    return make_solid(np.array(pts, dtype=F), np.array(tris, dtype=np.int32))
