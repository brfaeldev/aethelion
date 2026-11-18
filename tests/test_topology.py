"""Testa topologia two-manifold e unicidade de instâncias em scenes.crypt."""
from __future__ import annotations

import numpy as np

from aethelion.scenes import (
    box,
    build_crypt,
    cylinder,
    light_position,
    torus_solid,
    uv_sphere,
)


def _assert_orientable_outward(solid):
    assert solid.closed
    v0 = solid.positions[solid.indices[:, 0]]
    v1 = solid.positions[solid.indices[:, 1]]
    v2 = solid.positions[solid.indices[:, 2]]
    vol6 = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))))
    assert vol6 > 1e-4, f"Malha com volume assinalado não-positivo ({vol6}): normais invertidas"

    # Verifica que toda aresta compartilhada é percorrida em direções opostas
    edge_set = {(int(a), int(b)) for a, b in solid.dir_edges}
    for a, b in edge_set:
        assert (b, a) in edge_set, f"Aresta ({a}, {b}) sem meia-aresta oposta ({b}, {a})"


def test_primitives_are_closed_two_manifold():
    for s in (
        box(1.0, 2.0, 3.0),
        cylinder(12, 0.5, 1.5),
        uv_sphere(8, 12, 0.5),
        torus_solid(12, 8, 0.5, 0.15),
    ):
        _assert_orientable_outward(s)


def test_crypt_props_closed_and_uniquely_named():
    props = build_crypt(detail=1.0)
    names = [p.name for p in props]
    assert len(names) == len(set(names)), "Todos os Props da cripta devem ter nomes únicos"
    for p in props:
        assert p.solid.closed, f"Malha {p.name} não é two-manifold fechada"


def test_light_trajectory_never_penetrates_occluders():
    props = [p for p in build_crypt(detail=1.0) if p.occluder]
    bboxes = []
    for p in props:
        pos = p.solid.positions
        ph = np.concatenate([pos, np.ones((pos.shape[0], 1), dtype=np.float32)], axis=1)
        pw = (p.model @ ph.T).T[:, :3]
        bboxes.append((p.name, pw.min(axis=0), pw.max(axis=0)))

    for t in np.linspace(0.0, 100.0, 500):
        lp = light_position(float(t))
        for name, bmin, bmax in bboxes:
            inside = np.all((lp >= bmin - 0.05) & (lp <= bmax + 0.05))
            assert not inside, f"Luz colidiu com oclusor {name} em t={t:.2f}: {lp}"
