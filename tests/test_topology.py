"""Testa topologia two-manifold e unicidade de instâncias em scene_crypt."""
from __future__ import annotations

from aethelion.scenes.scene_crypt import (
    box,
    build_crypt,
    cylinder,
    torus_solid,
    uv_sphere,
)


def test_primitives_are_closed_two_manifold():
    assert box(1.0, 2.0, 3.0).closed
    assert cylinder(12, 0.5, 1.5).closed
    assert uv_sphere(8, 12, 0.5).closed
    assert torus_solid(12, 8, 0.5, 0.15).closed


def test_crypt_props_closed_and_uniquely_named():
    props = build_crypt(detail=1.0)
    names = [p.name for p in props]
    assert len(names) == len(set(names)), "Todos os Props da cripta devem ter nomes únicos"
    for p in props:
        assert p.solid.closed, f"Malha {p.name} não é two-manifold fechada"
