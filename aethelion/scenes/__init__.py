"""Cenas procedurais e utilitários de topologia two-manifold."""
from .scene_crypt import (
    Prop,
    Solid,
    box,
    build_crypt,
    cylinder,
    light_position,
    make_solid,
    scene_tri_count,
    topology,
    torus_solid,
    uv_sphere,
    vertex_normals,
    weld,
)

__all__ = [
    "Solid",
    "Prop",
    "weld",
    "topology",
    "vertex_normals",
    "make_solid",
    "box",
    "cylinder",
    "uv_sphere",
    "torus_solid",
    "build_crypt",
    "light_position",
    "scene_tri_count",
]
