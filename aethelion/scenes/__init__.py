"""Cenas procedurais e primitivas geométricas 3D."""
from . import crypt, primitives
from .crypt import build_crypt, light_position
from .primitives import box, cylinder, torus_solid, uv_sphere

__all__ = [
    "crypt",
    "primitives",
    "box",
    "cylinder",
    "uv_sphere",
    "torus_solid",
    "build_crypt",
    "light_position",
]
