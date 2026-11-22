"""
Subpacote de Cenas e Primitivas 3D (scenes/__init__.py)

Reúne e exporta os geradores de sólidos geométricos fechados (two-manifold)
e a montagem procedural da cena experimental da cripta juntamente com a
trajetória determinística da fonte de luz pontual.
"""
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
