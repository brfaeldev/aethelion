"""Aethelion — estudo de sombras volumétricas por stencil em rasterizador próprio.

Eixo 7: arquitetura de consoles antigos. O objeto de estudo é o efeito dos
recursos e limites do pipeline (stencil, profundidade, precisão, recorte) sobre
a robustez e o custo de um algoritmo clássico de sombras.
"""
from .pipeline import Renderer, RenderSettings, FrameStats
from .framebuffer import Framebuffer
from .texture import Texture, make_textures
from .stencil import StencilBuffer
from .sv_renderer import SVRenderer, SVSettings, SVStats, BACKEND
from . import geometry, mathutil, scene_crypt, shadow, truth, conformance

__version__ = "2.0.0"
__all__ = [
    "Renderer", "RenderSettings", "FrameStats", "Framebuffer",
    "Texture", "make_textures", "StencilBuffer",
    "SVRenderer", "SVSettings", "SVStats", "BACKEND",
    "geometry", "mathutil", "scene_crypt", "shadow", "truth", "conformance",
]
