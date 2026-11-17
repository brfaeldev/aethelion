"""Aethelion — estudo de sombras volumétricas por stencil em rasterizador próprio.

Eixo 7: arquitetura de consoles antigos. O objeto de estudo é o efeito dos
recursos e limites do pipeline (stencil, profundidade, precisão, recorte) sobre
a robustez, o consumo de memória e o custo de um algoritmo clássico de sombras.

Subpacotes:
  * aethelion.core        — motor gráfico e estágios do pipeline (v2)
  * aethelion.scenes      — cena procedural two-manifold da cripta (v2)
  * aethelion.evaluation  — conformidade, ray casting, roteiro e relatórios (v2)
  * aethelion.ui          — aplicação interativa em Pygame com inspeção 3D (v2)
  * aethelion.legacy      — pipeline retrô do estudo anterior (v1)
"""
from . import core, evaluation, legacy, scenes, ui
from . import conformance, experiments, framebuffer, geometry, mathutil, pipeline
from . import raster, report, scenario, scene_crypt, shadow, stencil, sv_renderer, texture, truth, viewer
from .core.clipping import clip_near
from .core.precision import apply_precision, quantize_depth
from .core.stencil import StencilBuffer
from .core.sv_renderer import BACKEND, SVRenderer, SVSettings, SVStats
from .core.texture import Texture, make_textures
from .legacy.framebuffer import Framebuffer
from .legacy.pipeline import FrameStats, Renderer, RenderSettings

__version__ = "2.1.0"
__all__ = [
    "core",
    "scenes",
    "evaluation",
    "ui",
    "legacy",
    "Renderer",
    "RenderSettings",
    "FrameStats",
    "Framebuffer",
    "Texture",
    "make_textures",
    "StencilBuffer",
    "SVRenderer",
    "SVSettings",
    "SVStats",
    "BACKEND",
    "apply_precision",
    "quantize_depth",
    "clip_near",
    "geometry",
    "mathutil",
    "raster",
    "stencil",
    "scene_crypt",
    "shadow",
    "sv_renderer",
    "truth",
    "conformance",
    "scenario",
    "report",
    "viewer",
    "texture",
    "pipeline",
    "framebuffer",
    "experiments",
]
