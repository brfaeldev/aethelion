"""Aethelion — estudo de sombras volumétricas por stencil em rasterizador próprio.

Eixo 7: arquitetura de consoles antigos. O objeto de estudo é o efeito dos
recursos e limites do pipeline (stencil, profundidade, precisão, recorte) sobre
a robustez, o consumo de memória e o custo de um algoritmo clássico de sombras.

Subpacotes:
  * aethelion.core        — motor gráfico e estágios do pipeline
  * aethelion.scenes      — cena procedural two-manifold da cripta
  * aethelion.evaluation  — conformidade, ray casting, roteiro e relatórios
  * aethelion.ui          — aplicação interativa em Pygame com inspeção 3D
"""
from . import cli, core, evaluation, scenes, ui
from .core.clipping import clip_near
from .core.precision import apply_precision, quantize_depth
from .core.stencil import StencilBuffer
from .core.sv_renderer import BACKEND, SVRenderer, SVSettings, SVStats
from .core.texture import Texture, make_textures

__version__ = "2.1.0"
__all__ = [
    "core",
    "scenes",
    "evaluation",
    "ui",
    "cli",
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
]
