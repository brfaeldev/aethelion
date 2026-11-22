"""
Pacote Principal Aethelion (aethelion/__init__.py)

Centraliza e reexporta a API pública do estudo experimental de sombras
volumétricas por stencil em rasterizador próprio (Eixo 7: arquitetura de
consoles antigos). Integra o motor gráfico (`core`), geradores de cena
(`scenes`), suíte de avaliação experimental (`evaluation`), interface
interativa em Pygame (`ui`) e interface de linha de comando (`cli`).
"""
from . import cli, core, evaluation, scenes, ui
from .core.backend import (
    RendererProtocol,
    available_backends,
    create_renderer,
)
from .core.base_renderer import BaseRenderer
from .core.clipping import clip_near
from .core.mesh import Prop, Solid, make_solid, topology, vertex_normals, weld
from .core.precision import apply_precision, quantize_depth
from .core.settings import BACKEND, BACKENDS, PRESETS, SVSettings, SVStats, apply_preset
from .core.stencil import StencilBuffer
from .core.sv_renderer import FastCPURenderer, SVRenderer
from .core.texture import Texture, make_textures

__version__ = "2.2.0"
__all__ = [
    "core",
    "scenes",
    "evaluation",
    "ui",
    "cli",
    "Solid",
    "Prop",
    "weld",
    "topology",
    "vertex_normals",
    "make_solid",
    "Texture",
    "make_textures",
    "StencilBuffer",
    "BaseRenderer",
    "SVRenderer",
    "FastCPURenderer",
    "RendererProtocol",
    "available_backends",
    "apply_preset",
    "create_renderer",
    "SVSettings",
    "SVStats",
    "BACKEND",
    "BACKENDS",
    "PRESETS",
    "apply_precision",
    "quantize_depth",
    "clip_near",
]
