"""
Núcleo do Motor Gráfico (core/__init__.py)

Agrupa e exporta as estruturas e funções fundamentais do pipeline de
renderização 3D e geração de sombras volumétricas, incluindo álgebra linear
homogênea, topologia two-manifold, recorte, rasterização, unidade de stencil,
texturas procedurais e backends de execução em CPU e GPU.
"""
from .backend import (
    RendererProtocol,
    available_backends,
    create_renderer,
)
from .base_renderer import BaseRenderer
from .clipping import clip_near
from .mathutil import (
    F,
    identity,
    look_at,
    normal_matrix,
    normalize,
    perspective,
    perspective_infinite,
    rotate_x,
    rotate_y,
    rotate_z,
    scale,
    translate,
)
from .mesh import Prop, Solid, make_solid, topology, vertex_normals, weld
from .overlay import draw_line_2d, render_volume_overlay
from .precision import apply_precision, quantize_depth
from .raster import scan_triangle, scan_triangle_fast
from .settings import BACKEND, BACKENDS, BAYER4, PRESETS, SVSettings, SVStats, apply_preset
from .shadow import (
    build_volume,
    build_volume_homogeneous,
    extract_silhouette_segments,
    face_facing,
    point_in_volume,
)
from .stencil import FUNCS, OPS, StencilBuffer, apply_op, compare
from .sv_renderer import FastCPURenderer, SVRenderer
from .texture import PATTERNS, Texture, make_textures

__all__ = [
    "F",
    "identity",
    "translate",
    "scale",
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "normalize",
    "normal_matrix",
    "look_at",
    "perspective",
    "perspective_infinite",
    "apply_precision",
    "quantize_depth",
    "Solid",
    "Prop",
    "weld",
    "topology",
    "vertex_normals",
    "make_solid",
    "clip_near",
    "scan_triangle",
    "scan_triangle_fast",
    "FUNCS",
    "OPS",
    "StencilBuffer",
    "compare",
    "apply_op",
    "PATTERNS",
    "Texture",
    "make_textures",
    "face_facing",
    "build_volume",
    "build_volume_homogeneous",
    "extract_silhouette_segments",
    "point_in_volume",
    "draw_line_2d",
    "render_volume_overlay",
    "BACKEND",
    "BACKENDS",
    "BAYER4",
    "PRESETS",
    "RendererProtocol",
    "available_backends",
    "apply_preset",
    "create_renderer",
    "SVSettings",
    "SVStats",
    "BaseRenderer",
    "SVRenderer",
    "FastCPURenderer",
]
