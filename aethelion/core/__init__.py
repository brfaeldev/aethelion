"""Núcleo do motor gráfico e estágios do pipeline de sombras volumétricas."""
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
from .precision import apply_precision, quantize_depth
from .clipping import clip_near
from .raster import scan_triangle
from .stencil import FUNCS, OPS, StencilBuffer, apply_op, compare
from .texture import PATTERNS, Texture, make_textures
from .shadow import (
    build_volume,
    build_volume_homogeneous,
    extract_silhouette_segments,
    face_facing,
    point_in_volume,
)
from .sv_renderer import BACKEND, FastCPURenderer, SVRenderer, SVSettings, SVStats
from .backend import (
    BACKENDS,
    PRESETS,
    RendererProtocol,
    apply_preset,
    available_backends,
    create_renderer,
)

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
    "clip_near",
    "scan_triangle",
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
    "BACKEND",
    "BACKENDS",
    "PRESETS",
    "RendererProtocol",
    "available_backends",
    "apply_preset",
    "create_renderer",
    "SVSettings",
    "SVStats",
    "SVRenderer",
    "FastCPURenderer",
]
