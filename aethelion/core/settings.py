"""Configurações, estatísticas de telemetria e presets do motor Aethelion."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

F = np.float32
BACKEND = "cpu-sw"
BACKENDS: tuple[str, ...] = ("cpu-sw", "cpu-fast", "gpu-gl")

BAYER4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=F) / F(16.0)

PRESETS: dict[str, dict] = {
    "fast": {
        "width": 160,
        "height": 120,
        "detail": 0.6,
        "textured": True,
        "shading": "flat",
        "texture_filter": "nearest",
        "color_bits": 8,
        "dither": False,
        "vertex_snap": 0,
    },
    "balanced": {
        "width": 256,
        "height": 192,
        "detail": 1.0,
        "textured": True,
        "shading": "gouraud",
        "texture_filter": "nearest",
        "color_bits": 8,
        "dither": False,
        "vertex_snap": 0,
    },
    "retro-ps1": {
        "width": 320,
        "height": 240,
        "precision": "fixed12_4",
        "vertex_snap": 1,
        "color_bits": 5,
        "dither": True,
        "texture_filter": "nearest",
        "textured": True,
        "shading": "gouraud",
    },
    "high": {
        "width": 512,
        "height": 384,
        "detail": 1.5,
        "texture_filter": "bilinear",
        "color_bits": 8,
        "dither": False,
        "vertex_snap": 0,
        "textured": True,
        "shading": "gouraud",
    },
}


@dataclass
class SVSettings:
    """Parâmetros do pipeline de renderização e geração de sombras volumétricas."""

    # Framebuffer
    width: int = 256
    height: int = 192
    # Algoritmo de sombra
    algorithm: str = "zfail"        # none | zpass | zfail
    stencil_bits: int = 8
    stencil_wrap: bool = True
    extrude_len: float = 40.0
    infinite_extrude: bool = False  # Everitt & Kilgard (2002): P_inf + extrusão homogênea w=0
    depth_clamp: bool = True        # NV_depth_clamp: satura z_ndc <= 1.0 em z-fail
    # Arquitetura do buffer de profundidade
    depth_bits: int = 0             # 0 = float32; senão quantiza em N bits por fragmento
    near: float = 0.15
    far: float = 60.0
    fov: float = 62.0
    # Precisão e sombreamento
    precision: str = "float32"
    vertex_snap: int = 0
    shading: str = "gouraud"        # gouraud | flat | unlit
    textured: bool = True
    texture_res: int = 64
    texture_filter: str = "nearest"
    color_bits: int = 8
    dither: bool = False
    # Cena e execução
    detail: float = 1.0
    ambient: float = 0.17
    display_filter: str = "nearest"  # nearest | bilinear (escala na janela)
    backend: str = "cpu-sw"          # cpu-sw | cpu-fast | gpu-gl

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SVSettings":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class SVStats:
    """Métricas de tempo, geometria, preenchimento e memória de um quadro."""

    ms_geometry: float = 0.0
    ms_silhouette: float = 0.0
    ms_stencil: float = 0.0
    ms_composite: float = 0.0
    ms_total: float = 0.0
    tris_scene: int = 0
    tris_scene_drawn: int = 0
    tris_volume: int = 0
    tris_volume_drawn: int = 0
    frag_scene: int = 0
    frag_stencil: int = 0
    pixels_visible: int = 0
    pixels_shadow: int = 0
    volumes_with_camera: int = 0
    stencil_saturation: int = 0
    stencil_wrap_events: int = 0
    stencil_max: int = 0
    mem_buffers_kb: float = 0.0
    mem_geometry_kb: float = 0.0
    mem_total_kb: float = 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.ms_total if self.ms_total > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fps"] = self.fps
        return d


def apply_preset(settings: SVSettings, preset_name: str) -> SVSettings:
    """Aplica um perfil de desempenho/qualidade em `SVSettings`."""
    if preset_name not in PRESETS:
        raise ValueError(
            f"Preset desconhecido '{preset_name}'. Opções: {list(PRESETS)}"
        )
    for k, v in PRESETS[preset_name].items():
        setattr(settings, k, v)
    return settings
