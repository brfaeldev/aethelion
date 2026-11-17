"""Camada de abstração de backends (CPU e GPU) e presets de desempenho/qualidade."""
from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import numpy as np

from .sv_renderer import FastCPURenderer, SVRenderer, SVSettings, SVStats

BACKENDS: tuple[str, ...] = ("cpu-sw", "cpu-fast", "gpu-gl")

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


@runtime_checkable
class RendererProtocol(Protocol):
    backend_name: str
    settings: SVSettings
    stats: SVStats
    camera_eye: np.ndarray
    camera_target: np.ndarray
    light: np.ndarray
    shadow_mask: np.ndarray
    visible_mask: np.ndarray
    capture_gbuffer: bool

    def render(self, t: float = 0.0) -> np.ndarray: ...
    def sync(self) -> None: ...
    def rebuild_scene(self) -> None: ...
    def render_volume_overlay(self, base_img: np.ndarray | None = None) -> np.ndarray: ...


def available_backends() -> list[str]:
    """Lista os backends disponíveis no ambiente atual."""
    avail = ["cpu-sw", "cpu-fast"]
    try:
        from .gpu_renderer import HAS_MODERNGL
        if HAS_MODERNGL:
            avail.append("gpu-gl")
    except Exception:
        pass
    return avail


def apply_preset(settings: SVSettings, preset_name: str) -> SVSettings:
    """Aplica um perfil de desempenho/qualidade em `SVSettings`."""
    if preset_name not in PRESETS:
        raise ValueError(
            f"Preset desconhecido '{preset_name}'. Opções: {list(PRESETS)}"
        )
    for k, v in PRESETS[preset_name].items():
        setattr(settings, k, v)
    return settings


def create_renderer(
    backend: str | None = None,
    settings: SVSettings | None = None,
    *,
    fallback: bool = True,
) -> RendererProtocol:
    """Fábrica unificada de renderizadores (`cpu-sw`, `cpu-fast`, `gpu-gl`)."""
    s = settings or SVSettings()
    chosen = (backend or s.backend or "cpu-sw").lower()
    s.backend = chosen

    if chosen == "cpu-sw":
        return SVRenderer(s)
    if chosen == "cpu-fast":
        return FastCPURenderer(s)
    if chosen == "gpu-gl":
        try:
            from .gpu_renderer import GLRenderer
            return GLRenderer(s)
        except Exception as exc:
            if not fallback:
                raise
            warnings.warn(
                f"Falha ao iniciar backend 'gpu-gl' ({exc}); usando 'cpu-fast'.",
                RuntimeWarning,
                stacklevel=2,
            )
            s.backend = "cpu-fast"
            return FastCPURenderer(s)
    raise ValueError(
        f"Backend desconhecido '{chosen}'. Opções válidas: {BACKENDS}"
    )
