"""Camada de abstração de backends (CPU e GPU) e fábrica unificada de renderizadores."""
from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import numpy as np

from .settings import BACKENDS, PRESETS, SVSettings, SVStats, apply_preset
from .sv_renderer import FastCPURenderer, SVRenderer

__all__ = [
    "BACKENDS",
    "PRESETS",
    "RendererProtocol",
    "available_backends",
    "apply_preset",
    "create_renderer",
]


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
