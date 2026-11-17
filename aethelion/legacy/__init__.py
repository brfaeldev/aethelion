"""Subsistema legado (v1): pipeline retrô e varreduras do estudo anterior."""
from .experiments import SWEEPS, measure, run_all, run_sweep
from .framebuffer import BAYER4, PALETTE_16, Framebuffer
from .geometry import Instance, Mesh, build_scene, cube, ground, scene_tri_count, sphere, torus
from .pipeline import FrameStats, Renderer, RenderSettings, apply_precision, clip_near

__all__ = [
    "BAYER4",
    "PALETTE_16",
    "Framebuffer",
    "Mesh",
    "Instance",
    "cube",
    "sphere",
    "torus",
    "ground",
    "build_scene",
    "scene_tri_count",
    "RenderSettings",
    "FrameStats",
    "Renderer",
    "apply_precision",
    "clip_near",
    "SWEEPS",
    "measure",
    "run_sweep",
    "run_all",
]
