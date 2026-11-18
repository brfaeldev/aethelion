"""Subpacote de avaliação experimental: conformidade, ray casting, fases, varreduras e relatórios."""
from . import conformance, phases, plots, report, runner, sweeps, truth
from .conformance import TESTS as CONFORMANCE_TESTS, run as run_conformance
from .phases import CSV_FIELDS, HIPOTESES, PHASES, Phase, camera_for
from .plots import plots as build_plots
from .report import build as build_report, write_report
from .runner import ExperimentRunner, run_headless
from .sweeps import run_sv_sweeps
from .truth import (
    Agreement,
    compare_masks,
    diff_image,
    evaluate_renderer_ground_truth,
    occluder_soup,
    shadow_mask_raycast,
)

__all__ = [
    "conformance",
    "truth",
    "phases",
    "runner",
    "sweeps",
    "plots",
    "report",
    "CONFORMANCE_TESTS",
    "run_conformance",
    "Agreement",
    "compare_masks",
    "diff_image",
    "occluder_soup",
    "shadow_mask_raycast",
    "evaluate_renderer_ground_truth",
    "Phase",
    "PHASES",
    "HIPOTESES",
    "CSV_FIELDS",
    "camera_for",
    "ExperimentRunner",
    "run_headless",
    "run_sv_sweeps",
    "build_plots",
    "write_report",
    "build_report",
]
