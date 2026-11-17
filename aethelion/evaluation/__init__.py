"""Metodologia experimental, verdade de referência, conformidade e relatórios."""
from .conformance import TESTS, run as run_conformance
from .report import build as build_report, plots, write_report
from .scenario import (
    CSV_FIELDS,
    HIPOTESES,
    PHASES,
    ExperimentRunner,
    Phase,
    camera_for,
    run_headless,
    run_sv_sweeps,
)
from .truth import (
    Agreement,
    compare_masks,
    diff_image,
    evaluate_renderer_ground_truth,
    occluder_soup,
    shadow_mask_raycast,
)

__all__ = [
    "Agreement",
    "occluder_soup",
    "shadow_mask_raycast",
    "evaluate_renderer_ground_truth",
    "compare_masks",
    "diff_image",
    "TESTS",
    "run_conformance",
    "Phase",
    "PHASES",
    "HIPOTESES",
    "CSV_FIELDS",
    "camera_for",
    "ExperimentRunner",
    "run_headless",
    "run_sv_sweeps",
    "plots",
    "write_report",
    "build_report",
]
