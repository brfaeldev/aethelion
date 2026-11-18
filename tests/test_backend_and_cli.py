"""Testes da abstração de backends, presets, cache de silhuetas e novos parâmetros da CLI."""
import json
import warnings

import numpy as np

from aethelion.cli import build_parser, settings_from
from aethelion.core.backend import (
    PRESETS,
    RendererProtocol,
    apply_preset,
    available_backends,
    create_renderer,
)
from aethelion.core.sv_renderer import SVSettings


def test_cpu_sw_and_cpu_fast_equivalence():
    s_sw = SVSettings(width=96, height=72, detail=0.6, backend="cpu-sw")
    s_fast = SVSettings(width=96, height=72, detail=0.6, backend="cpu-fast")
    r_sw = create_renderer("cpu-sw", s_sw)
    r_fast = create_renderer("cpu-fast", s_fast)
    assert isinstance(r_sw, RendererProtocol)
    assert isinstance(r_fast, RendererProtocol)
    assert r_sw.backend_name == "cpu-sw"
    assert r_fast.backend_name == "cpu-fast"

    img_sw = r_sw.render(0.4)
    img_fast = r_fast.render(0.4)
    assert np.array_equal(img_sw, img_fast)
    assert np.array_equal(r_sw.shadow_mask, r_fast.shadow_mask)


def test_volume_cache_is_exact_across_frames():
    s = SVSettings(width=96, height=72, detail=0.6)
    r = create_renderer("cpu-sw", s)
    img1 = r.render(0.25)
    mask1 = r.shadow_mask.copy()
    img2 = r.render(0.25)
    mask2 = r.shadow_mask.copy()
    assert np.array_equal(img1, img2)
    assert np.array_equal(mask1, mask2)


def test_gpu_fallback_when_moderngl_missing():
    avail = available_backends()
    assert "cpu-sw" in avail and "cpu-fast" in avail
    if "gpu-gl" not in avail:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = create_renderer("gpu-gl", SVSettings(width=64, height=48), fallback=True)
            assert r.backend_name == "cpu-fast"
            assert any("gpu-gl" in str(x.message) for x in w)
    else:
        from aethelion.evaluation.scenario import camera_for
        # Verifica z-fail e saturação vs wrap de 3 bits no backend gpu-gl
        s_sat = SVSettings(width=128, height=96, algorithm="zfail", stencil_bits=3, stencil_wrap=False)
        r_gpu = create_renderer("gpu-gl", s_sat, fallback=False)
        try:
            r_gpu.camera_eye, r_gpu.camera_target = camera_for("grade", 0.5)
            r_gpu.render(54.0)
            assert r_gpu.stats.stencil_saturation > 0
            assert r_gpu.stats.stencil_wrap_events == 0

            r_gpu.settings.stencil_wrap = True
            r_gpu.render(54.0)
            assert r_gpu.stats.stencil_saturation == 0
            assert r_gpu.stats.stencil_wrap_events > 0
        finally:
            r_gpu.release()


def test_presets_and_cli_flags(tmp_path):
    for name in PRESETS:
        s = apply_preset(SVSettings(), name)
        assert s.width > 0 and s.height > 0

    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"width": 192, "height": 144, "ambient": 0.35}), encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args([
        "--config", str(cfg_file),
        "--texture-filter", "bilinear",
        "--texture-res", "32",
        "--vertex-snap", "2",
        "--fov", "75",
        "--near", "0.2",
        "--far", "80",
        "--display-filter", "bilinear",
        "--backend", "cpu-fast",
    ])
    s = settings_from(args)
    assert s.width == 192 and s.height == 144
    assert s.ambient == 0.35
    assert s.texture_filter == "bilinear"
    assert s.texture_res == 32
    assert s.vertex_snap == 2
    assert s.fov == 75.0
    assert s.near == 0.2
    assert s.far == 80.0
    assert s.display_filter == "bilinear"
    assert s.backend == "cpu-fast"
