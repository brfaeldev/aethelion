"""
Testes de Backends, Presets, CLI e Execução Experimental (test_backend_and_cli.py)

Verifica a equivalência pixel a pixel entre `cpu-sw` e `cpu-fast`, a exatidão
do cache de volumes de sombra, o funcionamento e fallback do backend `gpu-gl`,
o carregamento de presets e flags da CLI, a validação defensiva de `SVSettings`
e a execução ponta a ponta do `ExperimentRunner` com geração de gráficos e relatório.
"""
import json
import warnings

import numpy as np

from aethelion.cli import build_parser, settings_from
from aethelion.core.backend import (
    RendererProtocol,
    available_backends,
    create_renderer,
)
from aethelion.core.settings import PRESETS, SVSettings, apply_preset


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


def test_volume_cache_is_exact_across_frames(small_settings):
    r = create_renderer("cpu-sw", small_settings)
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
        from aethelion.evaluation.phases import camera_for
        # Verifica z-fail e saturação vs wrap de 3 bits no backend gpu-gl
        s_sat = SVSettings(
            width=128, height=96, algorithm="zfail",
            stencil_bits=3, stencil_wrap=False,
            precision="fixed12_4", vertex_snap=1,
        )
        r_gpu = create_renderer("gpu-gl", s_sat, fallback=False)
        try:
            r_gpu.camera_eye, r_gpu.camera_target = camera_for("grade", 0.5)
            r_gpu.render(54.0)
            assert r_gpu.stats.pixels_visible > 0
            assert 0 <= r_gpu.stats.pixels_shadow <= r_gpu.stats.pixels_visible
            assert r_gpu.stats.stencil_saturation > 0
            assert r_gpu.stats.stencil_wrap_events == 0

            r_gpu.settings.stencil_wrap = True
            r_gpu.settings.depth_bits = 32
            r_gpu.settings.color_bits = 5
            r_gpu.settings.dither = True
            r_gpu.render(54.0)
            assert r_gpu.stats.stencil_saturation == 0
            assert r_gpu.stats.stencil_wrap_events > 0
            assert r_gpu.stats.frag_scene == r_gpu.stats.pixels_visible
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


def test_componentized_modules_and_custom_scene_injection(unit_box):
    from aethelion.core.base_renderer import BaseRenderer
    from aethelion.core.mesh import Prop
    from aethelion.core.sv_renderer import SVRenderer
    from aethelion.ui.camera import OrbitCamera
    from aethelion.ui.widgets import UIPanelCanvas

    # 1. Renderizador aceita cena customizada injetada sem depender de crypt
    custom_scene = lambda detail: [Prop(unit_box, name="cubo_unico")]
    custom_light = lambda t: np.array([0.0, 3.0, 2.0], dtype=np.float32)
    r = SVRenderer(SVSettings(width=64, height=48), scene_builder=custom_scene, light_fn=custom_light)
    assert isinstance(r, BaseRenderer)
    img = r.render(0.0)
    assert img.shape == (48, 64, 3)
    assert r.stats.tris_scene == 12

    # 2. OrbitCamera independente de Pygame window
    cam = OrbitCamera(yaw=0.0, pitch=0.0, dist=5.0)
    cam.orbit(100.0, 50.0)
    cam.zoom(2.0)
    cam.apply_to(r)
    assert np.allclose(r.camera_eye, cam.eye)

    # 3. UIPanelCanvas gerencia scroll
    canvas = UIPanelCanvas()
    canvas.content_h = 800
    canvas.handle_wheel(-2, view_h=400)
    assert 0 < canvas.scroll <= 400
    canvas.clamp_scroll(view_h=900)
    assert canvas.scroll == 0


def test_settings_validation_and_experiment_runner_e2e(tmp_path, unit_box):
    from aethelion.core.mesh import Prop
    from aethelion.core.sv_renderer import SVRenderer
    from aethelion.evaluation.phases import Phase
    from aethelion.evaluation.report import write_report
    from aethelion.evaluation.runner import ExperimentRunner

    # 1. Validação defensiva de limites e enums em SVSettings
    s_bad = SVSettings(
        width=2, height=-10, stencil_bits=99, color_bits=0,
        depth_bits=-4, near=-1.0, far=0.0, fov=500.0, detail=99.0, ambient=5.0,
        algorithm="invalido", precision="fp8", shading="phong",
        texture_filter="trilinear", display_filter="lanczos", backend="vulkan",
    )
    assert s_bad.width == 16 and s_bad.height == 16
    assert s_bad.stencil_bits == 8 and s_bad.color_bits == 1
    assert s_bad.depth_bits == 0
    assert s_bad.near >= 1e-3 and s_bad.far > s_bad.near
    assert s_bad.fov == 160.0 and s_bad.detail == 4.0 and s_bad.ambient == 1.0
    assert s_bad.algorithm == "zfail"
    assert s_bad.precision == "float32"
    assert s_bad.shading == "gouraud"
    assert s_bad.texture_filter == "nearest"
    assert s_bad.display_filter == "nearest"
    assert s_bad.backend == "cpu-sw"

    # 2. Execução ponta a ponta de ExperimentRunner + plots + write_report
    from aethelion.evaluation.plots import plots
    custom_scene = lambda detail: [Prop(unit_box, name="cubo_teste")]
    custom_light = lambda t: np.array([0.0, 3.0, 2.0], dtype=np.float32)
    r = SVRenderer(
        SVSettings(width=48, height=36, color_bits=5, dither=True, precision="fixed16_16"),
        scene_builder=custom_scene,
        light_fn=custom_light,
    )
    phases = [
        Phase("A", "z-pass curto", "H2", 1.0, {"algorithm": "zpass"}, "orbita", ""),
        Phase("B", "z-fail curto", "H2", 1.0, {"algorithm": "zfail"}, "orbita", ""),
    ]
    runner = ExperimentRunner(
        r, outdir=str(tmp_path), measure_every=0.5,
        phases=phases, deterministic=True, fixed_dt=0.5,
    )
    runner.start()
    while runner.active:
        runner.step()

    assert runner.finished
    assert abs(runner.elapsed_effective - 2.0) < 1e-6
    assert (tmp_path / "experimento.csv").exists()
    assert (tmp_path / "resumo.csv").exists()
    assert (tmp_path / "eventos.log").exists()
    log_txt = (tmp_path / "eventos.log").read_text(encoding="utf-8")
    assert "[t=   2.00s] FIM" in log_txt
    made_plots = plots(str(tmp_path))
    assert len(made_plots) >= 4
    rep_path = write_report(str(tmp_path))
    assert (tmp_path / "RELATORIO.md").exists()
    assert "Relatório Experimental" in open(rep_path, encoding="utf-8").read()


