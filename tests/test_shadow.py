"""Testa extrusão de volumes de sombra (finita e homogênea w=0), P_inf e métricas."""
from __future__ import annotations

import numpy as np

from aethelion.core import mathutil as mu
from aethelion.core.mesh import topology, weld
from aethelion.core.precision import quantize_depth
from aethelion.core.shadow import (
    build_volume,
    build_volume_homogeneous,
    extract_silhouette_segments,
    point_in_volume,
)
from aethelion.core.sv_renderer import SVRenderer
from aethelion.evaluation.truth import compare_masks, evaluate_renderer_ground_truth


def test_shadow_volume_is_closed_and_orientable_outward(unit_box):
    b = unit_box
    light = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    sides, front, back = build_volume(
        b.positions, b.indices, b.dir_edges, b.edge_face, b.edge_opp,
        light, length=10.0, caps=True,
    )
    assert sides.shape[0] > 0
    assert front.shape[0] > 0
    assert back.shape[0] > 0
    segs = extract_silhouette_segments(
        b.positions, b.indices, b.dir_edges, b.edge_face, b.edge_opp, light
    )
    assert segs.shape[0] > 0

    closed = np.concatenate([sides, front, back], axis=0)
    # 1. Ponto logo abaixo do cubo (0, -2, 0) deve estar dentro do volume de sombra
    assert point_in_volume(np.array([0.0, -2.0, 0.0], dtype=np.float32), closed)
    # 2. Ponto acima da luz (0, 5, 0) deve estar fora do volume de sombra
    assert not point_in_volume(np.array([0.0, 5.0, 0.0], dtype=np.float32), closed)

    # 3. O volume fechado soldado deve ser 2-manifold sem bordas abertas
    verts = closed.reshape(-1, 3)
    idx = np.arange(verts.shape[0], dtype=np.int32).reshape(-1, 3)
    w_pos, w_idx = weld(verts, idx)
    de, _, opp = topology(w_idx)
    assert np.all(opp >= 0), "Volume de sombra fechado não deve conter arestas abertas"

    # 4. Volume com sinal (Teorema da Divergência) deve ser estritamente positivo,
    # comprovando que todas as faces (front, sides e back) têm normais apontando para fora.
    v0, v1, v2 = closed[:, 0], closed[:, 1], closed[:, 2]
    signed_vol6 = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))))
    assert signed_vol6 > 1.0, f"Normais invertidas no volume de sombra: {signed_vol6}"


def test_build_volume_without_caps_for_zpass(unit_box):
    b = unit_box
    light = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    sides, front, back = build_volume(
        b.positions, b.indices, b.dir_edges, b.edge_face, b.edge_opp,
        light, length=10.0, caps=False,
    )
    assert sides.shape[0] > 0
    assert front.shape[0] == 0
    assert back.shape[0] == 0


def test_infinite_homogeneous_extrusion_and_p_inf(unit_box):
    b = unit_box
    light = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    sides, front, back = build_volume_homogeneous(
        b.positions, b.indices, b.dir_edges, b.edge_face, b.edge_opp,
        light, infinite=True, caps=True,
    )
    # Tampa frontal tem w == 1; tampa traseira no infinito tem w == 0
    assert np.allclose(front[..., 3], 1.0)
    assert np.allclose(back[..., 3], 0.0)

    # Projeção de ponto no infinito (z_view = -1, w = 0) com P_inf mapeia para z_ndc <= 1.0
    p_inf = mu.perspective_infinite(60.0, 4.0 / 3.0, 0.15)
    v_inf_view = np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32)
    clip = p_inf @ v_inf_view
    z_ndc = clip[2] / clip[3]
    assert 0.999 <= z_ndc <= 1.0


def test_per_fragment_depth_quantization_and_agreement():
    z = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    zq = quantize_depth(z, 4)
    assert len(np.unique(np.round(zq, 5))) <= 16

    pred = np.array([True, True, False, False])
    ref = np.array([True, False, True, False])
    ag = compare_masks(pred, ref)
    assert ag.tp == 1 and ag.fp == 1 and ag.fn == 1 and ag.tn == 1
    assert abs(ag.precision - 0.5) < 1e-6
    assert abs(ag.recall - 0.5) < 1e-6
    assert abs(ag.f1 - 0.5) < 1e-6


def test_renderer_runs_both_finite_and_infinite_and_overlay(small_settings):
    small_settings.infinite_extrude = True
    r = SVRenderer(small_settings)
    r.capture_gbuffer = True
    img = r.render(0.5)
    assert img.shape == (72, 96, 3)
    assert r.stats.mem_total_kb > 0.0
    ov = r.render_volume_overlay(img)
    assert ov.shape == (72, 96, 3)

    # Concordância de z-fail e z-pass com câmera externa (sem wrap espúrio em 8 bits)
    ref, pred, vis = evaluate_renderer_ground_truth(r)
    ag_zfail = compare_masks(pred[vis], ref[vis])
    assert ag_zfail.f1 > 0.92
    assert r.stats.stencil_wrap_events == 0

    r.settings.algorithm = "zpass"
    r.settings.infinite_extrude = False
    r.render(0.5)
    ref, pred, vis = evaluate_renderer_ground_truth(r)
    ag_zpass = compare_masks(pred[vis], ref[vis])
    assert ag_zpass.f1 > 0.92
    assert r.stats.stencil_wrap_events == 0
