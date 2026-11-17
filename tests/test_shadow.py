"""Testa extrusão de volumes de sombra (finita e homogênea w=0), P_inf e métricas."""
from __future__ import annotations

import numpy as np

from aethelion.core import mathutil as mu
from aethelion.core.precision import quantize_depth
from aethelion.core.shadow import (
    build_volume,
    build_volume_homogeneous,
    extract_silhouette_segments,
    point_in_volume,
)
from aethelion.core.sv_renderer import SVRenderer, SVSettings
from aethelion.evaluation.truth import compare_masks
from aethelion.scenes.scene_crypt import box, topology


def test_shadow_volume_is_closed_and_point_in_volume():
    b = box(1.0, 1.0, 1.0)
    light = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    sides, front, back = build_volume(
        b.positions, b.indices, b.dir_edges, b.edge_face, b.edge_opp,
        light, length=10.0, caps=True,
    )
    assert sides.shape[0] > 0
    assert front.shape[0] > 0
    assert back.shape[0] > 0

    closed = np.concatenate([sides, front, back], axis=0)
    # Ponto logo abaixo do cubo (0, -2, 0) deve estar dentro do volume de sombra
    assert point_in_volume(np.array([0.0, -2.0, 0.0], dtype=np.float32), closed)
    # Ponto acima da luz (0, 5, 0) deve estar fora do volume de sombra
    assert not point_in_volume(np.array([0.0, 5.0, 0.0], dtype=np.float32), closed)


def test_build_volume_without_caps_for_zpass():
    b = box(1.0, 1.0, 1.0)
    light = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    sides, front, back = build_volume(
        b.positions, b.indices, b.dir_edges, b.edge_face, b.edge_opp,
        light, length=10.0, caps=False,
    )
    assert sides.shape[0] > 0
    assert front.shape[0] == 0
    assert back.shape[0] == 0


def test_infinite_homogeneous_extrusion_and_p_inf():
    b = box(1.0, 1.0, 1.0)
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


def test_renderer_runs_both_finite_and_infinite_and_overlay():
    r = SVRenderer(SVSettings(width=64, height=48, detail=0.5, infinite_extrude=True))
    img = r.render(0.5)
    assert img.shape == (48, 64, 3)
    assert r.stats.mem_total_kb > 0.0
    ov = r.render_volume_overlay(img)
    assert ov.shape == (48, 64, 3)
