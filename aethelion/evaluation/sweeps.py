"""
Varreduras Paramétricas de Desempenho e Robustez (sweeps.py)

Avalia sistematicamente o comportamento do pipeline sob variação de três
fatores independentes — resolução interna do framebuffer, complexidade
geométrica da cena (`detail`) e regime de extrusão (finita de Carmack vs.
infinita homogênea `P_inf, w = 0` de Everitt & Kilgard) — gerando CSVs dedicados.
"""
from __future__ import annotations

import csv
import os
import statistics
from dataclasses import replace

from ..core.backend import create_renderer
from ..core.settings import SVSettings
from .phases import camera_for
from .runner import _iqr, _stdev
from .truth import compare_masks, evaluate_renderer_ground_truth


def _eval_sv_config(s: SVSettings, label: str, frames: int = 5) -> dict:
    r = create_renderer(s.backend, s)
    r.capture_gbuffer = False
    eye0, tgt0 = camera_for("orbita", 0.0)
    r.camera_eye, r.camera_target = eye0, tgt0
    r.render(t=0.0)  # Quadro de warm-up para estabilizar caches e JIT
    r.capture_gbuffer = True
    ms_list, fps_list, f1_list, prec_list, rec_list = [], [], [], [], []
    for i in range(frames):
        u = (i + 0.5) / max(1, frames)
        eye, tgt = camera_for("orbita", u)
        r.camera_eye, r.camera_target = eye, tgt
        r.render(t=u * 6.0)
        st = r.stats
        ms_list.append(st.ms_total)
        fps_list.append(st.fps)
        ref, pred, vis = evaluate_renderer_ground_truth(r)
        ag = compare_masks(pred[vis], ref[vis])
        f1_list.append(ag.f1)
        prec_list.append(ag.precision)
        rec_list.append(ag.recall)
    st = r.stats
    if hasattr(r, "release"):
        r.release()
    return {
        "variante": label,
        "largura": s.width,
        "altura": s.height,
        "detail": s.detail,
        "algoritmo": s.algorithm,
        "extrude_len": "inf(w=0)" if s.infinite_extrude else s.extrude_len,
        "tris_scene": st.tris_scene,
        "tris_volume": st.tris_volume,
        "frag_stencil": st.frag_stencil,
        "mem_buffers_kb": st.mem_buffers_kb,
        "mem_geometry_kb": st.mem_geometry_kb,
        "mem_total_kb": st.mem_total_kb,
        "mediana_ms_total": round(statistics.median(ms_list), 3),
        "iqr_ms_total": round(_iqr(ms_list), 3),
        "mediana_fps": round(statistics.median(fps_list), 3),
        "media_precision": round(statistics.mean(prec_list), 5),
        "media_recall": round(statistics.mean(rec_list), 5),
        "media_f1": round(statistics.mean(f1_list), 5),
        "dp_f1": round(_stdev(f1_list), 5),
    }


def run_sv_sweeps(
    base: SVSettings | None = None,
    outdir: str = "results",
    frames: int = 4,
) -> dict[str, list[dict]]:
    """Executa varreduras de resolução, complexidade da cripta e extrusão P_inf."""
    base = base or SVSettings()
    os.makedirs(outdir, exist_ok=True)
    results: dict[str, list[dict]] = {}

    # 1. Varredura de Resolução
    res_rows = []
    print("\n=== Varredura SV: Resolução ===")
    for w, h in [(160, 120), (256, 192), (320, 240), (400, 300)]:
        cfg = replace(base, width=w, height=h, algorithm="zfail", detail=1.0)
        row = _eval_sv_config(cfg, f"{w}x{h}", frames=frames)
        res_rows.append(row)
        print(
            f"  {row['variante']:>10} | {row['mediana_ms_total']:7.1f} ms "
            f"| {row['mediana_fps']:5.2f} fps | mem {row['mem_total_kb']:6.1f} KB "
            f"| F1 {row['media_f1']:.4f}"
        )
    p_res = os.path.join(outdir, "varredura_sv_resolucao.csv")
    with open(p_res, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(res_rows[0].keys()))
        wr.writeheader()
        wr.writerows(res_rows)
    print(f"  -> {p_res}")
    results["resolucao"] = res_rows

    # 2. Varredura de Complexidade Geométrica (detail)
    det_rows = []
    print("\n=== Varredura SV: Complexidade Geométrica (detail) ===")
    for d in [0.5, 1.0, 1.5, 2.0]:
        cfg = replace(base, width=256, height=192, algorithm="zfail", detail=d)
        row = _eval_sv_config(cfg, f"detail={d:.1f}", frames=frames)
        det_rows.append(row)
        print(
            f"  {row['variante']:>10} | tris_vol={row['tris_volume']:5d} "
            f"| {row['mediana_ms_total']:7.1f} ms | mem {row['mem_total_kb']:6.1f} KB "
            f"| F1 {row['media_f1']:.4f}"
        )
    p_det = os.path.join(outdir, "varredura_sv_complexidade.csv")
    with open(p_det, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(det_rows[0].keys()))
        wr.writeheader()
        wr.writerows(det_rows)
    print(f"  -> {p_det}")
    results["complexidade"] = det_rows

    # 3. Comparação Carmack (Finito Curto / Finito Longo) vs. Everitt & Kilgard (P_inf, w=0)
    ext_rows = []
    print("\n=== Varredura SV: Extrusão Finita (Carmack) vs. Infinita Homogênea (Everitt & Kilgard) ===")
    for label, length, inf_flag in [
        ("Finita curta (L=3.5)", 3.5, False),
        ("Finita padrão (L=40)", 40.0, False),
        ("Infinita P_inf (w=0)", 40.0, True),
    ]:
        cfg = replace(
            base, width=256, height=192, algorithm="zfail",
            extrude_len=length, infinite_extrude=inf_flag,
        )
        row = _eval_sv_config(cfg, label, frames=frames)
        ext_rows.append(row)
        print(
            f"  {row['variante']:>22} | recall={row['media_recall']:.4f} "
            f"| F1={row['media_f1']:.4f} | {row['mediana_ms_total']:7.1f} ms"
        )
    p_ext = os.path.join(outdir, "varredura_sv_extrusao.csv")
    with open(p_ext, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(ext_rows[0].keys()))
        wr.writeheader()
        wr.writerows(ext_rows)
    print(f"  -> {p_ext}")
    results["extrusao"] = ext_rows

    return results
