"""Roteiro experimental: 8 fases (uma por hipótese) e varreduras paramétricas.

Cada fase fixa uma configuração e uma trajetória de câmera determinística,
dura um tempo efetivo definido (descontando o tempo de ray casting da verdade
de referência) ou avança em passo fixo no modo determinístico (`--deterministic`),
e é registrada em log. Alterações manuais feitas durante a execução NÃO
invalidam o experimento: são registradas como eventos e marcam as linhas
afetadas com origem 'manual'.
"""
from __future__ import annotations

import csv
import os
import statistics
import time
from dataclasses import dataclass, replace

import numpy as np

from ..core.sv_renderer import BACKEND, SVRenderer, SVSettings
from .truth import Agreement, compare_masks, diff_image, evaluate_renderer_ground_truth

F = np.float32


@dataclass
class Phase:
    key: str
    titulo: str
    hipotese: str
    duracao: float
    config: dict
    camera: str
    nota: str


PHASES: list[Phase] = [
    Phase("A", "z-pass, câmera externa", "H2",
          12.0, {"algorithm": "zpass", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0, "infinite_extrude": False},
          "orbita", "Linha de base: câmera nunca entra em um volume de sombra."),
    Phase("B", "z-fail, câmera externa", "H2",
          12.0, {"algorithm": "zfail", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0, "infinite_extrude": False},
          "orbita", "Mesma trajetória de A: isola o custo das tampas."),
    Phase("C", "z-pass, câmera interna", "H1",
          12.0, {"algorithm": "zpass", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0, "infinite_extrude": False},
          "travessia", "Trajetória atravessa volumes: artefato esperado."),
    Phase("D", "z-fail, câmera interna", "H1",
          12.0, {"algorithm": "zfail", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0, "infinite_extrude": False},
          "travessia", "Mesma trajetória de C: robustez esperada."),
    Phase("E", "stencil 3 bits sem wrap", "H3",
          12.0, {"algorithm": "zfail", "stencil_bits": 3, "stencil_wrap": False,
                 "depth_bits": 0, "extrude_len": 40.0, "infinite_extrude": False},
          "grade", "Zona de alta sobreposição: saturação esperada."),
    Phase("F", "stencil 3 bits com wrap", "H3",
          12.0, {"algorithm": "zfail", "stencil_bits": 3, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0, "infinite_extrude": False},
          "grade", "Mesma cena de E com INCR_WRAP/DECR_WRAP."),
    Phase("G", "profundidade de 12 bits", "H4",
          12.0, {"algorithm": "zfail", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 12, "extrude_len": 40.0, "infinite_extrude": False},
          "rasante", "Buffer de profundidade estreito (quantizado por fragmento), vista rasante."),
    Phase("H", "extrusão curta", "H5",
          12.0, {"algorithm": "zfail", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 3.5, "infinite_extrude": False},
          "orbita", "Extrusão finita insuficiente vs. extrusão robusta: sombra truncada esperada."),
]

HIPOTESES = {
    "H1": "z-pass e z-fail concordam enquanto a câmera está fora dos volumes; "
          "a divergência acompanha o número de volumes que contêm a câmera.",
    "H2": "z-fail custa mais que z-pass por processar e preencher as tampas.",
    "H3": "sem operações de wrap, a saturação de um stencil estreito produz "
          "sombra errada; com wrap, o erro desaparece na mesma cena.",
    "H4": "reduzir a largura do buffer de profundidade degrada a máscara de "
          "sombra de forma distinta da degradação da visibilidade.",
    "H5": "extrusão finita insuficiente trunca o volume e reduz a revocação, "
          "enquanto a extrusão suficiente ou homogênea no infinito (P_inf, w=0) preserva o volume.",
}


# --------------------------------------------------------------------------- #
def camera_for(kind: str, u: float):
    """Trajetórias determinísticas. u in [0,1] dentro da fase."""
    tau = 2.0 * np.pi
    if kind == "orbita":
        # Permanece dentro das paredes laterais (|x| <= 3.8 < 5.2) e acima do
        # topo das vigas (y >= 2.18 > 2.07), garantindo volumes_with_camera == 0.
        ang = tau * u
        eye = np.array([3.8 * np.sin(ang), 2.34 + 0.16 * np.sin(2 * ang),
                        5.6 * np.cos(ang)], dtype=F)
        tgt = np.array([0.0, -0.35, 0.0], dtype=F)
    elif kind == "travessia":
        # Atravessa os cones de sombra das colunas, da estátua e da grade (volumes_with_camera > 0)
        ang = tau * u
        eye = np.array([3.35 * np.sin(ang), -0.15 + 0.25 * np.cos(2 * ang),
                        3.8 * np.cos(ang) - 1.2], dtype=F)
        tgt = np.array([0.0, -0.45, -0.5], dtype=F)
    elif kind == "grade":
        # Vista transversal ao leque das 11 barras da grade (z = -4.9): cada raio
        # cruza 8..11 volumes de sombra em série, excedendo o limite de 3 bits (max=7).
        eye = np.array([-3.65 + 0.35 * np.sin(tau * u),
                        -0.10 + 0.15 * np.cos(tau * u),
                        -5.55], dtype=F)
        tgt = np.array([2.80, -0.80, -5.65], dtype=F)
    elif kind == "rasante":
        eye = np.array([4.2 * np.cos(tau * u * 0.5), -1.05,
                        4.2 * np.sin(tau * u * 0.5)], dtype=F)
        tgt = np.array([0.0, -1.38, 0.0], dtype=F)
    else:
        eye = np.array([0.0, 1.0, 6.5], dtype=F)
        tgt = np.zeros(3, dtype=F)
    return eye, tgt


CSV_FIELDS = [
    "t_wall", "fase", "hipotese", "frame", "backend", "origem_config",
    "algoritmo", "stencil_bits", "stencil_wrap", "extrude_len",
    "infinite_extrude", "depth_bits", "precisao", "largura", "altura",
    "ms_total", "fps", "ms_geometry", "ms_silhouette", "ms_stencil",
    "ms_composite", "tris_scene", "tris_scene_drawn", "tris_volume",
    "tris_volume_drawn", "frag_scene", "frag_stencil", "pixels_visible",
    "pixels_shadow", "volumes_with_camera", "stencil_saturation",
    "stencil_wrap_events", "stencil_max", "mem_buffers_kb",
    "mem_geometry_kb", "mem_total_kb", "medido", "tp", "fp", "fn", "tn",
    "precision", "recall", "f1", "error_rate",
]


def _iqr(vals: list[float]) -> float:
    """Intervalo interquartil (Q3 - Q1)."""
    if len(vals) < 2:
        return 0.0
    if len(vals) < 4:
        return float(max(vals) - min(vals))
    qs = statistics.quantiles(vals, n=4)
    return float(qs[2] - qs[0])


def _stdev(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return float(statistics.stdev(vals))


class ExperimentRunner:
    """Executa o roteiro. Pode ser conduzido por uma GUI ou em modo headless."""

    def __init__(self, renderer: SVRenderer, outdir: str = "results",
                 measure_every: float = 2.5, phases: list[Phase] | None = None,
                 deterministic: bool = False, fixed_dt: float = 0.5):
        self.r = renderer
        self.outdir = outdir
        self.phases = phases or PHASES
        self.measure_every = float(measure_every)
        self.deterministic = bool(deterministic)
        self.fixed_dt = float(fixed_dt)
        self.rows: list[dict] = []
        self.events: list[str] = []
        self.active = False
        self.finished = False
        self.phase_index = 0
        self.frame = 0
        self.t0 = 0.0
        self.phase_t0 = 0.0
        self.gt_overhead = 0.0
        self.phase_gt_overhead = 0.0
        self.virtual_elapsed = 0.0
        self.virtual_phase_elapsed = 0.0
        self.origem = "roteiro"
        self._next_measure = 0.0
        self.last_agreement: Agreement | None = None
        self.last_diff = None
        self._shot_done: set[str] = set()
        os.makedirs(self.outdir, exist_ok=True)

    # ------------------------------------------------------------------ #
    @property
    def phase(self) -> Phase:
        return self.phases[min(self.phase_index, len(self.phases) - 1)]

    @property
    def total_duration(self) -> float:
        return float(sum(p.duracao for p in self.phases))

    @property
    def elapsed_effective(self) -> float:
        if not self.active:
            return 0.0
        if self.deterministic:
            return self.virtual_elapsed
        return max(0.0, (time.perf_counter() - self.t0) - self.gt_overhead)

    def log(self, msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        rel = self.elapsed_effective
        line = f"[{stamp}] [t={rel:7.2f}s] {msg}"
        self.events.append(line)
        print(line)

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self.active = True
        self.finished = False
        self.phase_index = 0
        self.frame = 0
        self.rows.clear()
        self.events.clear()
        self.t0 = time.perf_counter()
        self.phase_t0 = self.t0
        self.gt_overhead = 0.0
        self.phase_gt_overhead = 0.0
        self.virtual_elapsed = 0.0
        self.virtual_phase_elapsed = 0.0
        self._next_measure = 0.0
        self._shot_done.clear()
        modo_relogio = f"deterministico(dt={self.fixed_dt}s)" if self.deterministic else "parede"
        self.log(f"INICIO backend={BACKEND} resolucao="
                 f"{self.r.settings.width}x{self.r.settings.height} "
                 f"duracao_prevista={self.total_duration:.0f}s relogio={modo_relogio}")
        self._apply_phase()

    def note_manual(self, what: str) -> None:
        self.origem = "manual"
        self.log(f"OVERRIDE MANUAL: {what}")

    def _apply_phase(self) -> None:
        p = self.phase
        for k, v in p.config.items():
            setattr(self.r.settings, k, v)
        self.r.sync()
        self.origem = "roteiro"
        self.log(f"FASE {p.key} ({p.hipotese}) '{p.titulo}' "
                 f"config={p.config} camera={p.camera}")

    # ------------------------------------------------------------------ #
    def step(self, t_scene: float | None = None):
        """Renderiza e registra um quadro. Retorna a imagem."""
        if not self.active:
            return None
        p = self.phase
        if self.deterministic:
            elapsed = self.virtual_elapsed
            phase_elapsed = self.virtual_phase_elapsed
        else:
            now = time.perf_counter()
            elapsed = max(0.0, (now - self.t0) - self.gt_overhead)
            phase_elapsed = max(0.0, (now - self.phase_t0) - self.phase_gt_overhead)

        u = float(np.clip(phase_elapsed / max(1e-6, p.duracao), 0.0, 1.0))

        eye, tgt = camera_for(p.camera, u)
        self.r.camera_eye = eye
        self.r.camera_target = tgt

        measure = elapsed >= self._next_measure
        self.r.capture_gbuffer = measure
        img = self.r.render(t_scene if t_scene is not None else elapsed)

        row = self._row(elapsed, measure)
        if measure:
            self._next_measure = elapsed + self.measure_every
        self.rows.append(row)
        self.frame += 1

        if p.key not in self._shot_done and u > 0.5:
            t_png0 = time.perf_counter()
            self._shot_done.add(p.key)
            self._save_png(img, f"fase_{p.key}_{p.hipotese}.png")
            dt_png = time.perf_counter() - t_png0
            self.gt_overhead += dt_png
            self.phase_gt_overhead += dt_png

        # Atualiza o relógio APÓS o término completo do quadro e do ray casting
        if self.deterministic:
            self.virtual_elapsed += self.fixed_dt
            self.virtual_phase_elapsed += self.fixed_dt
            phase_done = self.virtual_phase_elapsed >= p.duracao - 1e-9
        else:
            now_after = time.perf_counter()
            phase_done = ((now_after - self.phase_t0) - self.phase_gt_overhead) >= p.duracao

        if phase_done:
            self.phase_index += 1
            if self.phase_index >= len(self.phases):
                self.stop()
            else:
                self.phase_t0 = time.perf_counter()
                self.phase_gt_overhead = 0.0
                self.virtual_phase_elapsed = 0.0
                self._apply_phase()
        return img

    # ------------------------------------------------------------------ #
    def _row(self, elapsed: float, measure: bool) -> dict:
        s, st, p = self.r.settings, self.r.stats, self.phase
        row = {f: "" for f in CSV_FIELDS}
        row.update({
            "t_wall": round(elapsed, 3), "fase": p.key, "hipotese": p.hipotese,
            "frame": self.frame, "backend": BACKEND, "origem_config": self.origem,
            "algoritmo": s.algorithm, "stencil_bits": s.stencil_bits,
            "stencil_wrap": int(s.stencil_wrap), "extrude_len": s.extrude_len,
            "infinite_extrude": int(s.infinite_extrude),
            "depth_bits": s.depth_bits, "precisao": s.precision,
            "largura": s.width, "altura": s.height,
            "ms_total": round(st.ms_total, 3), "fps": round(st.fps, 3),
            "ms_geometry": round(st.ms_geometry, 3),
            "ms_silhouette": round(st.ms_silhouette, 3),
            "ms_stencil": round(st.ms_stencil, 3),
            "ms_composite": round(st.ms_composite, 3),
            "tris_scene": st.tris_scene, "tris_scene_drawn": st.tris_scene_drawn,
            "tris_volume": st.tris_volume,
            "tris_volume_drawn": st.tris_volume_drawn,
            "frag_scene": st.frag_scene, "frag_stencil": st.frag_stencil,
            "pixels_visible": st.pixels_visible, "pixels_shadow": st.pixels_shadow,
            "volumes_with_camera": st.volumes_with_camera,
            "stencil_saturation": st.stencil_saturation,
            "stencil_wrap_events": st.stencil_wrap_events,
            "stencil_max": st.stencil_max,
            "mem_buffers_kb": st.mem_buffers_kb,
            "mem_geometry_kb": st.mem_geometry_kb,
            "mem_total_kb": st.mem_total_kb,
            "medido": int(measure),
        })
        if measure:
            t_gt0 = time.perf_counter()
            ref, pred, vis = evaluate_renderer_ground_truth(self.r)
            ag = compare_masks(pred[vis], ref[vis])
            self.last_agreement = ag
            self.last_diff = diff_image(pred, ref, vis)
            dt_gt = time.perf_counter() - t_gt0
            self.gt_overhead += dt_gt
            self.phase_gt_overhead += dt_gt
            row.update({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in ag.to_dict().items()})
        return row

    def _save_png(self, img, name: str) -> None:
        try:
            from PIL import Image
            up = max(1, 640 // max(1, img.shape[1]))
            Image.fromarray(img).resize((img.shape[1] * up, img.shape[0] * up),
                                        Image.NEAREST).save(
                os.path.join(self.outdir, name))
        except Exception as exc:        # pragma: no cover
            self.log(f"aviso: falha ao salvar {name}: {exc}")

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        self.finished = True
        self.log(f"FIM quadros={self.frame} linhas={len(self.rows)}")
        self.write_outputs()

    def write_outputs(self) -> None:
        phases_seen = {r["fase"] for r in self.rows}
        is_partial = len(phases_seen) < len(self.phases)
        res_default = os.path.join(self.outdir, "resumo.csv")
        preserve_existing = False
        if is_partial and os.path.exists(res_default):
            try:
                with open(res_default, newline="", encoding="utf-8") as f:
                    existing_rows = list(csv.DictReader(f))
                if len(existing_rows) >= len(self.phases):
                    preserve_existing = True
            except Exception:
                preserve_existing = False

        exp_name = "experimento_parcial.csv" if preserve_existing else "experimento.csv"
        res_name = "resumo_parcial.csv" if preserve_existing else "resumo.csv"

        exp = os.path.join(self.outdir, exp_name)
        with open(exp, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            wr.writeheader()
            wr.writerows(self.rows)
        self.log(f"-> {exp}")

        res = os.path.join(self.outdir, res_name)
        num = ["ms_total", "fps", "ms_geometry", "ms_silhouette", "ms_stencil",
               "tris_volume", "tris_volume_drawn", "frag_stencil",
               "pixels_shadow", "pixels_visible", "volumes_with_camera",
               "stencil_saturation", "stencil_wrap_events", "stencil_max",
               "mem_buffers_kb", "mem_geometry_kb", "mem_total_kb"]
        iqr_keys = ["ms_total", "fps", "ms_geometry", "ms_silhouette", "ms_stencil"]
        met = ["precision", "recall", "f1", "error_rate"]
        out = []
        for p in self.phases:
            sel = [r for r in self.rows if r["fase"] == p.key]
            if not sel:
                continue
            rec = {"fase": p.key, "titulo": p.titulo, "hipotese": p.hipotese,
                   "backend": BACKEND, "quadros": len(sel),
                   "algoritmo": sel[0]["algoritmo"],
                   "stencil_bits": sel[0]["stencil_bits"],
                   "stencil_wrap": sel[0]["stencil_wrap"],
                   "depth_bits": sel[0]["depth_bits"],
                   "extrude_len": sel[0]["extrude_len"],
                   "infinite_extrude": sel[0].get("infinite_extrude", 0)}
            for k in num:
                vals = [float(r[k]) for r in sel if r.get(k, "") != ""]
                rec[f"mediana_{k}"] = round(statistics.median(vals), 4) if vals else ""
                if k in iqr_keys:
                    rec[f"iqr_{k}"] = round(_iqr(vals), 4) if vals else ""
            msel = [r for r in sel if r["medido"] == 1 and r["f1"] != ""]
            for k in met:
                vals = [float(r[k]) for r in msel]
                rec[f"media_{k}"] = round(sum(vals) / len(vals), 5) if vals else ""
                rec[f"dp_{k}"] = round(_stdev(vals), 5) if vals else ""
            rec["quadros_medidos"] = len(msel)
            out.append(rec)
        if out:
            with open(res, "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
                wr.writeheader()
                wr.writerows(out)
            self.log(f"-> {res}")

        logp = os.path.join(self.outdir, "eventos.log")
        with open(logp, "w", encoding="utf-8") as f:
            f.write("\n".join(self.events) + "\n")
        print(f"-> {logp}")


def run_headless(settings: SVSettings, outdir: str = "results",
                 measure_every: float = 2.5,
                 deterministic: bool = False,
                 fixed_dt: float = 0.5) -> ExperimentRunner:
    r = SVRenderer(settings)
    run = ExperimentRunner(r, outdir, measure_every,
                           deterministic=deterministic, fixed_dt=fixed_dt)
    run.start()
    while run.active:
        run.step()
    return run


# --------------------------------------------------------------------------- #
# Varreduras complementares (Etapa 4: Resolução, Complexidade e Extrusão P_inf)
# --------------------------------------------------------------------------- #
def _eval_sv_config(s: SVSettings, label: str, frames: int = 5) -> dict:
    r = SVRenderer(s)
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


def run_sv_sweeps(base: SVSettings | None = None, outdir: str = "results",
                  frames: int = 4) -> dict[str, list[dict]]:
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
        print(f"  {row['variante']:>10} | {row['mediana_ms_total']:7.1f} ms "
              f"| {row['mediana_fps']:5.2f} fps | mem {row['mem_total_kb']:6.1f} KB "
              f"| F1 {row['media_f1']:.4f}")
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
        print(f"  {row['variante']:>10} | tris_vol={row['tris_volume']:5d} "
              f"| {row['mediana_ms_total']:7.1f} ms | mem {row['mem_total_kb']:6.1f} KB "
              f"| F1 {row['media_f1']:.4f}")
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
        cfg = replace(base, width=256, height=192, algorithm="zfail",
                      extrude_len=length, infinite_extrude=inf_flag)
        row = _eval_sv_config(cfg, label, frames=frames)
        ext_rows.append(row)
        print(f"  {row['variante']:>22} | recall={row['media_recall']:.4f} "
              f"| F1={row['media_f1']:.4f} | {row['mediana_ms_total']:7.1f} ms")
    p_ext = os.path.join(outdir, "varredura_sv_extrusao.csv")
    with open(p_ext, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(ext_rows[0].keys()))
        wr.writeheader()
        wr.writerows(ext_rows)
    print(f"  -> {p_ext}")
    results["extrusao"] = ext_rows

    return results
