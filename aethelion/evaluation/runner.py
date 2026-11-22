"""
Executor do Protocolo Experimental de 8 Fases (runner.py)

Conduz a execução sequencial das fases A–H (em modo headless ou integrado à
GUI), gerenciando o relógio efetivo com desconto do overhead de ray casting
e I/O, capturando imagens representativas por fase e gravando `experimento.csv`,
`resumo.csv` (com medianas, IQR, médias e desvios-padrão) e `eventos.log`.
"""
from __future__ import annotations

import csv
import os
import statistics
import time

import numpy as np

from ..core.backend import RendererProtocol, create_renderer
from ..core.settings import BACKEND, SVSettings
from .phases import CSV_FIELDS, PHASES, Phase, camera_for
from .truth import Agreement, compare_masks, diff_image, evaluate_renderer_ground_truth


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
    """Executa o roteiro de fases. Pode ser conduzido por uma GUI ou em modo headless."""

    def __init__(
        self,
        renderer: RendererProtocol,
        outdir: str = "results",
        measure_every: float = 2.5,
        phases: list[Phase] | None = None,
        deterministic: bool = False,
        fixed_dt: float = 0.5,
    ):
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
        # Tempos de I/O (capturas PNG) e de ray casting (verdade de referência) acumulados
        # para serem descontados do relógio de parede das fases.
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

    @property
    def phase(self) -> Phase:
        return self.phases[min(self.phase_index, len(self.phases) - 1)]

    @property
    def total_duration(self) -> float:
        return float(sum(p.duracao for p in self.phases))

    @property
    def elapsed_effective(self) -> float:
        if not self.active and not self.finished:
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
        backend_id = getattr(self.r, "backend_name", BACKEND)
        self.log(
            f"INICIO backend={backend_id} resolucao="
            f"{self.r.settings.width}x{self.r.settings.height} "
            f"duracao_prevista={self.total_duration:.0f}s relogio={modo_relogio}"
        )
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
        self.log(
            f"FASE {p.key} ({p.hipotese}) '{p.titulo}' "
            f"config={p.config} camera={p.camera}"
        )

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

        measure = phase_elapsed >= self._next_measure - 1e-9
        self.r.capture_gbuffer = measure
        img = self.r.render(
            t_scene if t_scene is not None else (p.t_offset + phase_elapsed)
        )

        row = self._row(elapsed, measure)
        if measure:
            self._next_measure = phase_elapsed + self.measure_every
        self.rows.append(row)
        self.frame += 1

        if p.key not in self._shot_done and u > 0.5:
            t_png0 = time.perf_counter()
            self._shot_done.add(p.key)
            self._save_png(img, f"fase_{p.key}_{p.hipotese}.png")
            dt_png = time.perf_counter() - t_png0
            self.gt_overhead += dt_png
            self.phase_gt_overhead += dt_png

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
                self._next_measure = 0.0
                self._apply_phase()
        return img

    def _row(self, elapsed: float, measure: bool) -> dict:
        s, st, p = self.r.settings, self.r.stats, self.phase
        row = {f: "" for f in CSV_FIELDS}
        row.update({
            "t_wall": round(elapsed, 3), "fase": p.key, "hipotese": p.hipotese,
            "frame": self.frame, "backend": getattr(self.r, "backend_name", BACKEND), "origem_config": self.origem,
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
            Image.fromarray(img).resize(
                (img.shape[1] * up, img.shape[0] * up),
                Image.NEAREST,
            ).save(os.path.join(self.outdir, name))
        except Exception as exc:  # pragma: no cover
            self.log(f"aviso: falha ao salvar {name}: {exc}")

    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        self.finished = True
        self.log(f"FIM quadros={self.frame} linhas={len(self.rows)}")
        self.write_outputs()

    def write_outputs(self) -> None:
        # Se a execução foi interrompida antes de percorrer todas as fases e já existe um
        # `resumo.csv` completo em disco, grava em `*_parcial.*` para não sobrescrever os
        # dados completos da bateria experimental.
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
        log_name = "eventos_parcial.log" if preserve_existing else "eventos.log"

        exp = os.path.join(self.outdir, exp_name)
        with open(exp, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            wr.writeheader()
            wr.writerows(self.rows)
        self.log(f"-> {exp}")

        res = os.path.join(self.outdir, res_name)
        num = [
            "ms_total", "fps", "ms_geometry", "ms_silhouette", "ms_stencil",
            "tris_volume", "tris_volume_drawn", "frag_stencil",
            "pixels_shadow", "pixels_visible", "volumes_with_camera",
            "stencil_saturation", "stencil_wrap_events", "stencil_max",
            "mem_buffers_kb", "mem_geometry_kb", "mem_total_kb",
        ]
        iqr_keys = ["ms_total", "fps", "ms_geometry", "ms_silhouette", "ms_stencil"]
        met = ["precision", "recall", "f1", "error_rate"]
        out = []
        for p in self.phases:
            sel = [r for r in self.rows if r["fase"] == p.key]
            if not sel:
                continue
            rec = {
                "fase": p.key, "titulo": p.titulo, "hipotese": p.hipotese,
                "backend": getattr(self.r, "backend_name", BACKEND), "quadros": len(sel),
                "algoritmo": sel[0]["algoritmo"],
                "stencil_bits": sel[0]["stencil_bits"],
                "stencil_wrap": sel[0]["stencil_wrap"],
                "depth_bits": sel[0]["depth_bits"],
                "extrude_len": sel[0]["extrude_len"],
                "infinite_extrude": sel[0].get("infinite_extrude", 0),
            }
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

        logp = os.path.join(self.outdir, log_name)
        with open(logp, "w", encoding="utf-8") as f:
            f.write("\n".join(self.events) + "\n")
        print(f"-> {logp}")


def run_headless(
    settings: SVSettings,
    outdir: str = "results",
    measure_every: float = 2.5,
    deterministic: bool = False,
    fixed_dt: float = 0.5,
) -> ExperimentRunner:
    r = create_renderer(settings.backend, settings)
    run = ExperimentRunner(
        r, outdir, measure_every,
        deterministic=deterministic, fixed_dt=fixed_dt,
    )
    run.start()
    while run.active:
        run.step()
    return run
