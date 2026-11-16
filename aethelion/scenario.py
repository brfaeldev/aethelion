"""Roteiro experimental: 8 fases, uma por hipótese.

Cada fase fixa uma configuração e uma trajetória de câmera determinística,
dura um tempo de parede definido e é registrada em log. Alterações manuais
feitas durante a execução NÃO invalidam o experimento: são registradas como
eventos e marcam as linhas afetadas com origem 'manual'.
"""
from __future__ import annotations

import csv
import os
import statistics
import time
from dataclasses import dataclass

import numpy as np

from .sv_renderer import BACKEND, SVRenderer, SVSettings
from .truth import Agreement, compare_masks

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
                 "depth_bits": 0, "extrude_len": 40.0},
          "orbita", "Linha de base: câmera nunca entra em um volume de sombra."),
    Phase("B", "z-fail, câmera externa", "H2",
          12.0, {"algorithm": "zfail", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0},
          "orbita", "Mesma trajetória de A: isola o custo das tampas."),
    Phase("C", "z-pass, câmera interna", "H1",
          12.0, {"algorithm": "zpass", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0},
          "travessia", "Trajetória atravessa volumes: artefato esperado."),
    Phase("D", "z-fail, câmera interna", "H1",
          12.0, {"algorithm": "zfail", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0},
          "travessia", "Mesma trajetória de C: robustez esperada."),
    Phase("E", "stencil 3 bits sem wrap", "H3",
          12.0, {"algorithm": "zfail", "stencil_bits": 3, "stencil_wrap": False,
                 "depth_bits": 0, "extrude_len": 40.0},
          "grade", "Zona de alta sobreposição: saturação esperada."),
    Phase("F", "stencil 3 bits com wrap", "H3",
          12.0, {"algorithm": "zfail", "stencil_bits": 3, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0},
          "grade", "Mesma cena de E com INCR_WRAP/DECR_WRAP."),
    Phase("G", "profundidade de 12 bits", "H4",
          12.0, {"algorithm": "zfail", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 12, "extrude_len": 40.0},
          "rasante", "Buffer de profundidade estreito, vista rasante."),
    Phase("H", "extrusão curta", "H5",
          12.0, {"algorithm": "zfail", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 3.5},
          "orbita", "Extrusão insuficiente: sombra truncada esperada."),
]

HIPOTESES = {
    "H1": "z-pass e z-fail concordam enquanto a câmera está fora dos volumes; "
          "a divergência acompanha o número de volumes que contêm a câmera.",
    "H2": "z-fail custa mais que z-pass por processar e preencher as tampas.",
    "H3": "sem operações de wrap, a saturação de um stencil estreito produz "
          "sombra errada; com wrap, o erro desaparece na mesma cena.",
    "H4": "reduzir a largura do buffer de profundidade degrada a máscara de "
          "sombra de forma distinta da degradação da visibilidade.",
    "H5": "extrusão finita insuficiente trunca o volume e reduz a revocação.",
}


# --------------------------------------------------------------------------- #
def camera_for(kind: str, u: float):
    """Trajetórias determinísticas. u in [0,1] dentro da fase."""
    tau = 2.0 * np.pi
    if kind == "orbita":
        ang = tau * u
        eye = np.array([6.6 * np.sin(ang), 1.55 + 0.5 * np.sin(2 * ang),
                        6.6 * np.cos(ang)], dtype=F)
        tgt = np.array([0.0, 0.25, 0.0], dtype=F)
    elif kind == "travessia":
        # Entra pela nave, passa rente às colunas e sai pelo fundo
        z = 7.0 - 15.0 * u
        eye = np.array([1.35 * np.sin(tau * u), 0.55 + 0.35 * np.sin(3 * tau * u),
                        z], dtype=F)
        tgt = np.array([0.0, 0.45, z - 3.0], dtype=F)
    elif kind == "grade":
        eye = np.array([-1.4 + 2.8 * u, 0.35 + 0.25 * np.sin(tau * u), -1.6],
                       dtype=F)
        tgt = np.array([0.0, 0.2, -5.1], dtype=F)
    elif kind == "rasante":
        eye = np.array([5.2 * np.cos(tau * u * 0.5), -0.95,
                        5.2 * np.sin(tau * u * 0.5)], dtype=F)
        tgt = np.array([0.0, -1.35, 0.0], dtype=F)
    else:
        eye = np.array([0.0, 1.0, 6.5], dtype=F)
        tgt = np.zeros(3, dtype=F)
    return eye, tgt


CSV_FIELDS = [
    "t_wall", "fase", "hipotese", "frame", "backend", "origem_config",
    "algoritmo", "stencil_bits", "stencil_wrap", "extrude_len", "depth_bits",
    "precisao", "largura", "altura", "ms_total", "fps", "ms_geometry",
    "ms_silhouette", "ms_stencil", "ms_composite", "tris_scene",
    "tris_scene_drawn", "tris_volume", "tris_volume_drawn", "frag_scene",
    "frag_stencil", "pixels_visible", "pixels_shadow", "volumes_with_camera",
    "stencil_saturation", "stencil_wrap_events", "stencil_max",
    "medido", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "error_rate",
]


class ExperimentRunner:
    """Executa o roteiro. Pode ser conduzido por uma GUI ou em modo headless."""

    def __init__(self, renderer: SVRenderer, outdir: str = "results",
                 measure_every: float = 2.5, phases: list[Phase] | None = None):
        self.r = renderer
        self.outdir = outdir
        self.phases = phases or PHASES
        self.measure_every = float(measure_every)
        self.rows: list[dict] = []
        self.events: list[str] = []
        self.active = False
        self.finished = False
        self.phase_index = 0
        self.frame = 0
        self.t0 = 0.0
        self.phase_t0 = 0.0
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

    def log(self, msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        rel = (time.perf_counter() - self.t0) if self.active else 0.0
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
        self._next_measure = 0.0
        self._shot_done.clear()
        self.log(f"INICIO backend={BACKEND} resolucao="
                 f"{self.r.settings.width}x{self.r.settings.height} "
                 f"duracao_prevista={self.total_duration:.0f}s")
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
        now = time.perf_counter()
        elapsed = now - self.t0
        p = self.phase
        u = float(np.clip((now - self.phase_t0) / p.duracao, 0.0, 1.0))

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
            self._shot_done.add(p.key)
            self._save_png(img, f"fase_{p.key}_{p.hipotese}.png")

        if (now - self.phase_t0) >= p.duracao:
            self.phase_index += 1
            if self.phase_index >= len(self.phases):
                self.stop()
            else:
                self.phase_t0 = now
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
            "stencil_max": st.stencil_max, "medido": int(measure),
        })
        if measure:
            ref, pred, vis = self.r.ground_truth()
            ag = compare_masks(pred[vis], ref[vis])
            self.last_agreement = ag
            from .truth import diff_image
            self.last_diff = diff_image(pred, ref, vis)
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
        exp = os.path.join(self.outdir, "experimento.csv")
        with open(exp, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            wr.writeheader()
            wr.writerows(self.rows)
        self.log(f"-> {exp}")

        res = os.path.join(self.outdir, "resumo.csv")
        num = ["ms_total", "fps", "ms_geometry", "ms_silhouette", "ms_stencil",
               "tris_volume", "tris_volume_drawn", "frag_stencil",
               "pixels_shadow", "pixels_visible", "volumes_with_camera",
               "stencil_saturation", "stencil_wrap_events", "stencil_max"]
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
                   "extrude_len": sel[0]["extrude_len"]}
            for k in num:
                vals = [float(r[k]) for r in sel if r[k] != ""]
                rec[f"mediana_{k}"] = round(statistics.median(vals), 4) if vals else ""
            msel = [r for r in sel if r["medido"] == 1 and r["f1"] != ""]
            for k in met:
                vals = [float(r[k]) for r in msel]
                rec[f"media_{k}"] = round(sum(vals) / len(vals), 5) if vals else ""
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
                 measure_every: float = 2.5) -> ExperimentRunner:
    r = SVRenderer(settings)
    run = ExperimentRunner(r, outdir, measure_every)
    run.start()
    while run.active:
        run.step()
    return run
