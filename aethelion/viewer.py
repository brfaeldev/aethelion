"""Aplicação de tela única (pygame): viewport, HUD e barra de status.

Tudo em uma janela: a cena à esquerda, os indicadores à direita e a barra de
estado embaixo. O experimento é iniciado com ESPAÇO e segue o roteiro de fases;
as teclas de override alteram a configuração em tempo real e cada alteração é
registrada em `eventos.log` com o instante e o quadro.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from .scenario import ExperimentRunner, PHASES
from .sv_renderer import BACKEND, SVRenderer, SVSettings

PANEL_W = 310
STATUS_H = 30
VIEW_W, VIEW_H = 760, 570

BG = (22, 22, 28)
PANEL_BG = (32, 32, 42)
FG = (232, 232, 240)
ACC = (143, 208, 255)
OK = (120, 220, 140)
WARN = (240, 170, 70)
BAD = (232, 92, 80)

RES_CYCLE = [(160, 120), (192, 144), (256, 192), (320, 240), (400, 300)]
BITS_CYCLE = [8, 5, 4, 3, 2, 1]
DEPTH_CYCLE = [0, 24, 16, 12, 10]
PREC_CYCLE = ["float32", "float16", "fixed16_16", "fixed12_4"]

HELP = [
    ("ESPACO", "iniciar / parar experimento"),
    ("1 / 2 / 0", "z-pass / z-fail / sem sombra"),
    ("W", "alterna INCR_WRAP"),
    ("B", "largura do stencil"),
    ("D", "largura do z-buffer"),
    ("P", "precisão numérica"),
    ("R", "resolução interna"),
    ("E", "comprimento de extrusão"),
    ("V", "vista: cena / stencil / diferença"),
    ("G", "medir contra referência agora"),
    ("C", "suíte de conformidade"),
    ("S", "screenshot"),
    ("setas", "orbitar (fora do roteiro)"),
    ("ESC", "sair"),
]


class Viewer:
    def __init__(self, settings: SVSettings | None = None,
                 outdir: str = "results"):
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption(
            "Aethelion — Sombras Volumétricas por Stencil (Eixo 7)")
        self.screen = pygame.display.set_mode(
            (VIEW_W + PANEL_W, VIEW_H + STATUS_H))
        self.font = self._font(13)
        self.font_b = self._font(13, bold=True)
        self.font_s = self._font(11)

        self.r = SVRenderer(settings or SVSettings())
        self.run = ExperimentRunner(self.r, outdir)
        self.outdir = outdir
        self.view_mode = 0           # 0 cena, 1 stencil, 2 diferença
        self.free_yaw, self.free_pitch, self.free_dist = 0.5, 0.18, 7.0
        self._apply_free_camera()
        self.clock = pygame.time.Clock()
        self.t_scene = 0.0
        self.last = time.perf_counter()
        self.conf_result = None
        self.alive = True

    def _font(self, size, bold=False):
        for name in ("consolas", "dejavusansmono", "couriernew", None):
            try:
                return self.pg.font.SysFont(name, size, bold=bold)
            except Exception:
                continue
        return self.pg.font.Font(None, size)

    # ------------------------------------------------------------------ #
    def _apply_free_camera(self):
        cp = np.cos(self.free_pitch)
        self.r.camera_eye = np.array([
            self.free_dist * cp * np.sin(self.free_yaw),
            self.free_dist * np.sin(self.free_pitch) + 0.8,
            self.free_dist * cp * np.cos(self.free_yaw)], dtype=np.float32)
        self.r.camera_target = np.array([0.0, 0.25, 0.0], dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _cycle(self, attr, values, label):
        cur = getattr(self.r.settings, attr)
        try:
            nxt = values[(values.index(cur) + 1) % len(values)]
        except ValueError:
            nxt = values[0]
        setattr(self.r.settings, attr, nxt)
        self.r.sync()
        if self.run.active:
            self.run.note_manual(f"{label} -> {nxt}")
        return nxt

    def _key(self, k):
        pg, s = self.pg, self.r.settings
        if k == pg.K_ESCAPE:
            self.alive = False
        elif k == pg.K_SPACE:
            if self.run.active:
                self.run.stop()
            else:
                self.run.start()
        elif k == pg.K_1:
            s.algorithm = "zpass"
            self.run.active and self.run.note_manual("algoritmo -> zpass")
        elif k == pg.K_2:
            s.algorithm = "zfail"
            self.run.active and self.run.note_manual("algoritmo -> zfail")
        elif k == pg.K_0:
            s.algorithm = "none"
            self.run.active and self.run.note_manual("algoritmo -> none")
        elif k == pg.K_w:
            s.stencil_wrap = not s.stencil_wrap
            self.r.sync()
            self.run.active and self.run.note_manual(f"wrap -> {s.stencil_wrap}")
        elif k == pg.K_b:
            self._cycle("stencil_bits", BITS_CYCLE, "stencil_bits")
        elif k == pg.K_d:
            self._cycle("depth_bits", DEPTH_CYCLE, "depth_bits")
        elif k == pg.K_p:
            self._cycle("precision", PREC_CYCLE, "precisao")
        elif k == pg.K_e:
            self._cycle("extrude_len", [40.0, 15.0, 7.0, 3.5, 80.0], "extrusao")
        elif k == pg.K_r:
            cur = (s.width, s.height)
            try:
                nxt = RES_CYCLE[(RES_CYCLE.index(cur) + 1) % len(RES_CYCLE)]
            except ValueError:
                nxt = RES_CYCLE[0]
            s.width, s.height = nxt
            self.r.sync()
            self.run.active and self.run.note_manual(f"resolucao -> {nxt}")
        elif k == pg.K_v:
            self.view_mode = (self.view_mode + 1) % 3
        elif k == pg.K_g:
            self._measure_now()
        elif k == pg.K_c:
            from . import conformance
            self.conf_result = conformance.run(self.outdir, verbose=True)
        elif k == pg.K_s:
            self._screenshot()

    def _measure_now(self):
        from .truth import compare_masks, diff_image
        self.r.capture_gbuffer = True
        self.r.render(self.t_scene)
        ref, pred, vis = self.r.ground_truth()
        self.run.last_agreement = compare_masks(pred[vis], ref[vis])
        self.run.last_diff = diff_image(pred, ref, vis)
        if self.run.active:
            self.run.note_manual("medicao manual contra a referencia")

    def _screenshot(self):
        try:
            from PIL import Image
            os.makedirs(self.outdir, exist_ok=True)
            img = self.r.render(self.t_scene)
            name = os.path.join(self.outdir, f"captura_{int(time.time())}.png")
            Image.fromarray(img).save(name)
            print(f"[Aethelion] captura salva: {name}")
        except Exception as exc:
            print(f"[Aethelion] falha na captura: {exc}")

    # ------------------------------------------------------------------ #
    def _frame_image(self):
        if self.run.active:
            img = self.run.step(self.t_scene)
            if img is None:
                img = self.r.render(self.t_scene)
        else:
            img = self.r.render(self.t_scene)

        if self.view_mode == 1:
            s = self.r.stencil
            norm = (s.buf.astype(np.float32)
                    / max(1.0, float(s.max_val)))
            img = np.stack([np.clip(norm * 255, 0, 255).astype(np.uint8)] * 3,
                           axis=-1)
            img[..., 0] = np.where(s.buf > 0, 240, img[..., 0])
        elif self.view_mode == 2 and self.run.last_diff is not None:
            img = self.run.last_diff
        return img

    # ------------------------------------------------------------------ #
    def _blit_view(self, img):
        pg = self.pg
        surf = pg.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
        scale = min(VIEW_W / img.shape[1], VIEW_H / img.shape[0])
        size = (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale)))
        surf = pg.transform.scale(surf, size)
        self.screen.blit(surf, ((VIEW_W - size[0]) // 2, (VIEW_H - size[1]) // 2))

    def _text(self, x, y, txt, color=FG, font=None):
        self.screen.blit((font or self.font).render(txt, True, color), (x, y))
        return y + (font or self.font).get_height() + 2

    def _panel(self):
        pg, s, st = self.pg, self.r.settings, self.r.stats
        x0 = VIEW_W
        pg.draw.rect(self.screen, PANEL_BG, (x0, 0, PANEL_W, VIEW_H))
        x, y = x0 + 10, 8

        y = self._text(x, y, "CONFIGURAÇÃO", ACC, self.font_b)
        wrap = "sim" if s.stencil_wrap else "NÃO"
        y = self._text(x, y, f"algoritmo    : {s.algorithm}")
        y = self._text(x, y, f"stencil      : {s.stencil_bits} bits, wrap={wrap}",
                       OK if s.stencil_wrap else WARN)
        y = self._text(x, y, f"z-buffer     : "
                             f"{'float32' if s.depth_bits == 0 else str(s.depth_bits) + ' bits'}")
        y = self._text(x, y, f"extrusão     : {s.extrude_len:.1f}")
        y = self._text(x, y, f"precisão     : {s.precision}")
        y = self._text(x, y, f"resolução    : {s.width}x{s.height}")
        y = self._text(x, y, f"backend      : {BACKEND}", WARN)

        y += 8
        y = self._text(x, y, "DESEMPENHO", ACC, self.font_b)
        y = self._text(x, y, f"quadro       : {st.ms_total:8.1f} ms "
                             f"({st.fps:5.2f} fps)")
        y = self._text(x, y, f"  geometria  : {st.ms_geometry:8.1f} ms")
        y = self._text(x, y, f"  silhueta   : {st.ms_silhouette:8.1f} ms")
        y = self._text(x, y, f"  stencil    : {st.ms_stencil:8.1f} ms")

        y += 8
        y = self._text(x, y, "CARGA", ACC, self.font_b)
        y = self._text(x, y, f"tris cena    : {st.tris_scene_drawn:6d} / {st.tris_scene}")
        y = self._text(x, y, f"tris volume  : {st.tris_volume_drawn:6d} / {st.tris_volume}")
        y = self._text(x, y, f"frag stencil : {st.frag_stencil:8d}")
        y = self._text(x, y, f"px sombra    : {st.pixels_shadow:8d} / {st.pixels_visible}")

        y += 8
        y = self._text(x, y, "STENCIL", ACC, self.font_b)
        col = BAD if st.stencil_saturation else FG
        y = self._text(x, y, f"saturações   : {st.stencil_saturation:8d}", col)
        y = self._text(x, y, f"transbordos  : {st.stencil_wrap_events:8d}", OK)
        y = self._text(x, y, f"valor máximo : {st.stencil_max:8d}")
        col = BAD if st.volumes_with_camera else OK
        y = self._text(x, y, f"vols c/ câmera: {st.volumes_with_camera:7d}", col)

        y += 8
        y = self._text(x, y, "CONCORDÂNCIA (ray cast)", ACC, self.font_b)
        ag = self.run.last_agreement
        if ag is None:
            y = self._text(x, y, "sem medição ainda (tecla G)", WARN)
        else:
            c = OK if ag.f1 > 0.98 else (WARN if ag.f1 > 0.90 else BAD)
            y = self._text(x, y, f"precisão     : {ag.precision:6.4f}")
            y = self._text(x, y, f"revocação    : {ag.recall:6.4f}")
            y = self._text(x, y, f"F1           : {ag.f1:6.4f}", c)
            y = self._text(x, y, f"FP / FN      : {ag.fp} / {ag.fn}")

        if self.conf_result:
            y += 6
            p, t = self.conf_result["passed"], self.conf_result["total"]
            y = self._text(x, y, f"conformidade : {p}/{t}",
                           OK if p == t else BAD, self.font_b)

        y += 8
        y = self._text(x, y, "TECLAS", ACC, self.font_b)
        for k, d in HELP:
            y = self._text(x, y, f"{k:<9} {d}", (150, 150, 165), self.font_s)

    def _status(self):
        pg = self.pg
        pg.draw.rect(self.screen, (14, 14, 18),
                     (0, VIEW_H, VIEW_W + PANEL_W, STATUS_H))
        if self.run.active:
            p = self.run.phase
            left = max(0.0, self.run.total_duration
                       - (time.perf_counter() - self.run.t0))
            txt = (f"EXPERIMENTO  fase {p.key} [{p.hipotese}] {p.titulo}  |  "
                   f"quadro {self.run.frame}  |  restam {left:5.1f}s  |  "
                   f"origem={self.run.origem}")
            color = OK
        elif self.run.finished:
            txt = (f"CONCLUÍDO — {self.run.frame} quadros gravados em "
                   f"'{self.outdir}'. ESPAÇO reinicia. Gere o relatório com "
                   f"python main.py --report")
            color = ACC
        else:
            txt = ("PRONTO — ESPAÇO inicia o experimento de "
                   f"{self.run.total_duration:.0f}s em {len(PHASES)} fases. "
                   "Altere parâmetros a qualquer momento: tudo é registrado.")
            color = FG
        self.screen.blit(self.font.render(txt, True, color), (10, VIEW_H + 8))

    # ------------------------------------------------------------------ #
    def loop(self):
        pg = self.pg
        while self.alive:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    self.alive = False
                elif ev.type == pg.KEYDOWN:
                    self._key(ev.key)

            if not self.run.active:
                keys = pg.key.get_pressed()
                moved = False
                if keys[pg.K_LEFT]:
                    self.free_yaw -= 0.05; moved = True
                if keys[pg.K_RIGHT]:
                    self.free_yaw += 0.05; moved = True
                if keys[pg.K_UP]:
                    self.free_pitch = min(1.3, self.free_pitch + 0.04); moved = True
                if keys[pg.K_DOWN]:
                    self.free_pitch = max(-1.0, self.free_pitch - 0.04); moved = True
                if moved:
                    self._apply_free_camera()

            now = time.perf_counter()
            self.t_scene += (now - self.last)
            self.last = now

            self.screen.fill(BG)
            self._blit_view(self._frame_image())
            self._panel()
            self._status()
            pg.display.flip()
            self.clock.tick(60)

        if self.run.active:
            self.run.stop()
        pg.quit()


def main(settings: SVSettings | None = None, outdir: str = "results"):
    try:
        Viewer(settings, outdir).loop()
    except ImportError:
        print("pygame não encontrado. Instale com: pip install pygame-ce",
              file=sys.stderr)
        return 1
    return 0
