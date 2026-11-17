"""Aplicação de tela única (pygame): viewport, HUD responsivo e barra de status.

Suporta:
  * Redimensionamento livre da janela (botão Maximizar do Windows ou arrastar bordas);
  * Modo Tela Cheia (Fullscreen) alternável a qualquer momento com F11 ou F;
  * Layout dinâmico: o viewport 3D, o painel lateral de telemetria e a barra de
    status se ajustam automaticamente ao tamanho atual da janela;
  * Controle da resolução interna de renderização em tempo real (teclas R, [ e ]).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from ..core.sv_renderer import BACKEND, SVRenderer, SVSettings
from ..evaluation.scenario import PHASES, ExperimentRunner

DEFAULT_WIN_W = 1280
DEFAULT_WIN_H = 780
MIN_PANEL_W = 360
STATUS_H = 34

BG = (22, 22, 28)
PANEL_BG = (32, 32, 42)
PANEL_DIV = (52, 52, 68)
FG = (232, 232, 240)
ACC = (143, 208, 255)
OK = (120, 220, 140)
WARN = (240, 170, 70)
BAD = (232, 92, 80)

RES_CYCLE = [
    (128, 96),
    (160, 120),
    (192, 144),
    (256, 192),
    (320, 240),
    (400, 300),
    (512, 384),
    (640, 480),
]
BITS_CYCLE = [8, 5, 4, 3, 2, 1]
DEPTH_CYCLE = [0, 24, 16, 12, 10]
PREC_CYCLE = ["float32", "float16", "fixed16_16", "fixed12_4"]
VIEW_NAMES = ["cena", "volumes 3D + silhueta", "buffer stencil", "diferença (ray cast)"]

HELP = [
    ("ESPACO", "iniciar / parar experimento"),
    ("F11 / F", "alternar TELA CHEIA"),
    ("R / ] / [", "resolução interna (+/-)"),
    ("1 / 2 / 0", "z-pass / z-fail / sem sombra"),
    ("W", "alterna INCR_WRAP / INCR"),
    ("B", "largura do stencil (1..8b)"),
    ("D", "largura do z-buffer"),
    ("P", "precisão numérica"),
    ("E", "extrusão finita (L)"),
    ("I", "alterna P_inf (extrusão w=0)"),
    ("V", "vista: cena/vol3D/stencil/diff"),
    ("G", "medir contra ray casting"),
    ("C", "suíte de conformidade"),
    ("S", "salvar captura PNG"),
    ("setas", "orbitar câmera livre"),
    ("ESC", "sair"),
]


class Viewer:
    def __init__(self, settings: SVSettings | None = None,
                 outdir: str = "results", deterministic: bool = False,
                 win_width: int = DEFAULT_WIN_W, win_height: int = DEFAULT_WIN_H,
                 fullscreen: bool = False):
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption(
            "Aethelion — Sombras Volumétricas por Stencil (Eixo 7)  [F11: Tela Cheia | R: Resolução]")

        info = pygame.display.Info()
        desk_w = info.current_w if info.current_w > 0 else DEFAULT_WIN_W
        desk_h = info.current_h if info.current_h > 0 else DEFAULT_WIN_H

        self.fullscreen = bool(fullscreen)
        self.windowed_size = (
            min(int(win_width), max(960, desk_w - 60)),
            min(int(win_height), max(680, desk_h - 80)),
        )
        self._apply_display_mode()

        self.r = SVRenderer(settings or SVSettings())
        self.run = ExperimentRunner(self.r, outdir, deterministic=deterministic)
        self.outdir = outdir
        self.view_mode = 0           # 0 cena, 1 volumes 3D, 2 stencil, 3 diferença
        self.free_yaw, self.free_pitch, self.free_dist = 0.5, 0.18, 7.0
        self._apply_free_camera()
        self.clock = pygame.time.Clock()
        self.t_scene = 0.0
        self.last = time.perf_counter()
        self.conf_result = None
        self.alive = True

    def _apply_display_mode(self):
        pg = self.pg
        if self.fullscreen:
            self.screen = pg.display.set_mode((0, 0), pg.FULLSCREEN | pg.HWSURFACE | pg.DOUBLEBUF)
        else:
            self.screen = pg.display.set_mode(self.windowed_size, pg.RESIZABLE)
        self.win_w, self.win_h = self.screen.get_size()
        self._update_fonts()

    def _update_fonts(self):
        # Ajusta tamanho das fontes proporcionalmente à altura da janela para caber tudo sem corte
        h = self.win_h
        if h >= 900:
            main_sz, small_sz = 15, 13
        elif h >= 740:
            main_sz, small_sz = 13, 12
        else:
            main_sz, small_sz = 12, 11
        self.font = self._font(main_sz)
        self.font_b = self._font(main_sz, bold=True)
        self.font_s = self._font(small_sz)

    def _font(self, size, bold=False):
        for name in ("consolas", "dejavusansmono", "couriernew", None):
            try:
                return self.pg.font.SysFont(name, size, bold=bold)
            except Exception:
                continue
        return self.pg.font.Font(None, size)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self._apply_display_mode()

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

    def _step_resolution(self, delta: int = 1):
        s = self.r.settings
        cur = (s.width, s.height)
        try:
            idx = RES_CYCLE.index(cur)
            nxt = RES_CYCLE[(idx + delta) % len(RES_CYCLE)]
        except ValueError:
            # Escolhe a mais próxima na lista
            dists = [abs(w * h - s.width * s.height) for w, h in RES_CYCLE]
            idx = int(np.argmin(dists))
            nxt = RES_CYCLE[(idx + delta) % len(RES_CYCLE)]
        s.width, s.height = nxt
        self.r.sync()
        if self.run.active:
            self.run.note_manual(f"resolucao -> {nxt[0]}x{nxt[1]}")

    def _key(self, k, mod=0):
        pg, s = self.pg, self.r.settings
        if k == pg.K_ESCAPE:
            if self.fullscreen:
                self.toggle_fullscreen()
            else:
                self.alive = False
        elif k in (pg.K_F11, pg.K_f):
            self.toggle_fullscreen()
        elif k == pg.K_RETURN and (mod & pg.KMOD_ALT):
            self.toggle_fullscreen()
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
            s.infinite_extrude = False
            self._cycle("extrude_len", [40.0, 15.0, 7.0, 3.5, 80.0], "extrusao")
        elif k == pg.K_i:
            s.infinite_extrude = not s.infinite_extrude
            self.run.active and self.run.note_manual(
                f"infinite_extrude (P_inf, w=0) -> {s.infinite_extrude}")
        elif k == pg.K_r:
            step = -1 if (mod & pg.KMOD_SHIFT) else 1
            self._step_resolution(step)
        elif k in (pg.K_RIGHTBRACKET, pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
            self._step_resolution(+1)
        elif k in (pg.K_LEFTBRACKET, pg.K_MINUS, pg.K_KP_MINUS):
            self._step_resolution(-1)
        elif k == pg.K_v:
            self.view_mode = (self.view_mode + 1) % 4
        elif k == pg.K_g:
            self._measure_now()
        elif k == pg.K_c:
            from ..evaluation import conformance
            self.conf_result = conformance.run(self.outdir, verbose=True)
        elif k == pg.K_s:
            self._screenshot()

    def _measure_now(self):
        from ..evaluation.truth import compare_masks, diff_image
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
            img = self._frame_image()
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
            img = self.r.render_volume_overlay(img)
        elif self.view_mode == 2:
            s = self.r.stencil
            norm = (s.buf.astype(np.float32)
                    / max(1.0, float(s.max_val)))
            img = np.stack([np.clip(norm * 255, 0, 255).astype(np.uint8)] * 3,
                           axis=-1)
            img[..., 0] = np.where(s.buf > 0, 240, img[..., 0])
        elif self.view_mode == 3 and self.run.last_diff is not None:
            img = self.run.last_diff
        return img

    # ------------------------------------------------------------------ #
    def _layout_metrics(self):
        """Calcula dimensões dinâmicas da área de visualização e do painel lateral."""
        self.win_w, self.win_h = self.screen.get_size()
        panel_w = max(MIN_PANEL_W, min(440, int(self.win_w * 0.30)))
        view_w = max(320, self.win_w - panel_w)
        view_h = max(240, self.win_h - STATUS_H)
        return view_w, view_h, panel_w

    def _blit_view(self, img):
        pg = self.pg
        view_w, view_h, _ = self._layout_metrics()
        surf = pg.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
        scale = min(view_w / max(1, img.shape[1]), view_h / max(1, img.shape[0]))
        size = (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale)))
        surf = pg.transform.scale(surf, size)
        self.screen.blit(surf, ((view_w - size[0]) // 2, (view_h - size[1]) // 2))

    def _text(self, x, y, txt, color=FG, font=None, gap=2):
        f = font or self.font
        self.screen.blit(f.render(txt, True, color), (x, y))
        return y + f.get_height() + gap

    def _panel(self):
        pg, s, st = self.pg, self.r.settings, self.r.stats
        view_w, view_h, panel_w = self._layout_metrics()
        x0 = view_w
        pg.draw.rect(self.screen, PANEL_BG, (x0, 0, panel_w, view_h))
        pg.draw.line(self.screen, PANEL_DIV, (x0, 0), (x0, view_h), 2)

        # Espaçamento adaptativo conforme a altura disponível
        compact = view_h < 720
        sec_gap = 4 if compact else 8
        line_gap = 1 if compact else 2
        x, y = x0 + 12, 8

        y = self._text(x, y, "CONFIGURAÇÃO", ACC, self.font_b, line_gap)
        wrap = "sim" if s.stencil_wrap else "NÃO"
        y = self._text(x, y, f"algoritmo    : {s.algorithm}", FG, self.font, line_gap)
        y = self._text(x, y, f"stencil      : {s.stencil_bits} bits, wrap={wrap}",
                       OK if s.stencil_wrap else WARN, self.font, line_gap)
        y = self._text(x, y, f"z-buffer     : "
                             f"{'float32' if s.depth_bits == 0 else str(s.depth_bits) + ' bits'}",
                       FG, self.font, line_gap)
        ext_txt = "P_inf (w=0)" if s.infinite_extrude else f"{s.extrude_len:.1f} (finita)"
        y = self._text(x, y, f"extrusão     : {ext_txt}",
                       OK if s.infinite_extrude else FG, self.font, line_gap)
        y = self._text(x, y, f"precisão     : {s.precision}", FG, self.font, line_gap)
        y = self._text(x, y, f"res. interna : {s.width}x{s.height}  (janela {self.win_w}x{self.win_h})",
                       FG, self.font, line_gap)
        y = self._text(x, y, f"vista (V)    : {VIEW_NAMES[self.view_mode]}", ACC, self.font, line_gap)

        y += sec_gap
        y = self._text(x, y, "DESEMPENHO & MEMÓRIA", ACC, self.font_b, line_gap)
        y = self._text(x, y, f"quadro       : {st.ms_total:7.1f} ms ({st.fps:5.2f} fps)",
                       FG, self.font, line_gap)
        y = self._text(x, y, f"  geometria  : {st.ms_geometry:7.1f} ms", FG, self.font, line_gap)
        y = self._text(x, y, f"  silhueta   : {st.ms_silhouette:7.1f} ms", FG, self.font, line_gap)
        y = self._text(x, y, f"  stencil    : {st.ms_stencil:7.1f} ms", FG, self.font, line_gap)
        y = self._text(x, y, f"memória tot. : {st.mem_total_kb:6.1f} KB (buf {st.mem_buffers_kb:.0f} KB)",
                       FG, self.font, line_gap)

        y += sec_gap
        y = self._text(x, y, "CARGA GEOMÉTRICA E FILL RATE", ACC, self.font_b, line_gap)
        y = self._text(x, y, f"tris cena    : {st.tris_scene_drawn:6d} / {st.tris_scene}",
                       FG, self.font, line_gap)
        y = self._text(x, y, f"tris volume  : {st.tris_volume_drawn:6d} / {st.tris_volume}",
                       FG, self.font, line_gap)
        y = self._text(x, y, f"frag stencil : {st.frag_stencil:8d}", FG, self.font, line_gap)
        y = self._text(x, y, f"px sombra    : {st.pixels_shadow:8d} / {st.pixels_visible}",
                       FG, self.font, line_gap)

        y += sec_gap
        y = self._text(x, y, "CONTADORES DE STENCIL", ACC, self.font_b, line_gap)
        col = BAD if st.stencil_saturation else FG
        y = self._text(x, y, f"saturações   : {st.stencil_saturation:8d}", col, self.font, line_gap)
        y = self._text(x, y, f"transbordos  : {st.stencil_wrap_events:8d}", OK, self.font, line_gap)
        y = self._text(x, y, f"valor máximo : {st.stencil_max:8d}", FG, self.font, line_gap)
        col = BAD if st.volumes_with_camera else OK
        y = self._text(x, y, f"vols c/ câmera: {st.volumes_with_camera:7d}", col, self.font, line_gap)

        y += sec_gap
        y = self._text(x, y, "CONCORDÂNCIA (RAY CASTING)", ACC, self.font_b, line_gap)
        ag = self.run.last_agreement
        if ag is None:
            y = self._text(x, y, "sem medição ainda (aperte G)", WARN, self.font, line_gap)
        else:
            c = OK if ag.f1 > 0.98 else (WARN if ag.f1 > 0.90 else BAD)
            y = self._text(x, y, f"precisão / rev: {ag.precision:6.4f} / {ag.recall:6.4f}",
                           FG, self.font, line_gap)
            y = self._text(x, y, f"F1-Score      : {ag.f1:6.4f}  (FP={ag.fp} FN={ag.fn})",
                           c, self.font, line_gap)

        if self.conf_result:
            p, t = self.conf_result["passed"], self.conf_result["total"]
            y = self._text(x, y, f"conformidade  : {p}/{t} PASS",
                           OK if p == t else BAD, self.font_b, line_gap)

        y += sec_gap
        y = self._text(x, y, "ATALHOS DE TECLADO", ACC, self.font_b, line_gap)
        for k, d in HELP:
            if y + self.font_s.get_height() > view_h - 4:
                break
            y = self._text(x, y, f"{k:<10} {d}", (165, 165, 180), self.font_s, 1)

    def _status(self):
        pg = self.pg
        view_w, view_h, panel_w = self._layout_metrics()
        total_w = view_w + panel_w
        pg.draw.rect(self.screen, (14, 14, 18),
                     (0, view_h, total_w, STATUS_H))
        pg.draw.line(self.screen, PANEL_DIV, (0, view_h), (total_w, view_h), 1)
        if self.run.active:
            p = self.run.phase
            left = max(0.0, self.run.total_duration - self.run.elapsed_effective)
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
            modo_tela = "TELA CHEIA (ESC/F11 sai)" if self.fullscreen else "F11/F: Tela Cheia"
            txt = (f"PRONTO — ESPAÇO inicia ({self.run.total_duration:.0f}s, {len(PHASES)} fases) | "
                   f"{modo_tela} | R/[/]: Resolução | V: Vistas (Volumes 3D)")
            color = FG
        self.screen.blit(self.font.render(txt, True, color), (12, view_h + (STATUS_H - self.font.get_height()) // 2))

    # ------------------------------------------------------------------ #
    def loop(self):
        pg = self.pg
        while self.alive:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    self.alive = False
                elif ev.type == pg.VIDEORESIZE and not self.fullscreen:
                    self.windowed_size = (max(880, ev.w), max(600, ev.h))
                    self.screen = pg.display.set_mode(self.windowed_size, pg.RESIZABLE)
                    self.win_w, self.win_h = self.screen.get_size()
                    self._update_fonts()
                elif ev.type == pg.KEYDOWN:
                    self._key(ev.key, getattr(ev, "mod", 0))

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


def main(settings: SVSettings | None = None, outdir: str = "results",
         deterministic: bool = False, win_width: int = DEFAULT_WIN_W,
         win_height: int = DEFAULT_WIN_H, fullscreen: bool = False):
    try:
        Viewer(settings, outdir, deterministic=deterministic,
               win_width=win_width, win_height=win_height,
               fullscreen=fullscreen).loop()
    except ImportError:
        print("pygame não encontrado. Instale com: pip install pygame-ce",
              file=sys.stderr)
        return 1
    return 0
