"""Aplicação de tela única (pygame): viewport, HUD interativo e barra de status.

Suporta:
  * Redimensionamento livre da janela e Tela Cheia (F11 / F);
  * Controle de câmera por teclado (setas) e mouse (arrastar esquerdo + scroll);
  * Painel lateral interativo clicável (ciclos, toggles, +/-, presets, backend);
  * Alternância em tempo real de backend (cpu-sw / cpu-fast / gpu-gl via TAB);
  * Pausa da luz (M) e renderização sob demanda (~0% CPU em repouso);
  * Persistência de configuração JSON (botão Salvar Config / --config).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from ..core.backend import (
    BACKENDS,
    PRESETS,
    apply_preset,
    available_backends,
    create_renderer,
)
from ..core.sv_renderer import SVSettings
from ..evaluation.scenario import PHASES, ExperimentRunner

DEFAULT_WIN_W = 1280
DEFAULT_WIN_H = 780
MIN_PANEL_W = 380
STATUS_H = 34

BG = (22, 22, 28)
PANEL_BG = (32, 32, 42)
PANEL_DIV = (52, 52, 68)
BTN_BG = (46, 48, 62)
BTN_HOVER = (64, 68, 88)
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
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]
ALGO_CYCLE = ["zfail", "zpass", "none"]
BITS_CYCLE = [8, 5, 4, 3, 2, 1]
DEPTH_CYCLE = [0, 24, 16, 12, 10]
PREC_CYCLE = ["float32", "float16", "fixed16_16", "fixed12_4"]
SHADING_CYCLE = ["gouraud", "flat", "unlit"]
TEX_RES_CYCLE = [16, 32, 64, 128]
SNAP_CYCLE = [0, 1, 2, 4]
VIEW_NAMES = ["cena", "volumes 3D + silhueta", "buffer stencil", "diferença (ray cast)"]
PRESET_NAMES = list(PRESETS.keys())

HELP = [
    ("ESPACO", "iniciar / parar experimento"),
    ("TAB", "alternar backend (CPU/GPU)"),
    ("M", "pausar/retomar luz (sob demanda)"),
    ("F11 / F", "alternar TELA CHEIA"),
    ("R / ] / [", "resolução interna (+/-)"),
    ("1 / 2 / 0", "z-pass / z-fail / sem sombra"),
    ("W / B / D", "wrap / stencil bits / z-buffer"),
    ("P / E / I", "precisão / extrusão L / P_inf"),
    ("V / G / C", "vista / ray cast / conformidade"),
    ("S / ESC", "captura PNG / sair"),
    ("mouse/setas", "arrastar=orbitar, roda=zoom"),
]


class Viewer:
    def __init__(self, settings: SVSettings | None = None,
                 outdir: str = "results", deterministic: bool = False,
                 win_width: int = DEFAULT_WIN_W, win_height: int = DEFAULT_WIN_H,
                 fullscreen: bool = False, config_path: str | None = None):
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption(
            "Aethelion — Sombras Volumétricas por Stencil (Eixo 7)  [F11: Tela Cheia | TAB: Backend]")

        info = pygame.display.Info()
        desk_w = info.current_w if info.current_w > 0 else DEFAULT_WIN_W
        desk_h = info.current_h if info.current_h > 0 else DEFAULT_WIN_H

        self.fullscreen = bool(fullscreen)
        self.windowed_size = (
            min(int(win_width), max(960, desk_w - 60)),
            min(int(win_height), max(680, desk_h - 80)),
        )
        self._apply_display_mode()

        s = settings or SVSettings()
        self.r = create_renderer(s.backend, s, fallback=True)
        self.run = ExperimentRunner(self.r, outdir, deterministic=deterministic)
        self.outdir = outdir
        self.config_path = config_path
        self.view_mode = 0           # 0 cena, 1 volumes 3D, 2 stencil, 3 diferença
        self.free_yaw, self.free_pitch, self.free_dist = 0.5, 0.18, 7.0
        self._apply_free_camera()
        self.clock = pygame.time.Clock()
        self.t_scene = 0.0
        self.last = time.perf_counter()
        self.conf_result = None
        self.alive = True

        # Otimização interativa e estado da UI
        self.light_paused = False
        self.adaptive_res = False
        self._adaptive_saved_res: tuple[int, int] | None = None
        self._moving_until = 0.0
        self._dirty = True
        self._last_img: np.ndarray | None = None
        self._dragging = False
        self._drag_last = (0, 0)
        self._panel_scroll = 0
        self._panel_content_h = 0
        self._hitboxes: list[tuple[object, callable]] = []
        self._active_preset = "custom"
        self._status_msg = ""
        self._status_until = 0.0

    def _flash(self, msg: str, secs: float = 3.0):
        self._status_msg = msg
        self._status_until = time.perf_counter() + secs

    def _mark_dirty(self):
        self._dirty = True

    def _apply_display_mode(self):
        pg = self.pg
        if self.fullscreen:
            self.screen = pg.display.set_mode((0, 0), pg.FULLSCREEN | pg.HWSURFACE | pg.DOUBLEBUF)
        else:
            self.screen = pg.display.set_mode(self.windowed_size, pg.RESIZABLE)
        self.win_w, self.win_h = self.screen.get_size()
        self._update_fonts()

    def _update_fonts(self):
        h = self.win_h
        if h >= 900:
            main_sz, small_sz = 14, 12
        elif h >= 740:
            main_sz, small_sz = 13, 11
        else:
            main_sz, small_sz = 12, 10
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
        self._mark_dirty()

    # ------------------------------------------------------------------ #
    def _apply_free_camera(self):
        cp = np.cos(self.free_pitch)
        self.r.camera_eye = np.array([
            self.free_dist * cp * np.sin(self.free_yaw),
            self.free_dist * np.sin(self.free_pitch) + 0.8,
            self.free_dist * cp * np.cos(self.free_yaw)], dtype=np.float32)
        self.r.camera_target = np.array([0.0, 0.25, 0.0], dtype=np.float32)
        self._mark_dirty()

    def _trigger_adaptive_move(self):
        if not self.adaptive_res or self.run.active:
            return
        s = self.r.settings
        if self._adaptive_saved_res is None:
            self._adaptive_saved_res = (s.width, s.height)
            nw = max(96, s.width // 2)
            nh = max(72, s.height // 2)
            if (nw, nh) != (s.width, s.height):
                s.width, s.height = nw, nh
                self.r.sync()
        self._moving_until = time.perf_counter() + 0.22

    # ------------------------------------------------------------------ #
    def _cycle(self, attr, values, label, delta: int = 1):
        cur = getattr(self.r.settings, attr)
        try:
            nxt = values[(values.index(cur) + delta) % len(values)]
        except ValueError:
            nxt = values[0]
        setattr(self.r.settings, attr, nxt)
        self.r.sync()
        if self.run.active:
            self.run.note_manual(f"{label} -> {nxt}")
        self._mark_dirty()
        return nxt

    def _toggle(self, attr, label):
        cur = bool(getattr(self.r.settings, attr))
        nxt = not cur
        setattr(self.r.settings, attr, nxt)
        self.r.sync()
        if self.run.active:
            self.run.note_manual(f"{label} -> {nxt}")
        self._mark_dirty()
        return nxt

    def _step_num(self, attr, delta, lo, hi, label, rebuild=False):
        cur = float(getattr(self.r.settings, attr))
        nxt = round(max(lo, min(hi, cur + delta)), 3)
        setattr(self.r.settings, attr, nxt)
        if rebuild:
            self.r.rebuild_scene()
        else:
            self.r.sync()
        if self.run.active:
            self.run.note_manual(f"{label} -> {nxt}")
        self._mark_dirty()
        return nxt

    def _step_resolution(self, delta: int = 1):
        s = self.r.settings
        self._adaptive_saved_res = None
        cur = (s.width, s.height)
        try:
            idx = RES_CYCLE.index(cur)
            nxt = RES_CYCLE[(idx + delta) % len(RES_CYCLE)]
        except ValueError:
            dists = [abs(w * h - s.width * s.height) for w, h in RES_CYCLE]
            idx = int(np.argmin(dists))
            nxt = RES_CYCLE[(idx + delta) % len(RES_CYCLE)]
        s.width, s.height = nxt
        self.r.sync()
        if self.run.active:
            self.run.note_manual(f"resolucao -> {nxt[0]}x{nxt[1]}")
        self._mark_dirty()

    def _switch_backend(self, target: str | None = None):
        avail = available_backends()
        cur = getattr(self.r, "backend_name", self.r.settings.backend)
        if target is None:
            try:
                nxt = avail[(avail.index(cur) + 1) % len(avail)]
            except ValueError:
                nxt = avail[0]
        else:
            nxt = target
        if nxt == cur:
            return
        eye = self.r.camera_eye.copy()
        tgt = self.r.camera_target.copy()
        old_r = self.r
        try:
            new_r = create_renderer(nxt, old_r.settings, fallback=False)
            new_r.camera_eye = eye
            new_r.camera_target = tgt
            self.r = new_r
            self.run.r = new_r
            if hasattr(old_r, "release"):
                old_r.release()
            self._flash(f"Backend alterado para {new_r.backend_name}")
        except Exception as exc:
            self._flash(f"Falha ao ativar {nxt}: {exc}", secs=4.5)
        self._mark_dirty()

    def _cycle_preset(self):
        if self._active_preset in PRESET_NAMES:
            nxt = PRESET_NAMES[(PRESET_NAMES.index(self._active_preset) + 1) % len(PRESET_NAMES)]
        else:
            nxt = PRESET_NAMES[0]
        apply_preset(self.r.settings, nxt)
        self._active_preset = nxt
        self._adaptive_saved_res = None
        self.r.rebuild_scene()
        self.r.sync()
        self._flash(f"Preset aplicado: {nxt}")
        self._mark_dirty()

    def _save_config(self):
        os.makedirs(self.outdir, exist_ok=True)
        path = self.config_path or os.path.join(self.outdir, "config_aethelion.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.r.settings.to_dict(), f, indent=2, ensure_ascii=False)
        self._flash(f"Config salva em {path}")

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
        elif k == pg.K_TAB:
            self._switch_backend()
        elif k == pg.K_m:
            self.light_paused = not self.light_paused
            self._flash("Luz pausada (render sob demanda)" if self.light_paused else "Luz animada")
            self._mark_dirty()
        elif k == pg.K_SPACE:
            if self.run.active:
                self.run.stop()
            else:
                self.t_scene = 0.0
                self.last = time.perf_counter()
                self.run.start()
            self._mark_dirty()
        elif k == pg.K_1:
            s.algorithm = "zpass"
            self.run.active and self.run.note_manual("algoritmo -> zpass")
            self._mark_dirty()
        elif k == pg.K_2:
            s.algorithm = "zfail"
            self.run.active and self.run.note_manual("algoritmo -> zfail")
            self._mark_dirty()
        elif k == pg.K_0:
            s.algorithm = "none"
            self.run.active and self.run.note_manual("algoritmo -> none")
            self._mark_dirty()
        elif k == pg.K_w:
            self._toggle("stencil_wrap", "wrap")
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
            self._toggle("infinite_extrude", "infinite_extrude (P_inf, w=0)")
        elif k == pg.K_r:
            step = -1 if (mod & pg.KMOD_SHIFT) else 1
            self._step_resolution(step)
        elif k in (pg.K_RIGHTBRACKET, pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
            self._step_resolution(+1)
        elif k in (pg.K_LEFTBRACKET, pg.K_MINUS, pg.K_KP_MINUS):
            self._step_resolution(-1)
        elif k == pg.K_v:
            self.view_mode = (self.view_mode + 1) % 4
            if self.view_mode == 3 and self.run.last_diff is None:
                self._measure_now()
            self._mark_dirty()
        elif k == pg.K_g:
            self._measure_now()
        elif k == pg.K_c:
            from ..evaluation import conformance
            self.conf_result = conformance.run(self.outdir, verbose=True)
            self._mark_dirty()
        elif k == pg.K_s:
            self._screenshot()

    def _measure_now(self):
        from ..evaluation.truth import compare_masks, diff_image, evaluate_renderer_ground_truth
        t_m0 = time.perf_counter()
        self.r.capture_gbuffer = True
        self.r.render(self.t_scene)
        ref, pred, vis = evaluate_renderer_ground_truth(self.r)
        self.run.last_agreement = compare_masks(pred[vis], ref[vis])
        self.run.last_diff = diff_image(pred, ref, vis)
        if self.run.active:
            self.run.note_manual("medicao manual contra a referencia")
        self.last += (time.perf_counter() - t_m0)
        self._mark_dirty()

    def _screenshot(self):
        try:
            from PIL import Image
            os.makedirs(self.outdir, exist_ok=True)
            img = self._frame_image()
            name = os.path.join(self.outdir, f"captura_{int(time.time())}.png")
            Image.fromarray(img).save(name)
            self._flash(f"Captura salva: {name}")
            print(f"[Aethelion] captura salva: {name}")
        except Exception as exc:
            print(f"[Aethelion] falha na captura: {exc}")

    # ------------------------------------------------------------------ #
    def _frame_image(self):
        if self.run.active:
            img = self.run.step()
            if img is None:
                img = self.r.render(self.t_scene)
            self._last_img = img
            self._dirty = False
        elif not self.light_paused or self._dirty or self._last_img is None:
            img = self.r.render(self.t_scene)
            self._last_img = img
            self._dirty = False
        else:
            img = self._last_img

        if self.view_mode == 1:
            img = self.r.render_volume_overlay(img)
        elif self.view_mode == 2:
            s = self.r.stencil
            norm = (s.buf.astype(np.float32) / max(1.0, float(s.max_val)))
            img = np.stack([np.clip(norm * 255, 0, 255).astype(np.uint8)] * 3, axis=-1)
            img[..., 0] = np.where(s.buf > 0, 240, img[..., 0])
        elif self.view_mode == 3 and self.run.last_diff is not None:
            img = self.run.last_diff
        return img

    # ------------------------------------------------------------------ #
    def _layout_metrics(self):
        self.win_w, self.win_h = self.screen.get_size()
        panel_w = max(MIN_PANEL_W, min(460, int(self.win_w * 0.32)))
        view_w = max(320, self.win_w - panel_w)
        view_h = max(240, self.win_h - STATUS_H)
        return view_w, view_h, panel_w

    def _blit_view(self, img):
        pg = self.pg
        view_w, view_h, _ = self._layout_metrics()
        surf = pg.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
        scale = min(view_w / max(1, img.shape[1]), view_h / max(1, img.shape[0]))
        size = (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale)))
        if self.r.settings.display_filter == "bilinear":
            try:
                surf = pg.transform.smoothscale(surf, size)
            except Exception:
                surf = pg.transform.scale(surf, size)
        else:
            surf = pg.transform.scale(surf, size)
        self.screen.blit(surf, ((view_w - size[0]) // 2, (view_h - size[1]) // 2))
        if self.view_mode == 3 and self.run.last_diff is None:
            msg = "Pressione G para calcular a diferença contra ray casting"
            lbl = self.font_b.render(msg, True, WARN, (18, 18, 24))
            self.screen.blit(lbl, ((view_w - lbl.get_width()) // 2, 16))

    def _text(self, x, y, txt, color=FG, font=None, gap=2):
        f = font or self.font
        self.screen.blit(f.render(txt, True, color), (x, y))
        return y + f.get_height() + gap

    def _btn(self, rect, label, cb, bg=BTN_BG, fg=FG, font=None):
        pg = self.pg
        f = font or self.font_s
        mpos = pg.mouse.get_pos()
        hover = rect.collidepoint(mpos)
        pg.draw.rect(self.screen, BTN_HOVER if hover else bg, rect, border_radius=4)
        pg.draw.rect(self.screen, PANEL_DIV, rect, 1, border_radius=4)
        surf = f.render(label, True, fg)
        self.screen.blit(surf, (rect.x + (rect.w - surf.get_width()) // 2,
                                rect.y + (rect.h - surf.get_height()) // 2))
        self._hitboxes.append((rect, cb))

    def _row_cycle(self, x, y, w, label, val_str, cb, color=FG):
        pg = self.pg
        h = self.font.get_height() + 4
        self.screen.blit(self.font.render(f"{label:<13}:", True, FG), (x, y + 2))
        bw = max(120, w - 125)
        rect = pg.Rect(x + 122, y, bw, h)
        self._btn(rect, str(val_str), cb, fg=color, font=self.font_s)
        return y + h + 2

    def _row_step(self, x, y, w, label, val_str, cb_minus, cb_plus, color=FG):
        pg = self.pg
        h = self.font.get_height() + 4
        self.screen.blit(self.font.render(f"{label:<13}:", True, FG), (x, y + 2))
        bw_step = 24
        val_w = max(66, w - 125 - bw_step * 2 - 8)
        vx = x + 122
        self._btn(pg.Rect(vx, y, bw_step, h), "-", cb_minus)
        lbl = self.font_s.render(str(val_str), True, color)
        self.screen.blit(lbl, (vx + bw_step + 4 + max(0, (val_w - lbl.get_width()) // 2), y + 3))
        self._btn(pg.Rect(vx + bw_step + 4 + val_w + 4, y, bw_step, h), "+", cb_plus)
        return y + h + 2

    def _panel(self):
        pg, s, st = self.pg, self.r.settings, self.r.stats
        view_w, view_h, panel_w = self._layout_metrics()
        x0 = view_w
        pg.draw.rect(self.screen, PANEL_BG, (x0, 0, panel_w, view_h))
        pg.draw.line(self.screen, PANEL_DIV, (x0, 0), (x0, view_h), 2)

        self._hitboxes.clear()
        old_clip = self.screen.get_clip()
        self.screen.set_clip(pg.Rect(x0, 0, panel_w, view_h))

        x = x0 + 10
        w = panel_w - 20
        y0 = 8 - self._panel_scroll
        y = y0
        gap = 2

        # ---- Controles Rápidos & Backend ---- #
        y = self._text(x, y, "EXECUÇÃO & CONTROLE (CLICÁVEL)", ACC, self.font_b, gap)
        bname = getattr(self.r, "backend_name", s.backend)
        y = self._row_cycle(x, y, w, "backend (TAB)", bname,
                            self._switch_backend, OK if bname != "cpu-sw" else ACC)
        y = self._row_cycle(x, y, w, "preset", self._active_preset,
                            self._cycle_preset, ACC)
        y = self._row_cycle(x, y, w, "luz (M)",
                            "PAUSADA" if self.light_paused else "animada",
                            lambda: self._key(pg.K_m),
                            WARN if self.light_paused else OK)
        y = self._row_cycle(x, y, w, "res. adaptat.",
                            "ON" if self.adaptive_res else "OFF",
                            lambda: setattr(self, "adaptive_res", not self.adaptive_res),
                            OK if self.adaptive_res else FG)

        # ---- Parâmetros do Pipeline ---- #
        y += 4
        y = self._text(x, y, "CONFIGURAÇÃO DO PIPELINE", ACC, self.font_b, gap)
        y = self._row_cycle(x, y, w, "algoritmo", s.algorithm,
                            lambda: self._cycle("algorithm", ALGO_CYCLE, "algoritmo"))
        y = self._row_cycle(x, y, w, "stencil bits", f"{s.stencil_bits}b",
                            lambda: self._cycle("stencil_bits", BITS_CYCLE, "stencil_bits"))
        y = self._row_cycle(x, y, w, "stencil wrap",
                            "INCR_WRAP" if s.stencil_wrap else "SATURADO",
                            lambda: self._toggle("stencil_wrap", "wrap"),
                            OK if s.stencil_wrap else WARN)
        y = self._row_cycle(x, y, w, "z-buffer",
                            "float32" if s.depth_bits == 0 else f"{s.depth_bits} bits",
                            lambda: self._cycle("depth_bits", DEPTH_CYCLE, "depth_bits"))
        y = self._row_cycle(x, y, w, "extrusão inf",
                            "P_inf (w=0)" if s.infinite_extrude else "finita",
                            lambda: self._toggle("infinite_extrude", "infinite_extrude"),
                            OK if s.infinite_extrude else FG)
        y = self._row_step(x, y, w, "extrusão L", f"{s.extrude_len:.1f}",
                           lambda: self._step_num("extrude_len", -5.0, 2.0, 120.0, "extrude_len"),
                           lambda: self._step_num("extrude_len", +5.0, 2.0, 120.0, "extrude_len"))
        y = self._row_cycle(x, y, w, "depth clamp",
                            "ON" if s.depth_clamp else "OFF",
                            lambda: self._toggle("depth_clamp", "depth_clamp"))
        y = self._row_cycle(x, y, w, "precisão", s.precision,
                            lambda: self._cycle("precision", PREC_CYCLE, "precisao"))
        y = self._row_step(x, y, w, "res. interna", f"{s.width}x{s.height}",
                           lambda: self._step_resolution(-1),
                           lambda: self._step_resolution(+1))

        # ---- Filtros, Textura & Câmera ---- #
        y += 4
        y = self._text(x, y, "FILTROS, TEXTURA & CÂMERA", ACC, self.font_b, gap)
        y = self._row_cycle(x, y, w, "shading", s.shading,
                            lambda: self._cycle("shading", SHADING_CYCLE, "shading"))
        y = self._row_cycle(x, y, w, "texturas", "ON" if s.textured else "OFF",
                            lambda: self._toggle("textured", "textured"))
        y = self._row_cycle(x, y, w, "filtro tex", s.texture_filter,
                            lambda: self._cycle("texture_filter", ["nearest", "bilinear"], "texture_filter"))
        y = self._row_cycle(x, y, w, "res. textura", f"{s.texture_res}px",
                            lambda: self._cycle("texture_res", TEX_RES_CYCLE, "texture_res"))
        y = self._row_cycle(x, y, w, "cor / dither",
                            f"{s.color_bits}b / {'ON' if s.dither else 'OFF'}",
                            lambda: self._cycle("color_bits", BITS_CYCLE, "color_bits"))
        y = self._row_cycle(x, y, w, "dither Bayer", "ON" if s.dither else "OFF",
                            lambda: self._toggle("dither", "dither"))
        y = self._row_cycle(x, y, w, "vertex snap",
                            "OFF" if s.vertex_snap == 0 else f"1/{s.vertex_snap}px",
                            lambda: self._cycle("vertex_snap", SNAP_CYCLE, "vertex_snap"))
        y = self._row_cycle(x, y, w, "filtro janela", s.display_filter,
                            lambda: self._cycle("display_filter", ["nearest", "bilinear"], "display_filter"))
        y = self._row_step(x, y, w, "ambiente", f"{s.ambient:.2f}",
                           lambda: self._step_num("ambient", -0.05, 0.0, 1.0, "ambient"),
                           lambda: self._step_num("ambient", +0.05, 0.0, 1.0, "ambient"))
        y = self._row_step(x, y, w, "triângulos", f"{s.detail:.2f}x ({st.tris_scene})",
                           lambda: self._step_num("detail", -0.25, 0.25, 4.0, "detail", rebuild=True),
                           lambda: self._step_num("detail", +0.25, 0.25, 4.0, "detail", rebuild=True))
        y = self._row_step(x, y, w, "fov (°)", f"{s.fov:.0f}",
                           lambda: self._step_num("fov", -5.0, 30.0, 110.0, "fov"),
                           lambda: self._step_num("fov", +5.0, 30.0, 110.0, "fov"))
        y = self._row_step(x, y, w, "near", f"{s.near:.2f}",
                           lambda: self._step_num("near", -0.05, 0.05, 2.0, "near"),
                           lambda: self._step_num("near", +0.05, 0.05, 2.0, "near"))
        y = self._row_step(x, y, w, "far", f"{s.far:.0f}",
                           lambda: self._step_num("far", -5.0, 10.0, 200.0, "far"),
                           lambda: self._step_num("far", +5.0, 10.0, 200.0, "far"))
        y = self._row_cycle(x, y, w, "vista (V)", VIEW_NAMES[self.view_mode],
                            lambda: self._key(pg.K_v), ACC)

        # ---- Botões de Ação ---- #
        y += 4
        bh = 24
        bw2 = (w - 6) // 2
        self._btn(pg.Rect(x, y, bw2, bh),
                  "Parar Exp" if self.run.active else "Iniciar Exp",
                  lambda: self._key(pg.K_SPACE),
                  bg=(45, 75, 55) if not self.run.active else (95, 50, 50))
        self._btn(pg.Rect(x + bw2 + 6, y, bw2, bh), "Medir RayCast (G)",
                  self._measure_now)
        y += bh + 4
        self._btn(pg.Rect(x, y, bw2, bh), "Conformidade (C)",
                  lambda: self._key(pg.K_c))
        self._btn(pg.Rect(x + bw2 + 6, y, bw2, bh), "Salvar Config",
                  self._save_config)
        y += bh + 6

        # ---- Telemetria ---- #
        y = self._text(x, y, "DESEMPENHO & MEMÓRIA", ACC, self.font_b, gap)
        y = self._text(x, y, f"quadro   : {st.ms_total:6.1f} ms ({st.fps:5.2f} fps)", FG, self.font, gap)
        y = self._text(x, y, f"  geom/sil/sten: {st.ms_geometry:.1f} / {st.ms_silhouette:.1f} / {st.ms_stencil:.1f} ms",
                       FG, self.font_s, gap)
        y = self._text(x, y, f"memória  : {st.mem_total_kb:6.1f} KB (buf {st.mem_buffers_kb:.0f} KB)", FG, self.font, gap)
        y = self._text(x, y, f"tris cena/vol : {st.tris_scene_drawn}/{st.tris_scene} | {st.tris_volume_drawn}/{st.tris_volume}",
                       FG, self.font_s, gap)
        y = self._text(x, y, f"frag st / px  : {st.frag_stencil} | {st.pixels_shadow}/{st.pixels_visible}",
                       FG, self.font_s, gap)

        col_sat = BAD if st.stencil_saturation else FG
        y = self._text(x, y, f"sat/wrap/cam  : {st.stencil_saturation} / {st.stencil_wrap_events} / {st.volumes_with_camera}",
                       col_sat, self.font_s, gap)

        ag = self.run.last_agreement
        if ag is None:
            y = self._text(x, y, "concordância  : sem medição (G)", WARN, self.font_s, gap)
        else:
            c = OK if ag.f1 > 0.98 else (WARN if ag.f1 > 0.90 else BAD)
            y = self._text(x, y, f"F1={ag.f1:.4f} (P={ag.precision:.4f} R={ag.recall:.4f})",
                           c, self.font_s, gap)

        if self.conf_result:
            p, t = self.conf_result["passed"], self.conf_result["total"]
            y = self._text(x, y, f"conformidade  : {p}/{t} PASS",
                           OK if p == t else BAD, self.font_b, gap)

        y += 4
        y = self._text(x, y, "ATALHOS", ACC, self.font_b, gap)
        for k, d in HELP:
            y = self._text(x, y, f"{k:<11} {d}", (165, 165, 180), self.font_s, 1)

        self._panel_content_h = y - y0 + 10
        max_scroll = max(0, self._panel_content_h - view_h)
        if self._panel_scroll > max_scroll:
            self._panel_scroll = max_scroll

        self.screen.set_clip(old_clip)

    def _status(self):
        pg = self.pg
        view_w, view_h, panel_w = self._layout_metrics()
        total_w = view_w + panel_w
        pg.draw.rect(self.screen, (14, 14, 18), (0, view_h, total_w, STATUS_H))
        pg.draw.line(self.screen, PANEL_DIV, (0, view_h), (total_w, view_h), 1)
        if self._status_msg and time.perf_counter() < self._status_until:
            txt, color = self._status_msg, ACC
        elif self.run.active:
            p = self.run.phase
            left = max(0.0, self.run.total_duration - self.run.elapsed_effective)
            txt = (f"EXPERIMENTO  fase {p.key} [{p.hipotese}] {p.titulo}  |  "
                   f"quadro {self.run.frame}  |  restam {left:5.1f}s  |  "
                   f"backend={getattr(self.r, 'backend_name', 'cpu-sw')}")
            color = OK
        elif self.run.finished:
            txt = (f"CONCLUÍDO — {self.run.frame} quadros gravados em "
                   f"'{self.outdir}'. ESPAÇO reinicia.")
            color = ACC
        elif self.view_mode == 3 and self.run.last_diff is None:
            txt = "VISTA DE DIFERENÇA — Pressione G para calcular a diferença contra ray casting"
            color = WARN
        else:
            bname = getattr(self.r, "backend_name", "cpu-sw")
            modo_tela = "TELA CHEIA" if self.fullscreen else "F11: Tela Cheia"
            txt = (f"PRONTO [{bname}] — TAB: Backend | M: Pausar Luz | Arraste: Câmera | "
                   f"ESPAÇO: Experimento | {modo_tela}")
            color = FG
        self.screen.blit(self.font.render(txt, True, color),
                         (12, view_h + (STATUS_H - self.font.get_height()) // 2))

    # ------------------------------------------------------------------ #
    def loop(self):
        pg = self.pg
        while self.alive:
            view_w, view_h, _ = self._layout_metrics()
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    self.alive = False
                elif ev.type == pg.VIDEORESIZE and not self.fullscreen:
                    self.windowed_size = (max(880, ev.w), max(600, ev.h))
                    self.screen = pg.display.set_mode(self.windowed_size, pg.RESIZABLE)
                    self.win_w, self.win_h = self.screen.get_size()
                    self._update_fonts()
                    self._mark_dirty()
                elif ev.type == pg.KEYDOWN:
                    self._key(ev.key, getattr(ev, "mod", 0))
                elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                    mx, my = ev.pos
                    if mx >= view_w and my < view_h:
                        for rect, cb in self._hitboxes:
                            if rect.collidepoint(mx, my):
                                cb()
                                break
                    elif mx < view_w and my < view_h and not self.run.active:
                        self._dragging = True
                        self._drag_last = (mx, my)
                elif ev.type == pg.MOUSEBUTTONUP and ev.button == 1:
                    self._dragging = False
                elif ev.type == pg.MOUSEMOTION and self._dragging and not self.run.active:
                    mx, my = ev.pos
                    dx = mx - self._drag_last[0]
                    dy = my - self._drag_last[1]
                    self._drag_last = (mx, my)
                    self.free_yaw += dx * 0.008
                    self.free_pitch = float(np.clip(self.free_pitch + dy * 0.008, -1.0, 1.3))
                    self._trigger_adaptive_move()
                    self._apply_free_camera()
                elif ev.type == pg.MOUSEWHEEL:
                    mx, my = pg.mouse.get_pos()
                    if mx >= view_w:
                        max_sc = max(0, self._panel_content_h - view_h)
                        self._panel_scroll = int(np.clip(self._panel_scroll - ev.y * 28, 0, max_sc))
                    elif not self.run.active:
                        self.free_dist = float(np.clip(self.free_dist - ev.y * 0.45, 2.0, 22.0))
                        self._trigger_adaptive_move()
                        self._apply_free_camera()

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
                    self._trigger_adaptive_move()
                    self._apply_free_camera()

            if self._adaptive_saved_res is not None and time.perf_counter() >= self._moving_until:
                self.r.settings.width, self.r.settings.height = self._adaptive_saved_res
                self._adaptive_saved_res = None
                self.r.sync()
                self._mark_dirty()

            now = time.perf_counter()
            if not self.light_paused:
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
        if hasattr(self.r, "release"):
            self.r.release()
        pg.quit()


def main(settings: SVSettings | None = None, outdir: str = "results",
         deterministic: bool = False, win_width: int = DEFAULT_WIN_W,
         win_height: int = DEFAULT_WIN_H, fullscreen: bool = False,
         config_path: str | None = None):
    try:
        Viewer(settings, outdir, deterministic=deterministic,
               win_width=win_width, win_height=win_height,
               fullscreen=fullscreen, config_path=config_path).loop()
    except ImportError:
        print("pygame não encontrado. Instale com: pip install pygame-ce",
              file=sys.stderr)
        return 1
    return 0
