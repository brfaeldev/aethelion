"""
Visualizador Interativo e Controlador de Janela (viewer.py)

Gerencia o ciclo principal de eventos e renderização em Pygame, integrando
alternância em tempo real de backends (`cpu-sw`, `cpu-fast`, `gpu-gl`), quatro
modos de inspeção visual (cena, volumes 3D + silhueta, buffer de stencil e
diferença contra ray casting), resolução adaptativa, tela cheia e captura PNG.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from ..core.backend import (
    available_backends,
    create_renderer,
)
from ..core.settings import SVSettings, apply_preset
from ..evaluation.runner import ExperimentRunner
from .camera import OrbitCamera
from .panel import render_control_panel, render_status_bar
from .theme import (
    BITS_CYCLE,
    BG,
    DEFAULT_WIN_H,
    DEFAULT_WIN_W,
    DEPTH_CYCLE,
    MIN_PANEL_W,
    PREC_CYCLE,
    PRESET_NAMES,
    RES_CYCLE,
    STATUS_H,
    WARN,
)
from .widgets import UIPanelCanvas

__all__ = [
    "Viewer",
    "main",
]


class Viewer:
    """Controlador principal da janela interativa em Pygame."""

    def __init__(
        self,
        settings: SVSettings | None = None,
        outdir: str = "results",
        deterministic: bool = False,
        win_width: int = DEFAULT_WIN_W,
        win_height: int = DEFAULT_WIN_H,
        fullscreen: bool = False,
        config_path: str | None = None,
    ):
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption(
            "Aethelion — Sombras Volumétricas por Stencil (Eixo 7)  [F11: Tela Cheia | TAB: Backend]"
        )

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
        self.view_mode = 0

        self.camera = OrbitCamera(yaw=0.5, pitch=0.18, dist=7.0)
        self.ui_canvas = UIPanelCanvas()
        self._apply_free_camera()

        self.clock = pygame.time.Clock()
        self.t_scene = 0.0
        self.last = time.perf_counter()
        self.conf_result = None
        self.alive = True

        self.light_paused = False
        self.adaptive_res = False
        self._adaptive_saved_res: tuple[int, int] | None = None
        self._moving_until = 0.0
        self._dirty = True
        self._last_img: np.ndarray | None = None
        self._dragging = False
        self._drag_last = (0, 0)
        self._active_preset = "custom"
        self._status_msg = ""
        self._status_until = 0.0

    def _flash(self, msg: str, secs: float = 3.0) -> None:
        self._status_msg = msg
        self._status_until = time.perf_counter() + secs

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _apply_display_mode(self) -> None:
        pg = self.pg
        if self.fullscreen:
            self.screen = pg.display.set_mode((0, 0), pg.FULLSCREEN | pg.HWSURFACE | pg.DOUBLEBUF)
        else:
            self.screen = pg.display.set_mode(self.windowed_size, pg.RESIZABLE)
        self.win_w, self.win_h = self.screen.get_size()
        self._update_fonts()

    def _update_fonts(self) -> None:
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

    def _font(self, size: int, bold: bool = False):
        for name in ("consolas", "dejavusansmono", "couriernew", None):
            try:
                return self.pg.font.SysFont(name, size, bold=bold)
            except Exception:
                continue
        return self.pg.font.Font(None, size)

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self._apply_display_mode()
        self._mark_dirty()

    def _apply_free_camera(self) -> None:
        self.camera.apply_to(self.r)
        self._mark_dirty()

    def _trigger_adaptive_move(self) -> None:
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

    def _cycle(self, attr: str, values: list, label: str, delta: int = 1):
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

    def _toggle(self, attr: str, label: str):
        cur = bool(getattr(self.r.settings, attr))
        nxt = not cur
        setattr(self.r.settings, attr, nxt)
        self.r.sync()
        if self.run.active:
            self.run.note_manual(f"{label} -> {nxt}")
        self._mark_dirty()
        return nxt

    def _step_num(
        self, attr: str, delta: float, lo: float, hi: float, label: str, rebuild: bool = False
    ):
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

    def _step_resolution(self, delta: int = 1) -> None:
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

    def _switch_backend(self, target: str | None = None) -> None:
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

    def _cycle_preset(self) -> None:
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

    def _save_config(self) -> None:
        os.makedirs(self.outdir, exist_ok=True)
        path = self.config_path or os.path.join(self.outdir, "config_aethelion.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.r.settings.to_dict(), f, indent=2, ensure_ascii=False)
        self._flash(f"Config salva em {path}")

    def _set_algorithm(self, algo: str) -> None:
        self.r.settings.algorithm = algo
        self.r.sync()
        if self.run.active:
            self.run.note_manual(f"algoritmo -> {algo}")
        self._mark_dirty()

    def _key(self, k: int, mod: int = 0) -> None:
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
            self._set_algorithm("zpass")
        elif k == pg.K_2:
            self._set_algorithm("zfail")
        elif k == pg.K_0:
            self._set_algorithm("none")
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

    def _measure_now(self) -> None:
        from ..evaluation.truth import compare_masks, diff_image, evaluate_renderer_ground_truth
        t_m0 = time.perf_counter()
        prev_gbuf = self.r.capture_gbuffer
        try:
            self.r.capture_gbuffer = True
            self.r.render(self.t_scene)
            ref, pred, vis = evaluate_renderer_ground_truth(self.r)
            self.run.last_agreement = compare_masks(pred[vis], ref[vis])
            self.run.last_diff = diff_image(pred, ref, vis)
            if self.run.active:
                self.run.note_manual("medicao manual contra a referencia")
        finally:
            self.r.capture_gbuffer = prev_gbuf
        self.last += (time.perf_counter() - t_m0)
        self._mark_dirty()

    def _screenshot(self) -> None:
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

    def _frame_image(self) -> np.ndarray:
        if self.run.active:
            img = self.run.step()
            if img is None:
                img = self.r.render(self.t_scene)
            self._last_img = img
            self._dirty = False
        elif not self.light_paused or self._dirty or self._last_img is None:
            need_diff = (self.view_mode == 3)
            prev_gbuf = self.r.capture_gbuffer
            try:
                if need_diff:
                    self.r.capture_gbuffer = True
                img = self.r.render(self.t_scene)
                if need_diff:
                    from ..evaluation.truth import (
                        compare_masks,
                        diff_image,
                        evaluate_renderer_ground_truth,
                    )
                    ref, pred, vis = evaluate_renderer_ground_truth(self.r)
                    self.run.last_agreement = compare_masks(pred[vis], ref[vis])
                    self.run.last_diff = diff_image(pred, ref, vis)
            finally:
                self.r.capture_gbuffer = prev_gbuf
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

    def _layout_metrics(self) -> tuple[int, int, int]:
        self.win_w, self.win_h = self.screen.get_size()
        panel_w = max(MIN_PANEL_W, min(460, int(self.win_w * 0.32)))
        view_w = max(320, self.win_w - panel_w)
        view_h = max(240, self.win_h - STATUS_H)
        return view_w, view_h, panel_w

    def _blit_view(self, img: np.ndarray) -> None:
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

    def _panel(self) -> None:
        view_w, view_h, panel_w = self._layout_metrics()
        render_control_panel(self, self.ui_canvas, view_w, view_h, panel_w)

    def _status(self) -> None:
        view_w, view_h, panel_w = self._layout_metrics()
        render_status_bar(self, view_w, view_h, panel_w)

    def loop(self) -> None:
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
                        self.ui_canvas.handle_click(mx, my)
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
                    self.camera.orbit(dx, dy)
                    self._trigger_adaptive_move()
                    self._apply_free_camera()
                elif ev.type == pg.MOUSEWHEEL:
                    mx, _ = pg.mouse.get_pos()
                    if mx >= view_w:
                        self.ui_canvas.handle_wheel(ev.y, view_h)
                    elif not self.run.active:
                        self.camera.zoom(ev.y)
                        self._trigger_adaptive_move()
                        self._apply_free_camera()

            if not self.run.active:
                keys = pg.key.get_pressed()
                if self.camera.handle_keys(keys, pg):
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


def main(
    settings: SVSettings | None = None,
    outdir: str = "results",
    deterministic: bool = False,
    win_width: int = DEFAULT_WIN_W,
    win_height: int = DEFAULT_WIN_H,
    fullscreen: bool = False,
    config_path: str | None = None,
) -> int:
    try:
        Viewer(
            settings, outdir, deterministic=deterministic,
            win_width=win_width, win_height=win_height,
            fullscreen=fullscreen, config_path=config_path,
        ).loop()
    except ImportError:
        print("pygame não encontrado. Instale com: pip install pygame-ce", file=sys.stderr)
        return 1
    return 0
