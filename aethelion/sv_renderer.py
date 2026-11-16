"""Rasterizador por software com stencil e volumes de sombra.

Fluxo por quadro:

    1. Passagem de geometria  -> profundidade, cor iluminada, cor ambiente,
                                 posição de mundo (G-buffer opcional)
    2. Construção de volumes  -> silhueta + extrusão + tampas
    3. Passagem de stencil    -> z-pass ou z-fail, sem escrita de cor ou de
                                 profundidade
    4. Composição             -> pixel sombreado recebe apenas o termo ambiente
    5. Resolve                -> quantização de cor e dithering

O backend é integralmente por software e identificado como 'cpu-sw' em todas as
saídas. Nenhum número produzido aqui descreve o desempenho de uma GPU real.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np

from . import mathutil as mu
from . import shadow as sv
from .pipeline import apply_precision, clip_near
from .raster import scan_triangle
from .scene_crypt import Prop, build_crypt, light_position
from .stencil import StencilBuffer
from .texture import make_textures

F = np.float32
BACKEND = "cpu-sw"


# --------------------------------------------------------------------------- #
@dataclass
class SVSettings:
    # Framebuffer
    width: int = 256
    height: int = 192
    # Algoritmo de sombra
    algorithm: str = "zfail"        # none | zpass | zfail
    stencil_bits: int = 8
    stencil_wrap: bool = True
    extrude_len: float = 40.0
    # Arquitetura do buffer de profundidade
    depth_bits: int = 0             # 0 = float32; senão quantiza em N bits
    near: float = 0.15
    far: float = 60.0
    fov: float = 62.0
    # Pipeline legado reaproveitado
    precision: str = "float32"
    vertex_snap: int = 0
    shading: str = "gouraud"        # gouraud | flat | unlit
    textured: bool = True
    texture_res: int = 64
    texture_filter: str = "nearest"
    color_bits: int = 8
    dither: bool = False
    # Cena
    detail: float = 1.0
    ambient: float = 0.17

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SVStats:
    ms_geometry: float = 0.0
    ms_silhouette: float = 0.0
    ms_stencil: float = 0.0
    ms_composite: float = 0.0
    ms_total: float = 0.0
    tris_scene: int = 0
    tris_scene_drawn: int = 0
    tris_volume: int = 0
    tris_volume_drawn: int = 0
    frag_scene: int = 0
    frag_stencil: int = 0
    pixels_visible: int = 0
    pixels_shadow: int = 0
    volumes_with_camera: int = 0
    stencil_saturation: int = 0
    stencil_wrap_events: int = 0
    stencil_max: int = 0

    @property
    def fps(self) -> float:
        return 1000.0 / self.ms_total if self.ms_total > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fps"] = self.fps
        return d


# --------------------------------------------------------------------------- #
def quantize_depth(z: np.ndarray, bits: int) -> np.ndarray:
    """Emula um buffer de profundidade de largura reduzida (z em NDC [-1,1])."""
    if bits <= 0 or bits >= 32:
        return z
    levels = F((1 << int(bits)) - 1)
    d = np.clip((z * F(0.5) + F(0.5)), 0.0, 1.0)
    return (np.round(d * levels) / levels * F(2.0) - F(1.0)).astype(F)


class SVRenderer:
    AMBIENT_FLOOR = F(0.04)

    def __init__(self, settings: SVSettings | None = None):
        self.settings = settings or SVSettings()
        self.scene: list[Prop] = build_crypt(self.settings.detail)
        self.textures = make_textures(self.settings.texture_res,
                                      self.settings.color_bits)
        self.stencil = StencilBuffer(self.settings.width, self.settings.height,
                                     self.settings.stencil_bits,
                                     self.settings.stencil_wrap)
        self.camera_eye = np.array([0.0, 0.9, 7.2], dtype=F)
        self.camera_target = np.array([0.0, 0.2, 0.0], dtype=F)
        self.light = light_position(0.0)
        self.stats = SVStats()
        self._alloc()
        self._soups: dict[int, tuple] = {}
        self.capture_gbuffer = False
        self.world_pos: np.ndarray | None = None
        self.shadow_mask = np.zeros((self.settings.height, self.settings.width), bool)
        self.visible_mask = np.zeros_like(self.shadow_mask)

    # ------------------------------------------------------------------ #
    def _alloc(self) -> None:
        s = self.settings
        shape = (s.height, s.width)
        self.depth = np.ones(shape, dtype=F)
        self.col_lit = np.zeros(shape + (3,), dtype=F)
        self.col_amb = np.zeros(shape + (3,), dtype=F)
        self.gbuf = np.zeros(shape + (3,), dtype=F)
        self.stencil.resize(s.width, s.height)

    def sync(self) -> None:
        """Reaplica configurações que exigem realocação."""
        s = self.settings
        if self.depth.shape != (s.height, s.width):
            self._alloc()
        self.stencil.set_bits(s.stencil_bits)
        self.stencil.wrap = bool(s.stencil_wrap)
        tex = next(iter(self.textures.values()), None)
        if tex is None or tex.res != s.texture_res or tex.color_bits != s.color_bits:
            self.textures = make_textures(s.texture_res, s.color_bits)

    def rebuild_scene(self) -> None:
        self.scene = build_crypt(self.settings.detail)
        self._soups.clear()

    # ------------------------------------------------------------------ #
    def _world(self, prop: Prop):
        """Posições e normais em espaço de mundo (cache por instância)."""
        key = id(prop)
        cached = self._soups.get(key)
        if cached is not None:
            return cached
        pos = prop.solid.positions
        ph = np.concatenate([pos, np.ones((pos.shape[0], 1), dtype=F)], axis=1)
        pw = (prop.model @ ph.T).T[:, :3].astype(F)
        m3 = prop.model[:3, :3]
        try:
            nm = np.linalg.inv(m3).T
        except np.linalg.LinAlgError:
            nm = m3
        nw = (nm @ prop.solid.normals.T).T
        nw = (nw / np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-9)).astype(F)
        self._soups[key] = (pw, nw)
        return pw, nw

    # ------------------------------------------------------------------ #
    def render(self, t: float = 0.0) -> np.ndarray:
        s = self.settings
        self.sync()
        st = SVStats()
        t0 = time.perf_counter()

        self.depth[:] = 1.0
        self.col_lit[:] = (0.035, 0.040, 0.060)
        self.col_amb[:] = (0.035, 0.040, 0.060)
        self.gbuf[:] = 0.0
        self.stencil.clear(0)
        self.stencil.reset_counters()

        self.light = light_position(t)
        aspect = s.width / max(1, s.height)
        view = mu.look_at(self.camera_eye, self.camera_target)
        proj = mu.perspective(s.fov, aspect, s.near, s.far)
        vp = (proj @ view).astype(F)

        # ---- 1. Geometria ---- #
        tg = time.perf_counter()
        for prop in self.scene:
            self._draw_scene_prop(prop, vp, st)
        st.ms_geometry = (time.perf_counter() - tg) * 1000.0

        # ---- 2 e 3. Volumes e stencil ---- #
        if s.algorithm in ("zpass", "zfail"):
            self._shadow_pass(vp, st)

        # ---- 4. Composição ---- #
        tc = time.perf_counter()
        self.visible_mask = self.depth < 1.0
        self.shadow_mask = (self.stencil.buf != 0)
        shadow3 = self.shadow_mask[..., None]
        out = np.where(shadow3, self.col_amb, self.col_lit)
        st.pixels_visible = int(np.count_nonzero(self.visible_mask))
        st.pixels_shadow = int(np.count_nonzero(self.shadow_mask & self.visible_mask))
        st.stencil_saturation = self.stencil.saturation_events
        st.stencil_wrap_events = self.stencil.wrap_events
        st.frag_stencil = self.stencil.fragments
        st.stencil_max = int(self.stencil.buf.max()) if self.stencil.buf.size else 0
        st.ms_composite = (time.perf_counter() - tc) * 1000.0

        img = self._resolve(out)
        st.ms_total = (time.perf_counter() - t0) * 1000.0
        self.stats = st
        return img

    # ------------------------------------------------------------------ #
    def _resolve(self, buf: np.ndarray) -> np.ndarray:
        s = self.settings
        img = np.clip(buf, 0.0, 1.0)
        if s.color_bits < 8:
            levels = F((1 << s.color_bits) - 1)
            if s.dither:
                yy, xx = np.mgrid[0:s.height, 0:s.width]
                bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6],
                                  [3, 11, 1, 9], [15, 7, 13, 5]], dtype=F) / 16.0
                img = np.clip(img + (bayer[yy % 4, xx % 4][..., None] - 0.5) / levels,
                              0.0, 1.0)
            img = np.round(img * levels) / levels
        return (img * 255.0 + 0.5).astype(np.uint8)

    # ------------------------------------------------------------------ #
    def _project(self, world_tris: np.ndarray, vp: np.ndarray):
        """(M,3,3) mundo -> (M,3,4) clip, com precisão numérica aplicada."""
        s = self.settings
        flat = world_tris.reshape(-1, 3)
        ph = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=F)], axis=1)
        clip = (vp @ ph.T).T.astype(F)
        clip = apply_precision(clip, s.precision)
        return clip.reshape(world_tris.shape[0], 3, 4)

    def _screen(self, clip_tri: np.ndarray):
        """(3,4) clip -> sx, sy, z(NDC), invw. None se degenerado."""
        s = self.settings
        w = clip_tri[:, 3]
        if np.any(w <= 1e-5) or not np.all(np.isfinite(w)):
            return None
        ndc = (clip_tri[:, :3] / w[:, None]).astype(F)
        ndc = apply_precision(ndc, s.precision)
        if not np.all(np.isfinite(ndc)):
            return None
        sx = (ndc[:, 0] * F(0.5) + F(0.5)) * F(s.width)
        sy = (F(1.0) - (ndc[:, 1] * F(0.5) + F(0.5))) * F(s.height)
        if s.vertex_snap > 0:
            g = F(s.vertex_snap)
            sx = np.round(sx * g) / g
            sy = np.round(sy * g) / g
        return sx, sy, quantize_depth(ndc[:, 2], s.depth_bits), (1.0 / w).astype(F)

    # ------------------------------------------------------------------ #
    def _draw_scene_prop(self, prop: Prop, vp: np.ndarray, st: SVStats) -> None:
        s = self.settings
        pw, nw = self._world(prop)
        idx = prop.solid.indices
        st.tris_scene += int(idx.shape[0])

        tri_w = np.stack([pw[idx[:, 0]], pw[idx[:, 1]], pw[idx[:, 2]]], axis=1)
        tri_n = np.stack([nw[idx[:, 0]], nw[idx[:, 1]], nw[idx[:, 2]]], axis=1)
        clip = self._project(tri_w, vp)

        tex = self.textures.get(prop.texture) if s.textured else None
        tint = np.array(prop.tint, dtype=F)

        # Iluminação difusa por vértice (fonte pontual)
        ldir = self.light[None, None, :] - tri_w
        ln = np.linalg.norm(ldir, axis=2, keepdims=True)
        ldir = ldir / np.maximum(ln, 1e-9)
        lam = np.clip(np.einsum("mvj,mvj->mv", tri_n, ldir), 0.0, 1.0)
        atten = F(1.0) / (F(1.0) + F(0.035) * (ln[..., 0] ** 2))
        lit = (lam * atten).astype(F)
        if s.shading == "unlit":
            lit = np.ones_like(lit)
        elif s.shading == "flat":
            lit = np.repeat(lit.mean(axis=1, keepdims=True), 3, axis=1)

        # UV planar simples (x+z, y) — suficiente para a cripta
        uv = np.stack([(tri_w[..., 0] + tri_w[..., 2]) * F(0.35),
                       (tri_w[..., 1] + tri_w[..., 2] * 0.15) * F(0.35)], axis=2).astype(F)

        for m in range(clip.shape[0]):
            parts = self._clip_parts(clip[m], [lit[m].reshape(3, 1), uv[m], tri_w[m]])
            for cp, (lv, uvv, wpv) in parts:
                scr = self._screen(cp)
                if scr is None:
                    continue
                sx, sy, z, invw = scr
                res = scan_triangle(sx, sy, s.width, s.height)
                if res is None:
                    continue
                yy, xx, l0, l1, l2, area = res
                if area >= 0.0:      # descarte de faces traseiras
                    continue
                # Correção de perspectiva
                iw = l0 * invw[0] + l1 * invw[1] + l2 * invw[2]
                iw = np.maximum(iw, 1e-9)
                c0 = l0 * invw[0] / iw
                c1 = l1 * invw[1] / iw
                c2 = l2 * invw[2] / iw
                zz = l0 * z[0] + l1 * z[1] + l2 * z[2]
                keep = (zz >= -1.0) & (zz <= 1.0) & (zz < self.depth[yy, xx])
                if not keep.any():
                    continue
                yy, xx = yy[keep], xx[keep]
                zz = zz[keep]
                c0, c1, c2 = c0[keep], c1[keep], c2[keep]

                shade = (c0 * lv[0, 0] + c1 * lv[1, 0] + c2 * lv[2, 0]).astype(F)
                if tex is not None:
                    u = c0 * uvv[0, 0] + c1 * uvv[1, 0] + c2 * uvv[2, 0]
                    v = c0 * uvv[0, 1] + c1 * uvv[1, 1] + c2 * uvv[2, 1]
                    albedo = tex.sample(u, v,
                                        bilinear=(s.texture_filter == "bilinear"))
                else:
                    albedo = np.full((zz.size, 3), 0.78, dtype=F)

                base = albedo * tint
                amb = np.clip(base * F(s.ambient), 0.0, 1.0)
                full = np.clip(base * (F(s.ambient) + (1.0 - F(s.ambient))
                                       * shade[:, None]), 0.0, 1.0)
                self.depth[yy, xx] = zz
                self.col_lit[yy, xx] = full
                self.col_amb[yy, xx] = amb
                if self.capture_gbuffer:
                    wp = (c0[:, None] * wpv[0] + c1[:, None] * wpv[1]
                          + c2[:, None] * wpv[2])
                    self.gbuf[yy, xx] = wp
                st.frag_scene += int(zz.size)
                st.tris_scene_drawn += 1

    # ------------------------------------------------------------------ #
    def _clip_parts(self, clip_tri, attrs):
        """Recorte contra o plano próximo; devolve lista de (clip(3,4), attrs)."""
        if np.all(clip_tri[:, 3] > 1e-5):
            return [(clip_tri, attrs)]
        poly, pattrs = clip_near(clip_tri, attrs)
        if poly is None:
            return []
        out = []
        for k in range(1, poly.shape[0] - 1):
            sel = [0, k, k + 1]
            out.append((poly[sel], [a[sel] for a in pattrs]))
        return out

    # ------------------------------------------------------------------ #
    def _shadow_pass(self, vp: np.ndarray, st: SVStats) -> None:
        s = self.settings
        zfail = (s.algorithm == "zfail")

        ts = time.perf_counter()
        batches = []
        for prop in self.scene:
            if not prop.occluder:
                continue
            pw, _ = self._world(prop)
            sides, front, back = sv.build_volume(
                pw, prop.solid.indices, prop.solid.dir_edges,
                prop.solid.edge_face, prop.solid.edge_opp,
                self.light, s.extrude_len, caps=True)
            if sides.shape[0] == 0 and front.shape[0] == 0:
                continue
            closed = np.concatenate([x for x in (sides, front, back)
                                     if x.shape[0]], axis=0)
            if sv.point_in_volume(self.camera_eye, closed):
                st.volumes_with_camera += 1
            draw = closed if zfail else (sides if sides.shape[0] else None)
            if draw is None:
                continue
            batches.append(draw)
        st.ms_silhouette = (time.perf_counter() - ts) * 1000.0

        if not batches:
            return
        volume = np.concatenate(batches, axis=0)
        st.tris_volume = int(volume.shape[0])

        tr = time.perf_counter()
        clip = self._project(volume, vp)
        incr, decr = self.stencil.incr_op(), self.stencil.decr_op()

        for m in range(clip.shape[0]):
            for cp, _ in self._clip_parts(clip[m], []):
                scr = self._screen(cp)
                if scr is None:
                    continue
                sx, sy, z, _ = scr
                res = scan_triangle(sx, sy, s.width, s.height)
                if res is None:
                    continue
                yy, xx, l0, l1, l2, area = res
                front_facing = area < 0.0
                zz = l0 * z[0] + l1 * z[1] + l2 * z[2]
                inside = (zz >= -1.0) & (zz <= 1.0)
                if not inside.any():
                    continue
                yy, xx, zz = yy[inside], xx[inside], zz[inside]
                depth_pass = zz < self.depth[yy, xx]

                if zfail:
                    # Carmack's reverse: conta na FALHA do teste de profundidade
                    op = incr if not front_facing else decr
                    self.stencil.write(yy, xx, depth_pass, op_dppass="KEEP",
                                       op_dpfail=op)
                else:
                    op = incr if front_facing else decr
                    self.stencil.write(yy, xx, depth_pass, op_dppass=op,
                                       op_dpfail="KEEP")
                st.tris_volume_drawn += 1
        st.ms_stencil = (time.perf_counter() - tr) * 1000.0

    # ------------------------------------------------------------------ #
    def ground_truth(self):
        """Executa a referência por ray casting para o quadro atual.

        Retorna (mask_ref, mask_pred, visible). Exige que o quadro anterior
        tenha sido renderizado com capture_gbuffer=True.
        """
        from .truth import occluder_soup, shadow_mask_raycast
        vis = self.visible_mask
        ref = np.zeros_like(vis)
        if not vis.any():
            return ref, self.shadow_mask, vis
        pts = self.gbuf[vis]
        tris = occluder_soup(self.scene)
        ref[vis] = shadow_mask_raycast(pts, self.light, tris)
        return ref, self.shadow_mask, vis
