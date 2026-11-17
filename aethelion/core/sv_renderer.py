"""Rasterizador por software com stencil e volumes de sombra.

Fluxo por quadro:

    1. Passagem de geometria  -> profundidade (quantizada por fragmento),
                                 cor iluminada, cor ambiente,
                                 posição de mundo (G-buffer opcional)
    2. Construção de volumes  -> silhueta + extrusão (finita ou homogênea w=0)
                                 + tampas (apenas em z-fail)
    3. Passagem de stencil    -> z-pass ou z-fail (com depth clamping opcional),
                                 sem escrita de cor ou de profundidade
    4. Composição             -> pixel sombreado recebe apenas o termo ambiente
    5. Resolve                -> quantização de cor e dithering Bayer pré-calculado

O backend é integralmente por software e identificado como 'cpu-sw' em todas as
saídas. Nenhum número produzido aqui descreve o desempenho de uma GPU real.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from ..scenes.scene_crypt import Prop, build_crypt, light_position
from . import mathutil as mu
from . import shadow as sv
from .clipping import clip_near
from .precision import apply_precision, quantize_depth
from .raster import scan_triangle
from .stencil import StencilBuffer
from .texture import make_textures

F = np.float32
BACKEND = "cpu-sw"

BAYER4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=F) / F(16.0)


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
    infinite_extrude: bool = False  # Everitt & Kilgard (2002): P_inf + extrusão homogênea w=0
    depth_clamp: bool = True        # NV_depth_clamp: satura z_ndc <= 1.0 em z-fail
    # Arquitetura do buffer de profundidade
    depth_bits: int = 0             # 0 = float32; senão quantiza em N bits por fragmento
    near: float = 0.15
    far: float = 60.0
    fov: float = 62.0
    # Precisão e sombreamento
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
    mem_buffers_kb: float = 0.0
    mem_geometry_kb: float = 0.0
    mem_total_kb: float = 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.ms_total if self.ms_total > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fps"] = self.fps
        return d


# --------------------------------------------------------------------------- #
class SVRenderer:
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
        self._soups: dict[int, tuple] = {}
        self.capture_gbuffer = False
        self._alloc()
        self.shadow_mask = np.zeros((self.settings.height, self.settings.width), bool)
        self.visible_mask = np.zeros_like(self.shadow_mask)
        self._last_vp: np.ndarray = mu.identity()

    # ------------------------------------------------------------------ #
    def _alloc(self) -> None:
        s = self.settings
        shape = (s.height, s.width)
        self.depth = np.ones(shape, dtype=F)
        self.col_lit = np.zeros(shape + (3,), dtype=F)
        self.col_amb = np.zeros(shape + (3,), dtype=F)
        self.gbuf = np.zeros(shape + (3,), dtype=F)
        self.stencil.resize(s.width, s.height)
        tile = np.tile(BAYER4, (s.height // 4 + 1, s.width // 4 + 1))
        self._bayer_tile = tile[:s.height, :s.width][..., None].astype(F)

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
        nm = mu.normal_matrix(prop.model)
        nw = (nm @ prop.solid.normals.T).T
        nw = (nw / np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-9)).astype(F)
        self._soups[key] = (pw, nw)
        return pw, nw

    def _compute_memory_stats(self, st: SVStats) -> None:
        """Calcula o perfilamento de memória arquitetural e geométrica (em KB)."""
        s = self.settings
        px = int(s.width) * int(s.height)
        color_bytes = px * 3.0 * (int(s.color_bits) / 8.0)
        depth_b = 32.0 if s.depth_bits <= 0 else float(s.depth_bits)
        depth_bytes = px * (depth_b / 8.0)
        stencil_bytes = px * (float(s.stencil_bits) / 8.0) if s.algorithm != "none" else 0.0
        tex_bytes = (
            len(self.textures) * (int(s.texture_res) ** 2) * 3.0 * (int(s.color_bits) / 8.0)
            if s.textured else 0.0
        )
        st.mem_buffers_kb = round((color_bytes + depth_bytes + stencil_bytes + tex_bytes) / 1024.0, 2)

        # Cada triângulo na cena tem 3 vértices (pos 3f + normal 3f) e no volume 3 vértices (3f finitos ou 4f homogêneos)
        vol_coords = 4 if (s.infinite_extrude or s.extrude_len <= 0.0) else 3
        scene_bytes = st.tris_scene * 3 * 6 * 4
        vol_bytes = st.tris_volume * 3 * vol_coords * 4
        st.mem_geometry_kb = round((scene_bytes + vol_bytes) / 1024.0, 2)
        st.mem_total_kb = round(st.mem_buffers_kb + st.mem_geometry_kb, 2)

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
        use_inf = bool(s.infinite_extrude or s.extrude_len <= 0.0)
        if use_inf:
            proj = mu.perspective_infinite(s.fov, aspect, s.near)
        else:
            proj = mu.perspective(s.fov, aspect, s.near, s.far)
        vp = (proj @ view).astype(F)
        self._last_vp = vp

        # ---- 1. Geometria ---- #
        tg = time.perf_counter()
        for prop in self.scene:
            self._draw_scene_prop(prop, vp, st)
        st.ms_geometry = (time.perf_counter() - tg) * 1000.0

        # ---- 2 e 3. Volumes e stencil ---- #
        obs_overhead = 0.0
        if s.algorithm in ("zpass", "zfail"):
            obs_overhead = self._shadow_pass(vp, st, use_inf=use_inf)

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
        # Desconta a instrumentação observacional (point_in_volume) de ms_total
        st.ms_total = max(0.001, (time.perf_counter() - t0 - obs_overhead) * 1000.0)
        self._compute_memory_stats(st)
        self.stats = st
        return img

    # ------------------------------------------------------------------ #
    def _resolve(self, buf: np.ndarray) -> np.ndarray:
        s = self.settings
        img = np.clip(buf, 0.0, 1.0)
        if s.color_bits < 8:
            levels = F((1 << s.color_bits) - 1)
            if s.dither:
                img = np.clip(img + (self._bayer_tile - F(0.5)) / levels,
                              0.0, 1.0)
            img = np.round(img * levels) / levels
        return (img * 255.0 + 0.5).astype(np.uint8)

    # ------------------------------------------------------------------ #
    def _project(self, world_tris: np.ndarray, vp: np.ndarray) -> np.ndarray:
        """(M,3,3) ou (M,3,4) mundo -> (M,3,4) clip, com precisão numérica aplicada."""
        s = self.settings
        if world_tris.shape[-1] == 4:
            ph = world_tris.reshape(-1, 4).astype(F, copy=False)
        else:
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
        return sx, sy, ndc[:, 2].astype(F), (1.0 / w).astype(F)

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

        W, H = s.width, s.height
        w_all = clip[..., 3]
        z_all = clip[..., 2]
        unclipped = (
            np.all(w_all > 1e-5, axis=1)
            & np.all((z_all + w_all) >= 0.0, axis=1)
            & np.all(np.isfinite(clip), axis=(1, 2))
        )

        # 1. Caminho rápido vetorizado para triângulos inteiramente à frente do near
        safe_ids = np.nonzero(unclipped)[0]
        if safe_ids.size:
            c_safe = clip[safe_ids]
            w_s = c_safe[..., 3]
            ndc = apply_precision((c_safe[..., :3] / w_s[..., None]).astype(F), s.precision)
            sx = (ndc[..., 0] * F(0.5) + F(0.5)) * F(W)
            sy = (F(1.0) - (ndc[..., 1] * F(0.5) + F(0.5))) * F(H)
            if s.vertex_snap > 0:
                g = F(s.vertex_snap)
                sx = np.round(sx * g) / g
                sy = np.round(sy * g) / g
            area = ((sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0])
                    - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0]))
            # Descarte antecipado em lote: faces traseiras (area >= 0) e fora da tela
            keep_tri = (
                np.all(np.isfinite(ndc), axis=(1, 2))
                & (area < -1e-9)
                & (np.floor(sx.max(axis=1) - 0.5) >= 0)
                & (np.ceil(sx.min(axis=1) - 0.5) <= W - 1)
                & (np.floor(sy.max(axis=1) - 0.5) >= 0)
                & (np.ceil(sy.min(axis=1) - 0.5) <= H - 1)
            )
            invw_s = (F(1.0) / w_s).astype(F)
            z_s = ndc[..., 2].astype(F)
            for k in np.nonzero(keep_tri)[0]:
                m = int(safe_ids[k])
                self._shade_scene_tri(
                    sx[k], sy[k], z_s[k], invw_s[k],
                    lit[m].reshape(3, 1), uv[m], tri_w[m], tex, tint, st)

        # 2. Caminho com recorte near para triângulos que cruzam o plano próximo
        for m in np.nonzero(~unclipped)[0]:
            parts = self._clip_parts(clip[m], [lit[m].reshape(3, 1), uv[m], tri_w[m]])
            for cp, (lv, uvv, wpv) in parts:
                scr = self._screen(cp)
                if scr is None:
                    continue
                sx, sy, z, invw = scr
                self._shade_scene_tri(sx, sy, z, invw, lv, uvv, wpv, tex, tint, st)

    def _shade_scene_tri(self, sx, sy, z, invw, lv, uvv, wpv, tex, tint, st: SVStats) -> None:
        s = self.settings
        res = scan_triangle(sx, sy, s.width, s.height, cull_backface=True)
        if res is None:
            return
        yy, xx, l0, l1, l2, _ = res

        # Correção de perspectiva
        iw = l0 * invw[0] + l1 * invw[1] + l2 * invw[2]
        iw = np.maximum(iw, 1e-9)
        c0 = l0 * invw[0] / iw
        c1 = l1 * invw[1] / iw
        c2 = l2 * invw[2] / iw

        # Profundidade interpolada e quantizada POR FRAGMENTO (Hipótese H4)
        zz = l0 * z[0] + l1 * z[1] + l2 * z[2]
        if s.depth_bits > 0:
            zz = quantize_depth(zz, s.depth_bits)

        keep = (zz >= -1.0) & (zz <= 1.0) & (zz < self.depth[yy, xx])
        if not keep.any():
            return
        yy, xx = yy[keep], xx[keep]
        zz = zz[keep]
        c0, c1, c2 = c0[keep], c1[keep], c2[keep]

        shade = (c0 * lv[0, 0] + c1 * lv[1, 0] + c2 * lv[2, 0]).astype(F)
        if tex is not None:
            u = c0 * uvv[0, 0] + c1 * uvv[1, 0] + c2 * uvv[2, 0]
            v = c0 * uvv[0, 1] + c1 * uvv[1, 1] + c2 * uvv[2, 1]
            albedo = tex.sample(u, v, bilinear=(s.texture_filter == "bilinear"))
        else:
            albedo = np.full((zz.size, 3), 0.78, dtype=F)

        base = albedo * tint
        amb = np.clip(base * F(s.ambient), 0.0, 1.0)
        full = np.clip(base * (F(s.ambient) + (F(1.0) - F(s.ambient)) * shade[:, None]),
                       0.0, 1.0)
        self.depth[yy, xx] = zz
        self.col_lit[yy, xx] = full
        self.col_amb[yy, xx] = amb
        if self.capture_gbuffer:
            wp = (c0[:, None] * wpv[0] + c1[:, None] * wpv[1] + c2[:, None] * wpv[2])
            self.gbuf[yy, xx] = wp
        st.frag_scene += int(zz.size)
        st.tris_scene_drawn += 1

    # ------------------------------------------------------------------ #
    def _clip_parts(self, clip_tri, attrs):
        """Recorte contra o plano próximo; devolve lista de (clip(3,4), attrs)."""
        w = clip_tri[:, 3]
        z = clip_tri[:, 2]
        if np.all(w > 1e-5) and np.all((z + w) >= 0.0):
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
    def _shadow_pass(self, vp: np.ndarray, st: SVStats,
                     use_inf: bool = False) -> float:
        """Executa a construção de volumes e a passagem de stencil.

        Retorna o tempo (em segundos) gasto apenas na instrumentação
        observacional (`point_in_volume`), para descontá-lo de `ms_total`.
        """
        s = self.settings
        zfail = (s.algorithm == "zfail")
        ext_len = s.extrude_len if not use_inf else 40.0

        # 2a. Construção de silhuetas e volumes do algoritmo (cronometrada)
        ts = time.perf_counter()
        batches = []
        occluder_cache = []
        for prop in self.scene:
            if not prop.occluder:
                continue
            pw, _ = self._world(prop)
            if use_inf:
                # Verifica se todos os raios extrudados apontam para o semiespaço frontal
                # da câmera (w_clip > 0). Para triângulos da tampa traseira no infinito (w_e = 0),
                # P_inf satisfaz z_clip = w_clip, logo o cruzamento com z_clip + w_clip = 0
                # ocorre em w_clip = 0 (plano do olho), exigindo rasterização homogênea 4D sem
                # divisão perspectiva. Quando algum raio aponta para trás (w_inf_clip <= 1e-3),
                # ancora-se a extrusão em distância finita longa (length=500.0) sob P_inf.
                d_world = pw - self.light[None, :]
                d_h = np.concatenate([d_world, np.zeros((pw.shape[0], 1), dtype=F)], axis=1)
                w_inf_clip = (vp @ d_h.T).T[:, 3]
                if np.all(w_inf_clip > 1e-3):
                    sides, front, back = sv.build_volume_homogeneous(
                        pw, prop.solid.indices, prop.solid.dir_edges,
                        prop.solid.edge_face, prop.solid.edge_opp,
                        self.light, length=ext_len, infinite=True, caps=zfail)
                else:
                    sides, front, back = sv.build_volume_homogeneous(
                        pw, prop.solid.indices, prop.solid.dir_edges,
                        prop.solid.edge_face, prop.solid.edge_opp,
                        self.light, length=500.0, infinite=False, caps=zfail)
                occluder_cache.append((prop, pw, None))
            else:
                sides, front, back = sv.build_volume(
                    pw, prop.solid.indices, prop.solid.dir_edges,
                    prop.solid.edge_face, prop.solid.edge_opp,
                    self.light, ext_len, caps=zfail)
                closed_fin = (sides, front, back) if zfail else None
                occluder_cache.append((prop, pw, closed_fin))
            if zfail:
                parts = [x for x in (sides, front, back) if x.shape[0]]
                if parts:
                    batches.append(np.concatenate(parts, axis=0))
            else:
                if sides.shape[0]:
                    batches.append(sides)
        st.ms_silhouette = (time.perf_counter() - ts) * 1000.0

        # 2b. Instrumentação observacional (volumes_with_camera)
        # Reaproveita os volumes finitos já construídos quando disponíveis e
        # executa FORA de ms_silhouette (descontado de ms_total).
        t_obs0 = time.perf_counter()
        vis_len = ext_len if ext_len > 0 else 40.0
        for prop, pw, closed_fin in occluder_cache:
            if closed_fin is not None:
                sides3, front3, back3 = closed_fin
            else:
                sides3, front3, back3 = sv.build_volume(
                    pw, prop.solid.indices, prop.solid.dir_edges,
                    prop.solid.edge_face, prop.solid.edge_opp,
                    self.light, vis_len, caps=True)
            if sides3.shape[0] == 0 and front3.shape[0] == 0:
                continue
            closed3 = np.concatenate([x for x in (sides3, front3, back3)
                                      if x.shape[0]], axis=0)
            if sv.point_in_volume(self.camera_eye, closed3):
                st.volumes_with_camera += 1
        obs_overhead = time.perf_counter() - t_obs0

        if not batches:
            return obs_overhead
        volume = np.concatenate(batches, axis=0)
        st.tris_volume = int(volume.shape[0])

        # 3. Passagem de Stencil (cronometrada)
        tr = time.perf_counter()
        clip = self._project(volume, vp)
        incr, decr = self.stencil.incr_op(), self.stencil.decr_op()
        W, H = s.width, s.height

        w_all = clip[..., 3]
        z_all = clip[..., 2]
        unclipped = (
            np.all(w_all > 1e-5, axis=1)
            & np.all((z_all + w_all) >= 0.0, axis=1)
            & np.all(np.isfinite(clip), axis=(1, 2))
        )

        # 3a. Triagem vetorizada para triângulos de volume totalmente à frente do near
        incr_tris = []
        decr_tris = []
        safe_ids = np.nonzero(unclipped)[0]
        if safe_ids.size:
            c_safe = clip[safe_ids]
            w_s = c_safe[..., 3]
            ndc = apply_precision((c_safe[..., :3] / w_s[..., None]).astype(F), s.precision)
            sx = (ndc[..., 0] * F(0.5) + F(0.5)) * F(W)
            sy = (F(1.0) - (ndc[..., 1] * F(0.5) + F(0.5))) * F(H)
            if s.vertex_snap > 0:
                g = F(s.vertex_snap)
                sx = np.round(sx * g) / g
                sy = np.round(sy * g) / g
            area = ((sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0])
                    - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0]))
            keep_tri = (
                np.all(np.isfinite(ndc), axis=(1, 2))
                & (np.abs(area) >= 1e-9)
                & (np.floor(sx.max(axis=1) - 0.5) >= 0)
                & (np.ceil(sx.min(axis=1) - 0.5) <= W - 1)
                & (np.floor(sy.max(axis=1) - 0.5) >= 0)
                & (np.ceil(sy.min(axis=1) - 0.5) <= H - 1)
            )
            z_s = ndc[..., 2].astype(F)
            for k in np.nonzero(keep_tri)[0]:
                front_facing = bool(area[k] < 0.0)
                is_incr = (not front_facing) if zfail else front_facing
                (incr_tris if is_incr else decr_tris).append((sx[k], sy[k], z_s[k]))

        # 3b. Triângulos de volume que cruzam o plano near
        for m in np.nonzero(~unclipped)[0]:
            for cp, _ in self._clip_parts(clip[m], []):
                scr = self._screen(cp)
                if scr is None:
                    continue
                sx_c, sy_c, z_c, _ = scr
                a_c = ((sx_c[1] - sx_c[0]) * (sy_c[2] - sy_c[0])
                       - (sx_c[2] - sx_c[0]) * (sy_c[1] - sy_c[0]))
                if not np.isfinite(a_c) or abs(a_c) < 1e-9:
                    continue
                front_facing = bool(a_c < 0.0)
                is_incr = (not front_facing) if zfail else front_facing
                (incr_tris if is_incr else decr_tris).append((sx_c, sy_c, z_c))

        # 3c. Ordem de duas passagens (Everitt & Kilgard, 2002, Seção 2.1):
        # primeiro as faces que incrementam o stencil (INCR/INCR_WRAP), depois as
        # que decrementam (DECR/DECR_WRAP), evitando underflow espúrio em 0.
        for sx_t, sy_t, z_t in incr_tris:
            self._raster_volume_tri(sx_t, sy_t, z_t, zfail, incr, decr, st)
        for sx_t, sy_t, z_t in decr_tris:
            self._raster_volume_tri(sx_t, sy_t, z_t, zfail, incr, decr, st)

        st.ms_stencil = (time.perf_counter() - tr) * 1000.0
        return obs_overhead

    def _raster_volume_tri(self, sx, sy, z, zfail: bool,
                           incr: str, decr: str, st: SVStats) -> None:
        s = self.settings
        res = scan_triangle(sx, sy, s.width, s.height)
        if res is None:
            return
        yy, xx, l0, l1, l2, area = res
        front_facing = area < 0.0
        zz = l0 * z[0] + l1 * z[1] + l2 * z[2]

        # Depth clamping no plano distante (NV_depth_clamp, Everitt & Kilgard 2002)
        if zfail and s.depth_clamp:
            zz = np.minimum(zz, F(1.0))

        # Quantização por fragmento quando depth_bits > 0 (Hipótese H4)
        # Um leve bias de meia unidade em z-buffer estreito evita colisão exata
        # na tampa frontal mantendo a degradação real nas vistas rasantes.
        if s.depth_bits > 0:
            zz = quantize_depth(zz, s.depth_bits)

        inside = (zz >= -1.0) & (zz <= 1.0)
        if not inside.any():
            return
        yy, xx, zz = yy[inside], xx[inside], zz[inside]
        depth_pass = zz < self.depth[yy, xx]

        if zfail:
            # Carmack's reverse: conta na FALHA do teste de profundidade (zz >= depth)
            op = incr if not front_facing else decr
            self.stencil.write(yy, xx, depth_pass, op_dppass="KEEP", op_dpfail=op)
        else:
            op = incr if front_facing else decr
            self.stencil.write(yy, xx, depth_pass, op_dppass=op, op_dpfail="KEEP")
        st.tris_volume_drawn += 1

    # ------------------------------------------------------------------ #
    def render_volume_overlay(self, base_img: np.ndarray | None = None) -> np.ndarray:
        """Gera vista didática 3D dos Volumes de Sombra e Arestas de Silhueta.

        Inspirada nas Figuras 2, 3 e 5 de Everitt & Kilgard (2002):
        - Faces frontais visíveis dos volumes de sombra em âmbar semitransparente;
        - Faces traseiras visíveis dos volumes em azul-ciano suave;
        - Arestas laterais extrudadas em amarelo e arestas de silhueta em laranja/ciano.
        """
        if base_img is None:
            base_img = self._resolve(np.where(self.shadow_mask[..., None],
                                              self.col_amb, self.col_lit))
        out = base_img.astype(F).copy()
        s = self.settings
        W, H = s.width, s.height
        vp = self._last_vp

        # Extrai laterais dos volumes e segmentos de silhueta sob demanda (apenas na vista 3D)
        vol_sides = []
        sil_segments = []
        use_inf = bool(s.infinite_extrude or s.extrude_len <= 0.0)
        vis_len = s.extrude_len if s.extrude_len > 0 else 40.0
        for prop in self.scene:
            if not prop.occluder:
                continue
            pw, _ = self._world(prop)
            sides_h, _, _ = sv.build_volume_homogeneous(
                pw, prop.solid.indices, prop.solid.dir_edges,
                prop.solid.edge_face, prop.solid.edge_opp,
                self.light, length=vis_len, infinite=use_inf, caps=False)
            if sides_h.shape[0]:
                vol_sides.append(sides_h)
            seg = sv.extract_silhouette_segments(
                pw, prop.solid.indices, prop.solid.dir_edges,
                prop.solid.edge_face, prop.solid.edge_opp, self.light)
            if seg.shape[0]:
                sil_segments.append(seg)

        # 1. Preenchimento semitransparente das faces laterais dos volumes
        if vol_sides:
            sides = np.concatenate(vol_sides, axis=0)
            clip = self._project(sides, vp)
            front_tint = np.array([255.0, 200.0, 50.0], dtype=F)
            back_tint = np.array([60.0, 170.0, 255.0], dtype=F)

            w_all = clip[..., 3]
            z_all = clip[..., 2]
            unclipped = (
                np.all(w_all > 1e-5, axis=1)
                & np.all((z_all + w_all) >= 0.0, axis=1)
                & np.all(np.isfinite(clip), axis=(1, 2))
            )
            ov_tris = []
            safe_ids = np.nonzero(unclipped)[0]
            if safe_ids.size:
                c_safe = clip[safe_ids]
                w_s = c_safe[..., 3]
                ndc = apply_precision((c_safe[..., :3] / w_s[..., None]).astype(F), s.precision)
                sx_b = (ndc[..., 0] * F(0.5) + F(0.5)) * F(W)
                sy_b = (F(1.0) - (ndc[..., 1] * F(0.5) + F(0.5))) * F(H)
                keep_tri = (
                    np.all(np.isfinite(ndc), axis=(1, 2))
                    & (np.floor(sx_b.max(axis=1) - 0.5) >= 0)
                    & (np.ceil(sx_b.min(axis=1) - 0.5) <= W - 1)
                    & (np.floor(sy_b.max(axis=1) - 0.5) >= 0)
                    & (np.ceil(sy_b.min(axis=1) - 0.5) <= H - 1)
                )
                z_b = ndc[..., 2].astype(F)
                for k in np.nonzero(keep_tri)[0]:
                    ov_tris.append((sx_b[k], sy_b[k], z_b[k]))

            for m in np.nonzero(~unclipped)[0]:
                for cp, _ in self._clip_parts(clip[m], []):
                    scr = self._screen(cp)
                    if scr is not None:
                        ov_tris.append((scr[0], scr[1], scr[2]))

            for sx, sy, z in ov_tris:
                res = scan_triangle(sx, sy, W, H)
                if res is None:
                    continue
                yy, xx, l0, l1, l2, area = res
                zz = l0 * z[0] + l1 * z[1] + l2 * z[2]
                vis = (zz >= -1.0) & (zz <= 1.0) & (zz < self.depth[yy, xx])
                if not vis.any():
                    continue
                yy, xx = yy[vis], xx[vis]
                if area < 0.0:
                    out[yy, xx] = out[yy, xx] * F(0.84) + front_tint * F(0.16)
                else:
                    out[yy, xx] = out[yy, xx] * F(0.90) + back_tint * F(0.10)
                for i in range(3):
                    j = (i + 1) % 3
                    self._draw_line_2d(out, sx[i], sy[i], sx[j], sy[j],
                                       (235, 185, 45), alpha=0.45,
                                       z0=float(z[i]), z1=float(z[j]),
                                       depth_buf=self.depth)

        # 2. Arestas de silhueta em destaque (ciano brilhante) com recorte near e teste de profundidade
        if sil_segments:
            segs = np.concatenate(sil_segments, axis=0)  # (S, 2, 3)
            pts = segs.reshape(-1, 3)
            ph = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=F)], axis=1)
            clip_s = (vp @ ph.T).T.astype(F).reshape(-1, 2, 4)
            for k in range(clip_s.shape[0]):
                p0, p1 = clip_s[k, 0].copy(), clip_s[k, 1].copy()
                d0, d1 = float(p0[2] + p0[3]), float(p1[2] + p1[3])
                if d0 < 0.0 and d1 < 0.0:
                    continue
                if d0 < 0.0 or d1 < 0.0:
                    t_c = d0 / (d0 - d1) if abs(d0 - d1) > 1e-9 else 0.5
                    pm = p0 + (p1 - p0) * F(t_c)
                    if d0 < 0.0:
                        p0 = pm
                    else:
                        p1 = pm
                w0, w1 = float(p0[3]), float(p1[3])
                if w0 <= 1e-4 or w1 <= 1e-4:
                    continue
                ndc0, ndc1 = p0[:3] / w0, p1[:3] / w1
                x0 = (ndc0[0] * 0.5 + 0.5) * W
                y0 = (1.0 - (ndc0[1] * 0.5 + 0.5)) * H
                x1 = (ndc1[0] * 0.5 + 0.5) * W
                y1 = (1.0 - (ndc1[1] * 0.5 + 0.5)) * H
                self._draw_line_2d(out, x0, y0, x1, y1, (0, 255, 220), alpha=0.95,
                                   z0=float(ndc0[2]), z1=float(ndc1[2]),
                                   depth_buf=self.depth)

        # 3. Marcador da posição da luz pontual
        lh = np.array([self.light[0], self.light[1], self.light[2], 1.0], dtype=F)
        lc = vp @ lh
        if lc[3] > 0.1:
            lndc = lc[:3] / lc[3]
            lx = int((lndc[0] * 0.5 + 0.5) * W)
            ly = int((1.0 - (lndc[1] * 0.5 + 0.5)) * H)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    py, px = ly + dy, lx + dx
                    if 0 <= py < H and 0 <= px < W:
                        out[py, px] = (255, 255, 120)

        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _draw_line_2d(buf: np.ndarray, x0: float, y0: float,
                      x1: float, y1: float, color: tuple, alpha: float = 0.8,
                      z0: float | None = None, z1: float | None = None,
                      depth_buf: np.ndarray | None = None) -> None:
        h, w, _ = buf.shape
        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
            return
        steps = int(max(abs(x1 - x0), abs(y1 - y0)))
        if steps <= 0 or steps > 2000:
            return
        xs = np.linspace(x0, x1, steps + 1).astype(np.int32)
        ys = np.linspace(y0, y1, steps + 1).astype(np.int32)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not ok.any():
            return
        if depth_buf is not None and z0 is not None and z1 is not None:
            zs = np.linspace(z0, z1, steps + 1, dtype=F)[ok]
            xs, ys = xs[ok], ys[ok]
            vis = (zs >= -1.0) & (zs <= depth_buf[ys, xs] + F(2.5e-3))
            if not vis.any():
                return
            xs, ys = xs[vis], ys[vis]
        else:
            xs, ys = xs[ok], ys[ok]
        col = np.array(color, dtype=F)
        buf[ys, xs] = buf[ys, xs] * F(1.0 - alpha) + col * F(alpha)

    # ------------------------------------------------------------------ #
    def ground_truth(self):
        """Atalho de conveniência para `evaluation.truth.evaluate_renderer_ground_truth`."""
        from ..evaluation.truth import evaluate_renderer_ground_truth
        return evaluate_renderer_ground_truth(self)
