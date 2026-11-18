"""Rasterizador por software com stencil e volumes de sombra (`cpu-sw` e `cpu-fast`).

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
"""
from __future__ import annotations

import time

import numpy as np

from . import mathutil as mu
from .base_renderer import BaseRenderer
from .mesh import Prop
from .precision import apply_precision, quantize_depth
from .raster import scan_triangle, scan_triangle_fast
from .settings import BACKEND, BAYER4, SVSettings, SVStats
from .texture import make_textures

F = np.float32

__all__ = [
    "SVRenderer",
    "FastCPURenderer",
]


class SVRenderer(BaseRenderer):
    """Backend de rasterização por software em Python/NumPy (`cpu-sw`)."""

    backend_name: str = BACKEND
    _scan_fn = staticmethod(scan_triangle)

    def __init__(
        self,
        settings: SVSettings | None = None,
        *,
        scene_builder=None,
        light_fn=None,
    ):
        super().__init__(settings, scene_builder=scene_builder, light_fn=light_fn)
        self._alloc()

    def _alloc(self) -> None:
        """Aloca os buffers por pixel do framebuffer em software."""
        s = self.settings
        shape = (s.height, s.width)
        self.depth = np.ones(shape, dtype=F)           # Z-buffer em NDC [-1, 1] (inicializado em +1 = far)
        self.col_lit = np.zeros(shape + (3,), dtype=F)  # Cor com iluminação difusa + ambiente
        self.col_amb = np.zeros(shape + (3,), dtype=F)  # Cor apenas com luz ambiente (usada onde stencil != 0)
        self.gbuf = np.zeros(shape + (3,), dtype=F)     # Posição 3D no mundo (para ray casting de referência)
        self.stencil.resize(s.width, s.height)
        # Pré-calcula o ladrilho da matriz de Bayer 4x4 no tamanho do framebuffer
        tile = np.tile(BAYER4, (s.height // 4 + 1, s.width // 4 + 1))
        self._bayer_tile = tile[:s.height, :s.width][..., None].astype(F)

    def sync(self) -> None:
        """Reaplica configurações que exigem realocação."""
        s = self.settings
        if self.depth.shape != (s.height, s.width):
            self._alloc()
            self.shadow_mask = np.zeros((s.height, s.width), bool)
            self.visible_mask = np.zeros_like(self.shadow_mask)
        self.stencil.set_bits(s.stencil_bits)
        self.stencil.wrap = bool(s.stencil_wrap)
        tex = next(iter(self.textures.values()), None)
        if tex is None or tex.res != s.texture_res or tex.color_bits != s.color_bits:
            self.textures = make_textures(s.texture_res, s.color_bits)

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

        self.light = self._light_fn(t)
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
        st.ms_total = max(0.001, (time.perf_counter() - t0 - obs_overhead) * 1000.0)
        self._compute_memory_stats(st)
        self.stats = st
        return img

    def _resolve(self, buf: np.ndarray) -> np.ndarray:
        """Aplica quantização de profundidade de cor (`color_bits`) e dithering ordenado Bayer 4x4."""
        s = self.settings
        img = np.clip(buf, 0.0, 1.0)
        if s.color_bits < 8:
            levels = F((1 << s.color_bits) - 1)
            if s.dither:
                # Adiciona limiar espacial Bayer centrado em [-0.5/levels, +0.5/levels] antes do arredondamento
                img = np.clip(img + (self._bayer_tile - F(0.5)) / levels, 0.0, 1.0)
            img = np.round(img * levels) / levels
        return (img * 255.0 + 0.5).astype(np.uint8)

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

        # Iluminação por vértice: difusa de Lambert + atenuação quadrática suave com a distância
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

        # Coordenadas UV planares derivadas da posição de mundo
        uv = np.stack(
            [(tri_w[..., 0] + tri_w[..., 2]) * F(0.35),
             (tri_w[..., 1] + tri_w[..., 2] * 0.15) * F(0.35)],
            axis=2,
        ).astype(F)

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
                sx = np.round(sx / g) * g
                sy = np.round(sy / g) * g
            # Como o eixo Y de tela cresce para baixo (oposto ao NDC), triângulos frontais
            # anti-horários (CCW em NDC) possuem área 2D com sinal negativo (area < 0) em tela.
            area = ((sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0])
                    - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0]))
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
                    lit[m].reshape(3, 1), uv[m], tri_w[m], tex, tint, st,
                )

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
        res = self._scan_fn(sx, sy, s.width, s.height, cull_backface=True)
        if res is None:
            return
        yy, xx, l0, l1, l2, _ = res

        # Pesos baricêntricos corrigidos pela perspectiva (interpolação hiperbólica em 1/w)
        iw = l0 * invw[0] + l1 * invw[1] + l2 * invw[2]
        iw = np.maximum(iw, 1e-9)
        c0 = l0 * invw[0] / iw
        c1 = l1 * invw[1] / iw
        c2 = l2 * invw[2] / iw

        # Profundidade z_ndc interpolada linearmente em tela e quantizada em N bits (H4)
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
        full = np.clip(
            base * (F(s.ambient) + (F(1.0) - F(s.ambient)) * shade[:, None]),
            0.0, 1.0,
        )
        self.depth[yy, xx] = zz
        self.col_lit[yy, xx] = full
        self.col_amb[yy, xx] = amb
        if self.capture_gbuffer:
            wp = (c0[:, None] * wpv[0] + c1[:, None] * wpv[1] + c2[:, None] * wpv[2])
            self.gbuf[yy, xx] = wp
        st.frag_scene += int(zz.size)
        st.tris_scene_drawn += 1

    def _shadow_pass(self, vp: np.ndarray, st: SVStats, use_inf: bool = False) -> float:
        """Executa a construção de volumes e a passagem de stencil."""
        s = self.settings
        zfail = (s.algorithm == "zfail")
        ext_len = s.extrude_len if not use_inf else 40.0

        batches, occluder_cache, st.ms_silhouette = self._build_cached_volumes(
            vp, use_inf
        )
        st.volumes_with_camera, obs_overhead = self._count_volumes_with_camera(
            occluder_cache, ext_len
        )

        if not batches:
            return obs_overhead
        volume = np.concatenate(batches, axis=0)
        st.tris_volume = int(volume.shape[0])

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
                sx = np.round(sx / g) * g
                sy = np.round(sy / g) * g
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

        # Separa os triângulos de volume em duas filas (incremento antes de decremento),
        # reproduzindo a execução clássica em duas sub-passagens por orientação de face
        # (cull back / cull front) do OpenGL 1.4 e tornando determinística a avaliação
        # de saturação em contadores estreitos sem wrap (Hipótese H3).
        for sx_t, sy_t, z_t in incr_tris:
            self._raster_volume_tri(sx_t, sy_t, z_t, zfail, incr, decr, st)
        for sx_t, sy_t, z_t in decr_tris:
            self._raster_volume_tri(sx_t, sy_t, z_t, zfail, incr, decr, st)

        st.ms_stencil = (time.perf_counter() - tr) * 1000.0
        return obs_overhead

    def _raster_volume_tri(
        self, sx, sy, z, zfail: bool, incr: str, decr: str, st: SVStats
    ) -> None:
        s = self.settings
        res = self._scan_fn(sx, sy, s.width, s.height)
        if res is None:
            return
        yy, xx, l0, l1, l2, area = res
        front_facing = area < 0.0
        zz = l0 * z[0] + l1 * z[1] + l2 * z[2]

        # NV_depth_clamp: impede que fragmentos de volumes extrudados além do plano far
        # (z_ndc > 1.0) sejam descartados na variante z-fail.
        if zfail and s.depth_clamp:
            zz = np.minimum(zz, F(1.0))

        if s.depth_bits > 0:
            zz = quantize_depth(zz, s.depth_bits)

        inside = (zz >= -1.0) & (zz <= 1.0)
        if not inside.any():
            return
        yy, xx, zz = yy[inside], xx[inside], zz[inside]
        depth_pass = zz < self.depth[yy, xx]

        # Regra de contagem no stencil:
        # - z-fail (Carmack's Reverse): opera quando o fragmento do volume está ATRÁS da superfície
        #   visível (dpfail: zz >= depth), incrementando nas faces traseiras e decrementando nas frontais.
        # - z-pass: opera quando o fragmento do volume está À FRENTE da superfície visível
        #   (dppass: zz < depth), incrementando nas faces frontais e decrementando nas traseiras.
        if zfail:
            op = incr if not front_facing else decr
            self.stencil.write(yy, xx, depth_pass, op_dppass="KEEP", op_dpfail=op)
        else:
            op = incr if front_facing else decr
            self.stencil.write(yy, xx, depth_pass, op_dppass=op, op_dpfail="KEEP")
        st.tris_volume_drawn += 1


class FastCPURenderer(SVRenderer):
    """Backend CPU otimizado (`cpu-fast`) usando `scan_triangle_fast` (Numba se disponível)."""

    backend_name: str = "cpu-fast"
    _scan_fn = staticmethod(scan_triangle_fast)
