"""Backend GPU (`gpu-gl`) baseado em ModernGL / OpenGL.

Implementa o mesmo contrato de `SVRenderer` herdando de `BaseRenderer` e usando FBO offscreen com:
  1. Passagem de geometria (color_lit, color_amb, world_pos, depth_texture)
  2. Passagem de volumes de sombra em duas sub-passagens aditivas (`z-pass` / `z-fail`)
  3. Passagem de composição e pós-processamento (stencil N bits saturado/wrap, color_bits, dither Bayer 4x4)

Requer a dependência opcional `moderngl>=5.8` (`pip install -e .[gpu]`).
"""
from __future__ import annotations

import time

import numpy as np

from . import mathutil as mu
from .base_renderer import BaseRenderer
from .mesh import Prop
from .settings import SVSettings, SVStats
from .shaders import COMP_FRAG, GEOM_FRAG, GEOM_VERT, QUAD_VERT, SV_FRAG, SV_VERT
from .texture import make_textures

try:
    import moderngl
    HAS_MODERNGL = True
except ImportError:  # pragma: no cover
    moderngl = None
    HAS_MODERNGL = False

F = np.float32


class GLRenderer(BaseRenderer):
    """Backend GPU (`gpu-gl`) compatível com o protocolo de `SVRenderer`."""

    backend_name: str = "gpu-gl"

    def __init__(
        self,
        settings: SVSettings | None = None,
        *,
        scene_builder=None,
        light_fn=None,
    ):
        if not HAS_MODERNGL:
            raise RuntimeError(
                "moderngl não está instalado. Instale com: pip install -e .[gpu]"
            )
        super().__init__(settings, scene_builder=scene_builder, light_fn=light_fn)
        self.ctx = moderngl.create_standalone_context()

        self._prog_geom = self.ctx.program(vertex_shader=GEOM_VERT, fragment_shader=GEOM_FRAG)
        self._prog_sv = self.ctx.program(vertex_shader=SV_VERT, fragment_shader=SV_FRAG)
        self._prog_comp = self.ctx.program(vertex_shader=QUAD_VERT, fragment_shader=COMP_FRAG)

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=F)
        self._quad_vbo = self.ctx.buffer(quad.tobytes())
        self._quad_vao = self.ctx.vertex_array(
            self._prog_comp, [(self._quad_vbo, "2f", "in_pos")]
        )
        self._sv_vbo = self.ctx.buffer(reserve=4 * 1024 * 1024, dynamic=True)
        self._sv_vao = self.ctx.vertex_array(
            self._prog_sv, [(self._sv_vbo, "4f", "in_pos")]
        )

        self._gl_textures: dict[str, moderngl.Texture] = {}
        self._prop_vaos: list[tuple[Prop, moderngl.Buffer, moderngl.VertexArray, int]] = []
        self._fbo = None
        self._sv_fbo = None
        self._comp_fbo = None
        self._tex_lit = None
        self._tex_amb = None
        self._tex_world = None
        self._tex_shadow = None
        self._tex_ds = None
        self._tex_out = None

        self._alloc()
        self._upload_textures()
        self._upload_scene()

        self.depth = np.ones((self.settings.height, self.settings.width), dtype=F)
        self.col_lit = np.zeros((self.settings.height, self.settings.width, 3), dtype=F)
        self.col_amb = np.zeros((self.settings.height, self.settings.width, 3), dtype=F)
        self.gbuf = np.zeros((self.settings.height, self.settings.width, 3), dtype=F)

    def _alloc(self) -> None:
        """Aloca os 3 FBOs offscreen (geometria MRT, acumulação de stencil RG32F e composição)."""
        s = self.settings
        w, h = int(s.width), int(s.height)
        for obj in (
            self._fbo, self._sv_fbo, self._comp_fbo, self._tex_lit,
            self._tex_amb, self._tex_world, self._tex_shadow,
            self._tex_ds, self._tex_out,
        ):
            if obj is not None:
                obj.release()
        # FBO 1 (Geometria): grava cor iluminada, cor ambiente, posição 3D de mundo (G-buffer) e depth
        self._tex_lit = self.ctx.texture((w, h), 4, dtype="f4")
        self._tex_amb = self.ctx.texture((w, h), 4, dtype="f4")
        self._tex_world = self.ctx.texture((w, h), 4, dtype="f4")
        # FBO 2 (Volumes de Sombra): acumula (n_inc, n_dec) em float32 RG32F via blending aditivo,
        # compartilhando o mesmo depth attachment (_tex_ds) com depth_mask=False. Isso permite
        # emular e instrumentar contadores de 1..8 bits (saturados ou wrap) sem depender de um
        # stencil de hardware fixo em 8 bits.
        self._tex_shadow = self.ctx.texture((w, h), 2, dtype="f4")
        self._tex_shadow.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._tex_ds = self.ctx.depth_texture((w, h))
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self._tex_lit, self._tex_amb, self._tex_world],
            depth_attachment=self._tex_ds,
        )
        self._fbo.depth_mask = True
        self._sv_fbo = self.ctx.framebuffer(
            color_attachments=[self._tex_shadow],
            depth_attachment=self._tex_ds,
        )
        self._sv_fbo.depth_mask = False
        # FBO 3 (Composição): resolve sombra, quantização de cor e dithering Bayer 4x4
        self._tex_out = self.ctx.texture((w, h), 4, dtype="f1")
        self._comp_fbo = self.ctx.framebuffer(color_attachments=[self._tex_out])
        self._zero_shadow = b"\x00" * (w * h * 8)
        self._alloc_size = (w, h)
        self.stencil.resize(w, h)

    def _upload_textures(self) -> None:
        for t in self._gl_textures.values():
            t.release()
        self._gl_textures.clear()
        filt = (
            moderngl.LINEAR if self.settings.texture_filter == "bilinear"
            else moderngl.NEAREST
        )
        for name, tex in self.textures.items():
            rgb = np.clip(tex.data * 255.0 + 0.5, 0, 255).astype(np.uint8)
            glt = self.ctx.texture((tex.res, tex.res), 3, rgb.tobytes())
            glt.filter = (filt, filt)
            glt.repeat_x = True
            glt.repeat_y = True
            self._gl_textures[name] = glt

    def _upload_scene(self) -> None:
        for _, vbo, vao, _ in self._prop_vaos:
            vao.release()
            vbo.release()
        self._prop_vaos.clear()
        for prop in self.scene:
            pw, nw = self._world(prop)
            idx = prop.solid.indices.ravel()
            verts = np.concatenate([pw[idx], nw[idx]], axis=1).astype(F)
            vbo = self.ctx.buffer(verts.tobytes())
            vao = self.ctx.vertex_array(
                self._prog_geom, [(vbo, "3f 3f", "in_pos", "in_norm")]
            )
            self._prop_vaos.append((prop, vbo, vao, int(idx.size)))

    def sync(self) -> None:
        s = self.settings
        if self._alloc_size != (int(s.width), int(s.height)):
            self._alloc()
            self.depth = np.ones((s.height, s.width), dtype=F)
            self.col_lit = np.zeros((s.height, s.width, 3), dtype=F)
            self.col_amb = np.zeros((s.height, s.width, 3), dtype=F)
            self.gbuf = np.zeros((s.height, s.width, 3), dtype=F)
            self.shadow_mask = np.zeros((s.height, s.width), bool)
            self.visible_mask = np.zeros_like(self.shadow_mask)
        self.stencil.set_bits(s.stencil_bits)
        self.stencil.wrap = bool(s.stencil_wrap)
        tex = next(iter(self.textures.values()), None)
        if tex is None or tex.res != s.texture_res or tex.color_bits != s.color_bits:
            self.textures = make_textures(s.texture_res, s.color_bits)
            self._upload_textures()
        else:
            filt = (
                moderngl.LINEAR if s.texture_filter == "bilinear"
                else moderngl.NEAREST
            )
            for glt in self._gl_textures.values():
                glt.filter = (filt, filt)

    def rebuild_scene(self) -> None:
        super().rebuild_scene()
        self._upload_scene()

    def render(self, t: float = 0.0) -> np.ndarray:
        s = self.settings
        self.sync()
        st = SVStats()
        t0 = time.perf_counter()

        self.light = self._light_fn(t)
        aspect = s.width / max(1, s.height)
        view = mu.look_at(self.camera_eye, self.camera_target)
        use_inf = bool(s.infinite_extrude or s.extrude_len <= 0.0)
        proj = (
            mu.perspective_infinite(s.fov, aspect, s.near) if use_inf
            else mu.perspective(s.fov, aspect, s.near, s.far)
        )
        vp = (proj @ view).astype(F)
        self._last_vp = vp
        # Transpõe para column-major (convenção de matrizes uniformes em GLSL)
        vp_gl = vp.T.astype(F).tobytes()

        # ---- 1. Geometria em GPU ---- #
        tg = time.perf_counter()
        self._fbo.use()
        self._fbo.depth_mask = True
        self.ctx.viewport = (0, 0, s.width, s.height)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.front_face = "ccw"
        self.ctx.cull_face = "back"
        self.ctx.depth_func = "<"
        self.ctx.clear(0.035, 0.040, 0.060, 0.0, depth=1.0)

        shading_id = {"gouraud": 0, "flat": 1, "unlit": 2}.get(s.shading, 0)
        prec_id = {"float32": 0, "float16": 1, "fixed16_16": 2, "fixed12_4": 3}.get(
            s.precision, 0
        )
        self._prog_geom["u_vp"].write(vp_gl)
        self._prog_geom["u_res"].value = (float(s.width), float(s.height))
        self._prog_geom["u_vertex_snap"].value = int(s.vertex_snap)
        self._prog_geom["u_precision"].value = prec_id
        self._prog_geom["u_light"].value = tuple(float(v) for v in self.light)
        self._prog_geom["u_ambient"].value = float(s.ambient)
        self._prog_geom["u_shading"].value = shading_id
        if "u_textured" in self._prog_geom:
            self._prog_geom["u_textured"].value = 1 if s.textured else 0
        if "u_depth_bits" in self._prog_geom:
            self._prog_geom["u_depth_bits"].value = int(s.depth_bits)
        self._prog_geom["u_tex"].value = 0

        for prop, _, vao, nverts in self._prop_vaos:
            st.tris_scene += nverts // 3
            st.tris_scene_drawn += nverts // 3
            self._prog_geom["u_tint"].value = tuple(float(v) for v in prop.tint)
            glt = self._gl_textures.get(prop.texture)
            if glt is not None:
                glt.use(0)
            vao.render(moderngl.TRIANGLES)
        self.ctx.finish()
        st.ms_geometry = (time.perf_counter() - tg) * 1000.0

        # ---- 2. Volumes de Sombra (construção compartilhada via BaseRenderer) ---- #
        ext_len = s.extrude_len if not use_inf else 40.0
        vol_tris = None
        obs_overhead = 0.0
        if s.algorithm in ("zpass", "zfail"):
            batches, occluder_cache, st.ms_silhouette = self._build_cached_volumes(
                vp, use_inf
            )
            st.volumes_with_camera, obs_overhead = self._count_volumes_with_camera(
                occluder_cache, ext_len
            )
            vol_tris = (
                np.concatenate(batches, axis=0)
                if batches else np.zeros((0, 3, 4), dtype=F)
            )

        # ---- 3. Passagem de Volumes / Contagem em GPU ---- #
        tr = time.perf_counter()
        self._tex_shadow.write(self._zero_shadow)
        self._sv_fbo.use()
        self._sv_fbo.depth_mask = False
        use_sv = (
            s.algorithm in ("zpass", "zfail")
            and vol_tris is not None
            and vol_tris.shape[0] > 0
        )
        if use_sv:
            zfail = (s.algorithm == "zfail")
            st.tris_volume = int(vol_tris.shape[0])
            st.tris_volume_drawn = int(vol_tris.shape[0])
            raw_v = vol_tris.reshape(-1, 4).astype(F).tobytes()
            if len(raw_v) > self._sv_vbo.size:
                self._sv_vbo.orphan(len(raw_v))
            self._sv_vbo.write(raw_v)

            self._prog_sv["u_vp"].write(vp_gl)
            self._prog_sv["u_res"].value = (float(s.width), float(s.height))
            self._prog_sv["u_vertex_snap"].value = int(s.vertex_snap)
            self._prog_sv["u_precision"].value = prec_id
            self._prog_sv["u_depth_clamp"].value = 1 if (zfail and s.depth_clamp) else 0
            self._prog_sv["u_depth_bits"].value = int(s.depth_bits)

            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
            self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            self.ctx.blend_equation = moderngl.FUNC_ADD
            self.ctx.depth_func = ">=" if zfail else "<"

            # Sub-passagem 1: faces que incrementam (+1 no canal R)
            self.ctx.cull_face = "front" if zfail else "back"
            self._prog_sv["u_inc_dec"].value = (1.0, 0.0)
            self._sv_vao.render(moderngl.TRIANGLES, vertices=vol_tris.shape[0] * 3)

            # Sub-passagem 2: faces que decrementam (+1 no canal G)
            self.ctx.cull_face = "back" if zfail else "front"
            self._prog_sv["u_inc_dec"].value = (0.0, 1.0)
            self._sv_vao.render(moderngl.TRIANGLES, vertices=vol_tris.shape[0] * 3)

            self.ctx.disable(moderngl.BLEND)
            self.ctx.depth_func = "<"
            self.ctx.cull_face = "back"
        self.ctx.finish()
        st.ms_stencil = (time.perf_counter() - tr) * 1000.0

        # ---- 4. Composição & Resolve em GPU ---- #
        tc = time.perf_counter()
        self._comp_fbo.use()
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._tex_lit.use(0)
        self._tex_amb.use(1)
        self._tex_shadow.use(2)
        self._prog_comp["u_lit"].value = 0
        self._prog_comp["u_amb"].value = 1
        self._prog_comp["u_shadow"].value = 2
        self._prog_comp["u_use_stencil"].value = 1 if use_sv else 0
        self._prog_comp["u_stencil_bits"].value = int(np.clip(s.stencil_bits, 1, 8))
        self._prog_comp["u_stencil_wrap"].value = 1 if s.stencil_wrap else 0
        self._prog_comp["u_color_bits"].value = int(s.color_bits)
        self._prog_comp["u_dither"].value = 1 if s.dither else 0
        self._quad_vao.render(moderngl.TRIANGLE_STRIP)

        # Leitura do FBO de composição: inverte o eixo Y ([::-1]) pois a origem de FBOs
        # no OpenGL é no canto inferior-esquerdo, enquanto NumPy/Pygame usam topo-esquerdo.
        raw = self._comp_fbo.read(components=4, dtype="f1")
        rgba = np.frombuffer(raw, dtype=np.uint8).reshape(s.height, s.width, 4)[::-1]
        img = np.ascontiguousarray(rgba[..., :3])
        self.visible_mask = np.ascontiguousarray(rgba[..., 3] > 127)

        # Atualiza contadores exatos de stencil (1..8 bits, saturado vs wrap) a partir
        # dos canais acumulados (n_inc no canal R, n_dec no canal G). O tempo desta leitura
        # de telemetria é isolado em `obs_overhead` para não inflar `ms_total`.
        t_st_read0 = time.perf_counter()
        if use_sv:
            sraw = self._sv_fbo.read(attachment=0, components=2, dtype="f4")
            sarr = np.frombuffer(sraw, dtype=F).reshape(s.height, s.width, 2)[::-1]
            n_inc = np.rint(sarr[..., 0]).astype(np.int32)
            n_dec = np.rint(sarr[..., 1]).astype(np.int32)
            max_val = int(self.stencil.max_val)
            mod_val = max_val + 1
            if s.stencil_wrap:
                s1 = n_inc & max_val
                s_final = (n_inc - n_dec) & max_val
                wrap_inc = n_inc // mod_val
                wrap_dec = np.where(n_dec > s1, 1 + (n_dec - s1 - 1) // mod_val, 0)
                st.stencil_saturation = 0
                st.stencil_wrap_events = int((wrap_inc + wrap_dec).sum())
            else:
                s1 = np.minimum(n_inc, max_val)
                s_final = np.maximum(0, s1 - n_dec)
                sat_inc = np.maximum(0, n_inc - max_val)
                sat_dec = np.maximum(0, n_dec - s1)
                st.stencil_saturation = int((sat_inc + sat_dec).sum())
                st.stencil_wrap_events = 0
            self.stencil.buf[:] = s_final
            self.shadow_mask = (s_final != 0)
            st.frag_stencil = int((n_inc + n_dec).sum())
            st.stencil_max = int(s_final.max()) if s_final.size else 0
        else:
            self.stencil.buf[:] = 0
            self.shadow_mask[:] = False
            st.frag_stencil = 0
            st.stencil_saturation = 0
            st.stencil_wrap_events = 0
            st.stencil_max = 0

        if self.capture_gbuffer:
            wraw = self._fbo.read(attachment=2, components=4, dtype="f4")
            warr = np.frombuffer(wraw, dtype=F).reshape(s.height, s.width, 4)[::-1]
            self.gbuf = np.ascontiguousarray(warr[..., :3])
            self.visible_mask = np.ascontiguousarray(warr[..., 3] > 0.5)
        st.pixels_visible = int(np.count_nonzero(self.visible_mask))
        st.frag_scene = st.pixels_visible
        st.pixels_shadow = int(np.count_nonzero(self.shadow_mask & self.visible_mask))
        obs_overhead += (time.perf_counter() - t_st_read0)

        st.ms_composite = (time.perf_counter() - tc) * 1000.0
        st.ms_total = max(0.001, (time.perf_counter() - t0 - obs_overhead) * 1000.0)
        self._compute_memory_stats(st)
        self.stats = st
        return img

    def render_volume_overlay(self, base_img: np.ndarray | None = None) -> np.ndarray:
        s = self.settings
        if base_img is None:
            raw = self._comp_fbo.read(components=3, dtype="f1")
            img = np.frombuffer(raw, dtype=np.uint8).reshape(s.height, s.width, 3)
            base_img = np.ascontiguousarray(img[::-1])
        d_raw = self._tex_ds.read()
        d_01 = np.frombuffer(d_raw, dtype=F).reshape(s.height, s.width)[::-1]
        self.depth = np.ascontiguousarray(d_01 * F(2.0) - F(1.0))
        return super().render_volume_overlay(base_img)

    def release(self) -> None:
        for obj in (
            self._fbo, self._sv_fbo, self._comp_fbo, self._tex_lit,
            self._tex_amb, self._tex_world, self._tex_shadow,
            self._tex_ds, self._tex_out,
            self._quad_vbo, self._sv_vbo, self._quad_vao, self._sv_vao,
            self._prog_geom, self._prog_sv, self._prog_comp,
        ):
            if obj is not None:
                obj.release()
        for t in self._gl_textures.values():
            t.release()
        for _, vbo, vao, _ in self._prop_vaos:
            vao.release()
            vbo.release()
        self.ctx.release()
