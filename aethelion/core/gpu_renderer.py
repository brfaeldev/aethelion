"""Backend GPU (`gpu-gl`) baseado em ModernGL / OpenGL.

Implementa o mesmo contrato de `SVRenderer` usando FBO offscreen com:
  1. Passagem de geometria (color_lit, color_amb, world_pos, depth24_stencil8)
  2. Passagem de volumes de sombra com stencil (`z-pass` / `z-fail`, wrap/saturado, depth clamp)
  3. Passagem de composição e pós-processamento (color_bits, dither Bayer 4x4)

Requer a dependência opcional `moderngl>=5.8` (`pip install -e .[gpu]`).
"""
from __future__ import annotations

import time
import numpy as np

from ..scenes.scene_crypt import Prop, build_crypt, light_position
from . import mathutil as mu
from . import shadow as sv
from .stencil import StencilBuffer
from .sv_renderer import SVRenderer, SVSettings, SVStats
from .texture import make_textures

try:
    import moderngl
    HAS_MODERNGL = True
except ImportError:  # pragma: no cover
    moderngl = None
    HAS_MODERNGL = False

F = np.float32

_GEOM_VERT = """
#version 330
in vec3 in_pos;
in vec3 in_norm;
uniform mat4 u_vp;
uniform vec2 u_res;
uniform int u_vertex_snap;
out vec3 v_world;
out vec3 v_norm;
void main() {
    vec4 clip = u_vp * vec4(in_pos, 1.0);
    if (u_vertex_snap > 0 && clip.w > 1e-5) {
        vec3 ndc = clip.xyz / clip.w;
        float g = float(u_vertex_snap);
        vec2 scr = vec2((ndc.x * 0.5 + 0.5) * u_res.x,
                        (1.0 - (ndc.y * 0.5 + 0.5)) * u_res.y);
        scr = round(scr * g) / g;
        ndc.x = (scr.x / u_res.x) * 2.0 - 1.0;
        ndc.y = (1.0 - scr.y / u_res.y) * 2.0 - 1.0;
        clip.xyz = ndc * clip.w;
    }
    v_world = in_pos;
    v_norm = in_norm;
    gl_Position = clip;
}
"""

_GEOM_FRAG = """
#version 330
in vec3 v_world;
in vec3 v_norm;
uniform vec3 u_light;
uniform vec3 u_tint;
uniform float u_ambient;
uniform int u_shading;     // 0=gouraud, 1=flat, 2=unlit
uniform int u_textured;
uniform int u_depth_bits;
uniform sampler2D u_tex;

layout(location = 0) out vec4 out_lit;
layout(location = 1) out vec4 out_amb;
layout(location = 2) out vec4 out_world;

void main() {
    vec3 n = normalize(v_norm);
    if (u_shading == 1) {
        vec3 dx = dFdx(v_world);
        vec3 dy = dFdy(v_world);
        vec3 fn = cross(dx, dy);
        if (dot(fn, fn) > 1e-12) n = normalize(fn);
    }
    float shade = 1.0;
    if (u_shading != 2) {
        vec3 ldir = u_light - v_world;
        float dist = length(ldir);
        ldir = ldir / max(dist, 1e-6);
        float lam = clamp(dot(n, ldir), 0.0, 1.0);
        float atten = 1.0 / (1.0 + 0.035 * dist * dist);
        shade = lam * atten;
    }
    vec3 albedo = vec3(0.78);
    if (u_textured != 0) {
        vec2 uv = vec2((v_world.x + v_world.z) * 0.35,
                       (v_world.y + v_world.z * 0.15) * 0.35);
        albedo = texture(u_tex, uv).rgb;
    }
    vec3 base = albedo * u_tint;
    vec3 amb = clamp(base * u_ambient, 0.0, 1.0);
    vec3 lit = clamp(base * (u_ambient + (1.0 - u_ambient) * shade), 0.0, 1.0);
    out_lit = vec4(lit, 1.0);
    out_amb = vec4(amb, 1.0);
    out_world = vec4(v_world, 1.0);
    if (u_depth_bits > 0) {
        float levels = float((1 << u_depth_bits) - 1);
        gl_FragDepth = round(gl_FragCoord.z * levels) / levels;
    } else {
        gl_FragDepth = gl_FragCoord.z;
    }
}
"""

_SV_VERT = """
#version 330
in vec4 in_pos;
uniform mat4 u_vp;
uniform int u_depth_clamp;
void main() {
    vec4 clip = u_vp * in_pos;
    if (u_depth_clamp != 0 && clip.w > 0.0 && clip.z > clip.w) {
        clip.z = clip.w;
    }
    gl_Position = clip;
}
"""

_SV_FRAG = """
#version 330
uniform int u_depth_bits;
uniform float u_sign;
out vec4 out_col;
void main() {
    if (u_depth_bits > 0) {
        float levels = float((1 << u_depth_bits) - 1);
        gl_FragDepth = round(gl_FragCoord.z * levels) / levels;
    } else {
        gl_FragDepth = gl_FragCoord.z;
    }
    out_col = vec4(u_sign, 0.0, 0.0, 1.0);
}
"""

_QUAD_VERT = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_COMP_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D u_lit;
uniform sampler2D u_amb;
uniform sampler2D u_shadow;
uniform int u_use_stencil;
uniform int u_color_bits;
uniform int u_dither;
out vec4 out_col;

float bayer4(ivec2 p) {
    int idx = (p.y & 3) * 4 + (p.x & 3);
    int m[16] = int[16](
        0, 8, 2, 10,
        12, 4, 14, 6,
        3, 11, 1, 9,
        15, 7, 13, 5
    );
    return float(m[idx]) / 16.0;
}

void main() {
    vec3 lit = texture(u_lit, v_uv).rgb;
    vec3 amb = texture(u_amb, v_uv).rgb;
    float sv = (u_use_stencil != 0) ? texture(u_shadow, v_uv).r : 0.0;
    vec3 col = (abs(sv) > 0.5) ? amb : lit;
    if (u_color_bits < 8) {
        float levels = float((1 << u_color_bits) - 1);
        if (u_dither != 0) {
            float b = bayer4(ivec2(gl_FragCoord.xy));
            col = clamp(col + (b - 0.5) / levels, 0.0, 1.0);
        }
        col = round(col * levels) / levels;
    }
    out_col = vec4(col, 1.0);
}
"""


class GLRenderer:
    """Backend GPU (`gpu-gl`) compatível com o protocolo de `SVRenderer`."""
    backend_name: str = "gpu-gl"

    def __init__(self, settings: SVSettings | None = None):
        if not HAS_MODERNGL:
            raise RuntimeError(
                "moderngl não está instalado. Instale com: pip install -e .[gpu]"
            )
        self.settings = settings or SVSettings()
        self.ctx = moderngl.create_standalone_context()
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
        self._vol_cache_key: tuple | None = None
        self._vol_cache_data: tuple | None = None
        self.capture_gbuffer = False
        self._last_vp: np.ndarray = mu.identity()

        self._prog_geom = self.ctx.program(vertex_shader=_GEOM_VERT, fragment_shader=_GEOM_FRAG)
        self._prog_sv = self.ctx.program(vertex_shader=_SV_VERT, fragment_shader=_SV_FRAG)
        self._prog_comp = self.ctx.program(vertex_shader=_QUAD_VERT, fragment_shader=_COMP_FRAG)

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
        self._prop_vaos: list[tuple[Prop, moderngl.VertexArray, int]] = []
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

        self.shadow_mask = np.zeros((self.settings.height, self.settings.width), bool)
        self.visible_mask = np.zeros_like(self.shadow_mask)
        self.depth = np.ones((self.settings.height, self.settings.width), dtype=F)
        self.col_lit = np.zeros((self.settings.height, self.settings.width, 3), dtype=F)
        self.col_amb = np.zeros((self.settings.height, self.settings.width, 3), dtype=F)
        self.gbuf = np.zeros((self.settings.height, self.settings.width, 3), dtype=F)

    def _alloc(self) -> None:
        s = self.settings
        w, h = int(s.width), int(s.height)
        for obj in (self._fbo, self._sv_fbo, self._comp_fbo, self._tex_lit,
                    self._tex_amb, self._tex_world, self._tex_shadow,
                    self._tex_ds, self._tex_out):
            if obj is not None:
                obj.release()
        self._tex_lit = self.ctx.texture((w, h), 4, dtype="f4")
        self._tex_amb = self.ctx.texture((w, h), 4, dtype="f4")
        self._tex_world = self.ctx.texture((w, h), 4, dtype="f4")
        self._tex_shadow = self.ctx.texture((w, h), 1, dtype="f4")
        self._tex_shadow.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._tex_ds = self.ctx.depth_renderbuffer((w, h))
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self._tex_lit, self._tex_amb, self._tex_world],
            depth_attachment=self._tex_ds,
        )
        self._sv_fbo = self.ctx.framebuffer(
            color_attachments=[self._tex_shadow],
            depth_attachment=self._tex_ds,
        )
        self._tex_out = self.ctx.texture((w, h), 4, dtype="f1")
        self._comp_fbo = self.ctx.framebuffer(color_attachments=[self._tex_out])
        self._zero_shadow = b"\x00" * (w * h * 4)
        self._alloc_size = (w, h)
        self.stencil.resize(w, h)

    def _upload_textures(self) -> None:
        for t in self._gl_textures.values():
            t.release()
        self._gl_textures.clear()
        filt = (moderngl.LINEAR if self.settings.texture_filter == "bilinear"
                else moderngl.NEAREST)
        for name, tex in self.textures.items():
            rgb = np.clip(tex.data * 255.0 + 0.5, 0, 255).astype(np.uint8)
            glt = self.ctx.texture((tex.res, tex.res), 3, rgb.tobytes())
            glt.filter = (filt, filt)
            glt.repeat_x = True
            glt.repeat_y = True
            self._gl_textures[name] = glt

    def _upload_scene(self) -> None:
        for _, vao, _ in self._prop_vaos:
            vao.release()
        self._prop_vaos.clear()
        for prop in self.scene:
            pw, nw = self._world(prop)
            idx = prop.solid.indices.ravel()
            verts = np.concatenate([pw[idx], nw[idx]], axis=1).astype(F)
            vbo = self.ctx.buffer(verts.tobytes())
            vao = self.ctx.vertex_array(
                self._prog_geom, [(vbo, "3f 3f", "in_pos", "in_norm")]
            )
            self._prop_vaos.append((prop, vao, int(idx.size)))

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
            filt = (moderngl.LINEAR if s.texture_filter == "bilinear"
                    else moderngl.NEAREST)
            for glt in self._gl_textures.values():
                glt.filter = (filt, filt)

    def rebuild_scene(self) -> None:
        self.scene = build_crypt(self.settings.detail)
        self._soups.clear()
        self._vol_cache_key = None
        self._vol_cache_data = None
        self._upload_scene()

    def _world(self, prop: Prop):
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

    def render(self, t: float = 0.0) -> np.ndarray:
        s = self.settings
        self.sync()
        st = SVStats()
        t0 = time.perf_counter()

        self.light = light_position(t)
        aspect = s.width / max(1, s.height)
        view = mu.look_at(self.camera_eye, self.camera_target)
        use_inf = bool(s.infinite_extrude or s.extrude_len <= 0.0)
        proj = (mu.perspective_infinite(s.fov, aspect, s.near) if use_inf
                else mu.perspective(s.fov, aspect, s.near, s.far))
        vp = (proj @ view).astype(F)
        self._last_vp = vp
        vp_gl = vp.T.astype(F).tobytes()

        # ---- 1. Geometria em GPU ---- #
        tg = time.perf_counter()
        self._fbo.use()
        self.ctx.viewport = (0, 0, s.width, s.height)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.front_face = "ccw"
        self.ctx.cull_face = "back"
        self.ctx.depth_mask = True
        self.ctx.color_mask = (True, True, True, True)
        self.ctx.clear(0.035, 0.040, 0.060, 0.0, depth=1.0)

        shading_id = {"gouraud": 0, "flat": 1, "unlit": 2}.get(s.shading, 0)
        self._prog_geom["u_vp"].write(vp_gl)
        self._prog_geom["u_res"].value = (float(s.width), float(s.height))
        self._prog_geom["u_vertex_snap"].value = int(s.vertex_snap)
        self._prog_geom["u_light"].value = tuple(float(v) for v in self.light)
        self._prog_geom["u_ambient"].value = float(s.ambient)
        self._prog_geom["u_shading"].value = shading_id
        self._prog_geom["u_textured"].value = 1 if s.textured else 0
        self._prog_geom["u_depth_bits"].value = int(s.depth_bits)
        self._prog_geom["u_tex"].value = 0

        for prop, vao, nverts in self._prop_vaos:
            st.tris_scene += nverts // 3
            st.tris_scene_drawn += nverts // 3
            self._prog_geom["u_tint"].value = tuple(float(v) for v in prop.tint)
            glt = self._gl_textures.get(prop.texture)
            if glt is not None:
                glt.use(0)
            vao.render(moderngl.TRIANGLES)
        self.ctx.finish()
        st.ms_geometry = (time.perf_counter() - tg) * 1000.0

        # ---- 2. Volumes de Sombra (construção CPU com cache) ---- #
        ts = time.perf_counter()
        ext_len = s.extrude_len if not use_inf else 40.0
        vol_tris = None
        if s.algorithm in ("zpass", "zfail"):
            zfail = (s.algorithm == "zfail")
            cache_key = (
                tuple(float(v) for v in self.light),
                s.algorithm, use_inf, float(ext_len), float(s.detail),
                len(self.scene), vp.tobytes() if use_inf else b"",
            )
            if self._vol_cache_key == cache_key and self._vol_cache_data is not None:
                vol_tris = self._vol_cache_data
            else:
                batches = []
                for prop in self.scene:
                    if not prop.occluder:
                        continue
                    pw, _ = self._world(prop)
                    if use_inf:
                        d_world = pw - self.light[None, :]
                        d_h = np.concatenate([d_world, np.zeros((pw.shape[0], 1), dtype=F)], axis=1)
                        w_inf_clip = (vp @ d_h.T).T[:, 3]
                        inf_ok = bool(np.all(w_inf_clip > 1e-3))
                        sides, front, back = sv.build_volume_homogeneous(
                            pw, prop.solid.indices, prop.solid.dir_edges,
                            prop.solid.edge_face, prop.solid.edge_opp,
                            self.light, length=(ext_len if inf_ok else 500.0),
                            infinite=inf_ok, caps=zfail)
                    else:
                        sides, front, back = sv.build_volume_homogeneous(
                            pw, prop.solid.indices, prop.solid.dir_edges,
                            prop.solid.edge_face, prop.solid.edge_opp,
                            self.light, length=ext_len, infinite=False, caps=zfail)
                    if zfail:
                        parts = [x for x in (sides, front, back) if x.shape[0]]
                        if parts:
                            batches.append(np.concatenate(parts, axis=0))
                    elif sides.shape[0]:
                        batches.append(sides)
                vol_tris = np.concatenate(batches, axis=0) if batches else np.zeros((0, 3, 4), dtype=F)
                self._vol_cache_key = cache_key
                self._vol_cache_data = vol_tris
        st.ms_silhouette = (time.perf_counter() - ts) * 1000.0

        # ---- 3. Passagem de Volumes / Contagem em GPU ---- #
        tr = time.perf_counter()
        self._tex_shadow.write(self._zero_shadow)
        self._sv_fbo.use()
        use_sv = s.algorithm in ("zpass", "zfail") and vol_tris is not None and vol_tris.shape[0] > 0
        if use_sv:
            zfail = (s.algorithm == "zfail")
            st.tris_volume = int(vol_tris.shape[0])
            st.tris_volume_drawn = int(vol_tris.shape[0])
            raw_v = vol_tris.reshape(-1, 4).astype(F).tobytes()
            if len(raw_v) > self._sv_vbo.size:
                self._sv_vbo.orphan(len(raw_v))
            self._sv_vbo.write(raw_v)

            self._prog_sv["u_vp"].write(vp_gl)
            self._prog_sv["u_depth_clamp"].value = 1 if (zfail and s.depth_clamp) else 0
            self._prog_sv["u_depth_bits"].value = int(s.depth_bits)

            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
            self.ctx.depth_mask = False
            self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            self.ctx.blend_equation = moderngl.FUNC_ADD
            self.ctx.depth_func = ">=" if zfail else "<"

            # Sub-passagem 1: faces que incrementam (+1)
            self.ctx.cull_face = "front" if zfail else "back"
            self._prog_sv["u_sign"].value = 1.0
            self._sv_vao.render(moderngl.TRIANGLES, vertices=vol_tris.shape[0] * 3)

            # Sub-passagem 2: faces que decrementam (-1)
            self.ctx.cull_face = "back" if zfail else "front"
            self._prog_sv["u_sign"].value = -1.0
            self._sv_vao.render(moderngl.TRIANGLES, vertices=vol_tris.shape[0] * 3)

            self.ctx.disable(moderngl.BLEND)
            self.ctx.depth_func = "<"
            self.ctx.cull_face = "back"
            self.ctx.depth_mask = True
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
        self._prog_comp["u_color_bits"].value = int(s.color_bits)
        self._prog_comp["u_dither"].value = 1 if s.dither else 0
        self._quad_vao.render(moderngl.TRIANGLE_STRIP)

        raw = self._comp_fbo.read(components=3, dtype="f1")
        img = np.frombuffer(raw, dtype=np.uint8).reshape(s.height, s.width, 3)
        img = np.ascontiguousarray(img[::-1])  # OpenGL origem embaixo -> topo

        if self.capture_gbuffer:
            wraw = self._fbo.read(attachment=2, components=4, dtype="f4")
            warr = np.frombuffer(wraw, dtype=F).reshape(s.height, s.width, 4)[::-1]
            self.gbuf = np.ascontiguousarray(warr[..., :3])
            self.visible_mask = warr[..., 3] > 0.5
            sraw = self._sv_fbo.read(attachment=0, components=1, dtype="f4")
            sarr = np.frombuffer(sraw, dtype=F).reshape(s.height, s.width)[::-1]
            self.shadow_mask = np.abs(sarr) > 0.5
            self.stencil.buf[:] = np.clip(np.round(sarr), 0, self.stencil.max_val).astype(np.int32)
            st.pixels_visible = int(np.count_nonzero(self.visible_mask))
            st.pixels_shadow = int(np.count_nonzero(self.shadow_mask & self.visible_mask))

        st.ms_composite = (time.perf_counter() - tc) * 1000.0
        st.ms_total = max(0.001, (time.perf_counter() - t0) * 1000.0)
        SVRenderer._compute_memory_stats(self, st)
        self.stats = st
        return img

    def render_volume_overlay(self, base_img: np.ndarray | None = None) -> np.ndarray:
        return SVRenderer.render_volume_overlay(self, base_img)

    def ground_truth(self):
        from ..evaluation.truth import evaluate_renderer_ground_truth
        return evaluate_renderer_ground_truth(self)

    def release(self) -> None:
        for obj in (self._fbo, self._sv_fbo, self._comp_fbo, self._tex_lit,
                    self._tex_amb, self._tex_world, self._tex_shadow,
                    self._tex_ds, self._tex_out,
                    self._quad_vbo, self._sv_vbo, self._quad_vao, self._sv_vao,
                    self._prog_geom, self._prog_sv, self._prog_comp):
            if obj is not None:
                obj.release()
        for t in self._gl_textures.values():
            t.release()
        for _, vao, _ in self._prop_vaos:
            vao.release()
        self.ctx.release()
