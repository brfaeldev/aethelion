"""
Programas de Shader GLSL para o Backend OpenGL (shaders.py)

Contém o código-fonte GLSL 330 dos shaders de vértice e fragmento utilizados
pelo backend `gpu-gl`, reproduzindo em hardware a quantização de precisão
(`float16`, `fixed16_16`, `fixed12_4`), `vertex_snap`, sombreamento
Gouraud/Flat/Unlit, depth clamping, stencil de N bits e dithering Bayer 4x4.
"""
from __future__ import annotations

GEOM_VERT = """
#version 330
in vec3 in_pos;
in vec3 in_norm;
uniform mat4 u_vp;
uniform vec2 u_res;
uniform int u_vertex_snap;
uniform int u_precision;
uniform vec3 u_light;
out vec3 v_world;
out vec3 v_norm;
out float v_shade;

// Emula os modos de precisão aritmética do pipeline:
//   0 = float32 (nativo)
//   1 = float16 (quantização da mantissa de 10 bits via degrau 2^(floor(log2(|x|)) - 10))
//   2 = fixed16_16 (ponto fixo Q16.16, resolução 1/65536)
//   3 = fixed12_4 (ponto fixo Q12.4 estilo PS1/GTE, resolução 1/16)
vec4 apply_prec4(vec4 v, int mode) {
    if (mode == 1) {
        vec4 c = clamp(v, -65500.0, 65500.0);
        vec4 a = abs(c);
        vec4 step_val = exp2(floor(log2(max(a, vec4(1e-8)))) - 10.0);
        return round(c / step_val) * step_val;
    }
    if (mode == 2) {
        vec4 c = clamp(v, -32767.0, 32767.0);
        return round(c * 65536.0) / 65536.0;
    }
    if (mode == 3) {
        vec4 c = clamp(v, -2047.0, 2047.0);
        return round(c * 16.0) / 16.0;
    }
    return v;
}

void main() {
    vec4 clip = apply_prec4(u_vp * vec4(in_pos, 1.0), u_precision);
    if (clip.w > 1e-5) {
        vec3 ndc = apply_prec4(vec4(clip.xyz / clip.w, 0.0), u_precision).xyz;
        if (u_vertex_snap > 0) {
            float g = float(u_vertex_snap);
            vec2 scr = vec2((ndc.x * 0.5 + 0.5) * u_res.x,
                            (1.0 - (ndc.y * 0.5 + 0.5)) * u_res.y);
            scr = round(scr / g) * g;
            ndc.x = (scr.x / u_res.x) * 2.0 - 1.0;
            ndc.y = (1.0 - scr.y / u_res.y) * 2.0 - 1.0;
        }
        clip.xyz = ndc * clip.w;
    }
    vec3 ldir = u_light - in_pos;
    float dist = length(ldir);
    ldir = ldir / max(dist, 1e-9);
    float lam = clamp(dot(normalize(in_norm), ldir), 0.0, 1.0);
    float atten = 1.0 / (1.0 + 0.035 * dist * dist);
    v_shade = lam * atten;
    v_world = in_pos;
    v_norm = in_norm;
    gl_Position = clip;
}
"""

GEOM_FRAG = """
#version 330
in vec3 v_world;
in vec3 v_norm;
in float v_shade;
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
    float shade = v_shade;
    if (u_shading == 2) {
        shade = 1.0;
    } else if (u_shading == 1) {
        // Sombreamento flat: reconstrói a normal geométrica da face usando derivadas
        // de tela (dFdx/dFdy) sobre a posição de mundo interpolada.
        vec3 n = normalize(v_norm);
        vec3 dx = dFdx(v_world);
        vec3 dy = dFdy(v_world);
        vec3 fn = cross(dx, dy);
        if (dot(fn, fn) > 1e-12) n = normalize(fn);
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
    // Quantização por fragmento da profundidade em N bits (Hipótese H4)
    if (u_depth_bits > 0 && u_depth_bits < 24) {
        float levels = exp2(float(u_depth_bits)) - 1.0;
        gl_FragDepth = round(gl_FragCoord.z * levels) / levels;
    } else {
        gl_FragDepth = gl_FragCoord.z;
    }
}
"""

SV_VERT = """
#version 330
in vec4 in_pos;
uniform mat4 u_vp;
uniform vec2 u_res;
uniform int u_vertex_snap;
uniform int u_precision;
uniform int u_depth_clamp;

// Emula os modos de precisão aritmética do pipeline:
//   0 = float32 (nativo)
//   1 = float16 (quantização da mantissa de 10 bits via degrau 2^(floor(log2(|x|)) - 10))
//   2 = fixed16_16 (ponto fixo Q16.16, resolução 1/65536)
//   3 = fixed12_4 (ponto fixo Q12.4 estilo PS1/GTE, resolução 1/16)
vec4 apply_prec4(vec4 v, int mode) {
    if (mode == 1) {
        vec4 c = clamp(v, -65500.0, 65500.0);
        vec4 a = abs(c);
        vec4 step_val = exp2(floor(log2(max(a, vec4(1e-8)))) - 10.0);
        return round(c / step_val) * step_val;
    }
    if (mode == 2) {
        vec4 c = clamp(v, -32767.0, 32767.0);
        return round(c * 65536.0) / 65536.0;
    }
    if (mode == 3) {
        vec4 c = clamp(v, -2047.0, 2047.0);
        return round(c * 16.0) / 16.0;
    }
    return v;
}

void main() {
    vec4 clip = apply_prec4(u_vp * in_pos, u_precision);
    if (clip.w > 1e-5) {
        vec3 ndc = apply_prec4(vec4(clip.xyz / clip.w, 0.0), u_precision).xyz;
        if (u_vertex_snap > 0) {
            float g = float(u_vertex_snap);
            vec2 scr = vec2((ndc.x * 0.5 + 0.5) * u_res.x,
                            (1.0 - (ndc.y * 0.5 + 0.5)) * u_res.y);
            scr = round(scr / g) * g;
            ndc.x = (scr.x / u_res.x) * 2.0 - 1.0;
            ndc.y = (1.0 - scr.y / u_res.y) * 2.0 - 1.0;
        }
        clip.xyz = ndc * clip.w;
    }
    // Depth clamping em espaço de clip (z_clip <= w_clip <=> z_ndc <= 1.0) para z-fail
    if (u_depth_clamp != 0 && clip.w > 0.0 && clip.z > clip.w) {
        clip.z = clip.w;
    }
    gl_Position = clip;
}
"""

SV_FRAG = """
#version 330
uniform int u_depth_bits;
uniform vec2 u_inc_dec;
out vec4 out_col;
void main() {
    if (u_depth_bits > 0 && u_depth_bits < 24) {
        float levels = exp2(float(u_depth_bits)) - 1.0;
        gl_FragDepth = round(gl_FragCoord.z * levels) / levels;
    } else {
        gl_FragDepth = gl_FragCoord.z;
    }
    out_col = vec4(u_inc_dec, 0.0, 1.0);
}
"""

QUAD_VERT = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

COMP_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D u_lit;
uniform sampler2D u_amb;
uniform sampler2D u_shadow;
uniform int u_use_stencil;
uniform int u_stencil_bits;
uniform int u_stencil_wrap;
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
    vec4 lit_rgba = texture(u_lit, v_uv);
    vec3 lit = lit_rgba.rgb;
    vec3 amb = texture(u_amb, v_uv).rgb;
    int s_val = 0;
    if (u_use_stencil != 0) {
        // Avalia o registrador de stencil de N bits a partir das contagens acumuladas (n_inc, n_dec):
        // - modo wrap (OpenGL 1.4 INCR_WRAP/DECR_WRAP): aritmética modular em 2^b
        // - modo saturado (INCR/DECR clássico): satura incrementos em max_val antes de subtrair n_dec
        vec2 sv = texture(u_shadow, v_uv).rg;
        int n_inc = int(round(sv.r));
        int n_dec = int(round(sv.g));
        int max_val = (1 << u_stencil_bits) - 1;
        if (u_stencil_wrap != 0) {
            s_val = (n_inc - n_dec) & max_val;
        } else {
            int s1 = min(n_inc, max_val);
            s_val = max(0, s1 - n_dec);
        }
    }
    vec3 col = (s_val != 0) ? amb : lit;
    if (u_color_bits < 8) {
        float levels = float((1 << u_color_bits) - 1);
        if (u_dither != 0) {
            ivec2 ts = textureSize(u_lit, 0);
            ivec2 px = ivec2(int(gl_FragCoord.x), ts.y - 1 - int(gl_FragCoord.y));
            float b = bayer4(px);
            col = clamp(col + (b - 0.5) / levels, 0.0, 1.0);
        }
        col = round(col * levels) / levels;
    }
    out_col = vec4(col, lit_rgba.a);
}
"""
