"""Shaders GLSL para o backend OpenGL/ModernGL (`gpu-gl`)."""
from __future__ import annotations

GEOM_VERT = """
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

GEOM_FRAG = """
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

SV_VERT = """
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

SV_FRAG = """
#version 330
uniform int u_depth_bits;
uniform vec2 u_inc_dec;
out vec4 out_col;
void main() {
    if (u_depth_bits > 0) {
        float levels = float((1 << u_depth_bits) - 1);
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
    vec3 lit = texture(u_lit, v_uv).rgb;
    vec3 amb = texture(u_amb, v_uv).rgb;
    int s_val = 0;
    if (u_use_stencil != 0) {
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
            float b = bayer4(ivec2(gl_FragCoord.xy));
            col = clamp(col + (b - 0.5) / levels, 0.0, 1.0);
        }
        col = round(col * levels) / levels;
    }
    out_col = vec4(col, 1.0);
}
"""
