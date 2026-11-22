"""
Sobreposição Visual 3D de Volumes de Sombra e Silhuetas (overlay.py)

Renderiza uma vista didática semitransparente sobre o framebuffer da cena,
destacando faces frontais (âmbar) e traseiras (azul-ciano) dos volumes de
sombra extrudados, arestas de silhueta em ciano brilhante e a posição da
fonte de luz pontual com teste de profundidade contra o Z-buffer.
"""
from __future__ import annotations

import numpy as np

from . import shadow as sv
from .precision import apply_precision
from .raster import scan_triangle

F = np.float32


def draw_line_2d(
    buf: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple,
    alpha: float = 0.8,
    z0: float | None = None,
    z1: float | None = None,
    depth_buf: np.ndarray | None = None,
) -> None:
    """Rasteriza um segmento de reta 2D com blend alfa e teste opcional de profundidade."""
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


def render_volume_overlay(renderer, base_img: np.ndarray | None = None) -> np.ndarray:
    """Gera vista didática 3D dos Volumes de Sombra e Arestas de Silhueta.

    Inspirada nas Figuras 2, 3 e 5 de Everitt & Kilgard (2002):
    - Faces frontais visíveis dos volumes de sombra em âmbar semitransparente;
    - Faces traseiras visíveis dos volumes em azul-ciano suave;
    - Arestas laterais extrudadas em amarelo e arestas de silhueta em ciano brilhante.
    """
    if base_img is None:
        if hasattr(renderer, "_resolve") and hasattr(renderer, "col_amb") and hasattr(renderer, "col_lit"):
            base_img = renderer._resolve(
                np.where(renderer.shadow_mask[..., None], renderer.col_amb, renderer.col_lit)
            )
        else:
            base_img = np.zeros((renderer.settings.height, renderer.settings.width, 3), dtype=np.uint8)
    out = base_img.astype(F).copy()
    s = renderer.settings
    W, H = s.width, s.height
    vp = renderer._last_vp

    vol_sides = []
    sil_segments = []
    use_inf = bool(s.infinite_extrude or s.extrude_len <= 0.0)
    vis_len = s.extrude_len if s.extrude_len > 0 else 40.0
    for prop in renderer.scene:
        if not prop.occluder:
            continue
        pw, _ = renderer._world(prop)
        sides_h, _, _ = sv.build_volume_homogeneous(
            pw, prop.solid.indices, prop.solid.dir_edges,
            prop.solid.edge_face, prop.solid.edge_opp,
            renderer.light, length=vis_len, infinite=use_inf, caps=False,
        )
        if sides_h.shape[0]:
            vol_sides.append(sides_h)
        seg = sv.extract_silhouette_segments(
            pw, prop.solid.indices, prop.solid.dir_edges,
            prop.solid.edge_face, prop.solid.edge_opp, renderer.light,
        )
        if seg.shape[0]:
            sil_segments.append(seg)

    # 1. Preenchimento semitransparente das faces laterais dos volumes
    if vol_sides:
        sides = np.concatenate(vol_sides, axis=0)
        clip = renderer._project(sides, vp)
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
            if s.vertex_snap > 0:
                g = F(s.vertex_snap)
                sx_b = np.round(sx_b / g) * g
                sy_b = np.round(sy_b / g) * g
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
            for cp, _ in renderer._clip_parts(clip[m], []):
                scr = renderer._screen(cp)
                if scr is not None:
                    ov_tris.append((scr[0], scr[1], scr[2]))

        for sx, sy, z in ov_tris:
            res = scan_triangle(sx, sy, W, H)
            if res is None:
                continue
            yy, xx, l0, l1, l2, area = res
            zz = l0 * z[0] + l1 * z[1] + l2 * z[2]
            vis = (zz >= -1.0) & (zz <= 1.0) & (zz < renderer.depth[yy, xx])
            if not vis.any():
                continue
            yy, xx = yy[vis], xx[vis]
            if area < 0.0:
                out[yy, xx] = out[yy, xx] * F(0.84) + front_tint * F(0.16)
            else:
                out[yy, xx] = out[yy, xx] * F(0.90) + back_tint * F(0.10)
            for i in range(3):
                j = (i + 1) % 3
                draw_line_2d(
                    out, sx[i], sy[i], sx[j], sy[j],
                    (235, 185, 45), alpha=0.45,
                    z0=float(z[i]), z1=float(z[j]),
                    depth_buf=renderer.depth,
                )

    # 2. Arestas de silhueta em destaque (ciano brilhante) com recorte near e teste de profundidade
    if sil_segments:
        segs = np.concatenate(sil_segments, axis=0)
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
            draw_line_2d(
                out, x0, y0, x1, y1, (0, 255, 220), alpha=0.95,
                z0=float(ndc0[2]), z1=float(ndc1[2]),
                depth_buf=renderer.depth,
            )

    # 3. Marcador da posição da luz pontual
    lh = np.array([renderer.light[0], renderer.light[1], renderer.light[2], 1.0], dtype=F)
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
