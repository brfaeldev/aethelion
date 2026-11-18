"""Renderização do painel lateral de controle (HUD) e da barra de status."""
from __future__ import annotations

import time

from .theme import (
    ACC,
    ALGO_CYCLE,
    BAD,
    BITS_CYCLE,
    DEPTH_CYCLE,
    FG,
    HELP,
    OK,
    PANEL_BG,
    PANEL_DIV,
    PREC_CYCLE,
    SHADING_CYCLE,
    SNAP_CYCLE,
    STATUS_H,
    TEX_RES_CYCLE,
    VIEW_NAMES,
    WARN,
)
from .widgets import UIPanelCanvas


def render_control_panel(viewer, canvas: UIPanelCanvas, view_w: int, view_h: int, panel_w: int) -> None:
    """Desenha o painel lateral interativo e registra hitboxes clicáveis."""
    pg, s, st = viewer.pg, viewer.r.settings, viewer.r.stats
    canvas.bind(pg, viewer.screen, viewer.font, viewer.font_b, viewer.font_s)

    x0 = view_w
    pg.draw.rect(viewer.screen, PANEL_BG, (x0, 0, panel_w, view_h))
    pg.draw.line(viewer.screen, PANEL_DIV, (x0, 0), (x0, view_h), 2)

    canvas.clear_hitboxes()
    old_clip = viewer.screen.get_clip()
    viewer.screen.set_clip(pg.Rect(x0, 0, panel_w, view_h))

    x = x0 + 10
    w = panel_w - 20
    y0 = 8 - canvas.scroll
    y = y0
    gap = 2

    # ---- Controles Rápidos & Backend ---- #
    y = canvas.text(x, y, "EXECUÇÃO & CONTROLE (CLICÁVEL)", ACC, viewer.font_b, gap)
    bname = getattr(viewer.r, "backend_name", s.backend)
    y = canvas.row_cycle(
        x, y, w, "backend (TAB)", bname,
        viewer._switch_backend, OK if bname != "cpu-sw" else ACC,
    )
    y = canvas.row_cycle(x, y, w, "preset", viewer._active_preset, viewer._cycle_preset, ACC)
    y = canvas.row_cycle(
        x, y, w, "luz (M)",
        "PAUSADA" if viewer.light_paused else "animada",
        lambda: viewer._key(pg.K_m),
        WARN if viewer.light_paused else OK,
    )
    y = canvas.row_cycle(
        x, y, w, "res. adaptat.",
        "ON" if viewer.adaptive_res else "OFF",
        lambda: setattr(viewer, "adaptive_res", not viewer.adaptive_res),
        OK if viewer.adaptive_res else FG,
    )

    # ---- Parâmetros do Pipeline ---- #
    y += 4
    y = canvas.text(x, y, "CONFIGURAÇÃO DO PIPELINE", ACC, viewer.font_b, gap)
    y = canvas.row_cycle(
        x, y, w, "algoritmo", s.algorithm,
        lambda: viewer._cycle("algorithm", ALGO_CYCLE, "algoritmo"),
    )
    y = canvas.row_cycle(
        x, y, w, "stencil bits", f"{s.stencil_bits}b",
        lambda: viewer._cycle("stencil_bits", BITS_CYCLE, "stencil_bits"),
    )
    y = canvas.row_cycle(
        x, y, w, "stencil wrap",
        "INCR_WRAP" if s.stencil_wrap else "SATURADO",
        lambda: viewer._toggle("stencil_wrap", "wrap"),
        OK if s.stencil_wrap else WARN,
    )
    y = canvas.row_cycle(
        x, y, w, "z-buffer",
        "float32" if s.depth_bits == 0 else f"{s.depth_bits} bits",
        lambda: viewer._cycle("depth_bits", DEPTH_CYCLE, "depth_bits"),
    )
    y = canvas.row_cycle(
        x, y, w, "extrusão inf",
        "P_inf (w=0)" if s.infinite_extrude else "finita",
        lambda: viewer._toggle("infinite_extrude", "infinite_extrude"),
        OK if s.infinite_extrude else FG,
    )
    y = canvas.row_step(
        x, y, w, "extrusão L", f"{s.extrude_len:.1f}",
        lambda: viewer._step_num("extrude_len", -5.0, 2.0, 120.0, "extrude_len"),
        lambda: viewer._step_num("extrude_len", +5.0, 2.0, 120.0, "extrude_len"),
    )
    y = canvas.row_cycle(
        x, y, w, "depth clamp",
        "ON" if s.depth_clamp else "OFF",
        lambda: viewer._toggle("depth_clamp", "depth_clamp"),
    )
    y = canvas.row_cycle(
        x, y, w, "precisão", s.precision,
        lambda: viewer._cycle("precision", PREC_CYCLE, "precisao"),
    )
    y = canvas.row_step(
        x, y, w, "res. interna", f"{s.width}x{s.height}",
        lambda: viewer._step_resolution(-1),
        lambda: viewer._step_resolution(+1),
    )

    # ---- Filtros, Textura & Câmera ---- #
    y += 4
    y = canvas.text(x, y, "FILTROS, TEXTURA & CÂMERA", ACC, viewer.font_b, gap)
    y = canvas.row_cycle(
        x, y, w, "shading", s.shading,
        lambda: viewer._cycle("shading", SHADING_CYCLE, "shading"),
    )
    y = canvas.row_cycle(
        x, y, w, "texturas", "ON" if s.textured else "OFF",
        lambda: viewer._toggle("textured", "textured"),
    )
    y = canvas.row_cycle(
        x, y, w, "filtro tex", s.texture_filter,
        lambda: viewer._cycle("texture_filter", ["nearest", "bilinear"], "texture_filter"),
    )
    y = canvas.row_cycle(
        x, y, w, "res. textura", f"{s.texture_res}px",
        lambda: viewer._cycle("texture_res", TEX_RES_CYCLE, "texture_res"),
    )
    y = canvas.row_cycle(
        x, y, w, "cor / dither",
        f"{s.color_bits}b / {'ON' if s.dither else 'OFF'}",
        lambda: viewer._cycle("color_bits", BITS_CYCLE, "color_bits"),
    )
    y = canvas.row_cycle(
        x, y, w, "dither Bayer", "ON" if s.dither else "OFF",
        lambda: viewer._toggle("dither", "dither"),
    )
    y = canvas.row_cycle(
        x, y, w, "vertex snap",
        "OFF" if s.vertex_snap == 0 else f"1/{s.vertex_snap}px",
        lambda: viewer._cycle("vertex_snap", SNAP_CYCLE, "vertex_snap"),
    )
    y = canvas.row_cycle(
        x, y, w, "filtro janela", s.display_filter,
        lambda: viewer._cycle("display_filter", ["nearest", "bilinear"], "display_filter"),
    )
    y = canvas.row_step(
        x, y, w, "ambiente", f"{s.ambient:.2f}",
        lambda: viewer._step_num("ambient", -0.05, 0.0, 1.0, "ambient"),
        lambda: viewer._step_num("ambient", +0.05, 0.0, 1.0, "ambient"),
    )
    y = canvas.row_step(
        x, y, w, "triângulos", f"{s.detail:.2f}x ({st.tris_scene})",
        lambda: viewer._step_num("detail", -0.25, 0.25, 4.0, "detail", rebuild=True),
        lambda: viewer._step_num("detail", +0.25, 0.25, 4.0, "detail", rebuild=True),
    )
    y = canvas.row_step(
        x, y, w, "fov (°)", f"{s.fov:.0f}",
        lambda: viewer._step_num("fov", -5.0, 30.0, 110.0, "fov"),
        lambda: viewer._step_num("fov", +5.0, 30.0, 110.0, "fov"),
    )
    y = canvas.row_step(
        x, y, w, "near", f"{s.near:.2f}",
        lambda: viewer._step_num("near", -0.05, 0.05, 2.0, "near"),
        lambda: viewer._step_num("near", +0.05, 0.05, 2.0, "near"),
    )
    y = canvas.row_step(
        x, y, w, "far", f"{s.far:.0f}",
        lambda: viewer._step_num("far", -5.0, 10.0, 200.0, "far"),
        lambda: viewer._step_num("far", +5.0, 10.0, 200.0, "far"),
    )
    y = canvas.row_cycle(
        x, y, w, "vista (V)", VIEW_NAMES[viewer.view_mode],
        lambda: viewer._key(pg.K_v), ACC,
    )

    # ---- Botões de Ação ---- #
    y += 4
    bh = 24
    bw2 = (w - 6) // 2
    canvas.button(
        pg.Rect(x, y, bw2, bh),
        "Parar Exp" if viewer.run.active else "Iniciar Exp",
        lambda: viewer._key(pg.K_SPACE),
        bg=(45, 75, 55) if not viewer.run.active else (95, 50, 50),
    )
    canvas.button(pg.Rect(x + bw2 + 6, y, bw2, bh), "Medir RayCast (G)", viewer._measure_now)
    y += bh + 4
    canvas.button(pg.Rect(x, y, bw2, bh), "Conformidade (C)", lambda: viewer._key(pg.K_c))
    canvas.button(pg.Rect(x + bw2 + 6, y, bw2, bh), "Salvar Config", viewer._save_config)
    y += bh + 6

    # ---- Telemetria ---- #
    y = canvas.text(x, y, "DESEMPENHO & MEMÓRIA", ACC, viewer.font_b, gap)
    y = canvas.text(x, y, f"quadro   : {st.ms_total:6.1f} ms ({st.fps:5.2f} fps)", FG, viewer.font, gap)
    y = canvas.text(
        x, y,
        f"  geom/sil/sten: {st.ms_geometry:.1f} / {st.ms_silhouette:.1f} / {st.ms_stencil:.1f} ms",
        FG, viewer.font_s, gap,
    )
    y = canvas.text(
        x, y,
        f"memória  : {st.mem_total_kb:6.1f} KB (buf {st.mem_buffers_kb:.0f} KB)",
        FG, viewer.font, gap,
    )
    y = canvas.text(
        x, y,
        f"tris cena/vol : {st.tris_scene_drawn}/{st.tris_scene} | {st.tris_volume_drawn}/{st.tris_volume}",
        FG, viewer.font_s, gap,
    )
    y = canvas.text(
        x, y,
        f"frag st / px  : {st.frag_stencil} | {st.pixels_shadow}/{st.pixels_visible}",
        FG, viewer.font_s, gap,
    )

    col_sat = BAD if st.stencil_saturation else FG
    y = canvas.text(
        x, y,
        f"sat/wrap/cam  : {st.stencil_saturation} / {st.stencil_wrap_events} / {st.volumes_with_camera}",
        col_sat, viewer.font_s, gap,
    )

    ag = viewer.run.last_agreement
    if ag is None:
        y = canvas.text(x, y, "concordância  : sem medição (G)", WARN, viewer.font_s, gap)
    else:
        c = OK if ag.f1 > 0.98 else (WARN if ag.f1 > 0.90 else BAD)
        y = canvas.text(
            x, y,
            f"F1={ag.f1:.4f} (P={ag.precision:.4f} R={ag.recall:.4f})",
            c, viewer.font_s, gap,
        )

    if viewer.conf_result:
        p, t = viewer.conf_result["passed"], viewer.conf_result["total"]
        y = canvas.text(
            x, y, f"conformidade  : {p}/{t} PASS",
            OK if p == t else BAD, viewer.font_b, gap,
        )

    y += 4
    y = canvas.text(x, y, "ATALHOS", ACC, viewer.font_b, gap)
    for k, d in HELP:
        y = canvas.text(x, y, f"{k:<11} {d}", (165, 165, 180), viewer.font_s, 1)

    canvas.content_h = y - y0 + 10
    canvas.clamp_scroll(view_h)
    viewer.screen.set_clip(old_clip)


def render_status_bar(viewer, view_w: int, view_h: int, panel_w: int) -> None:
    """Desenha a barra de estado inferior da janela."""
    pg = viewer.pg
    total_w = view_w + panel_w
    pg.draw.rect(viewer.screen, (14, 14, 18), (0, view_h, total_w, STATUS_H))
    pg.draw.line(viewer.screen, PANEL_DIV, (0, view_h), (total_w, view_h), 1)
    if viewer._status_msg and time.perf_counter() < viewer._status_until:
        txt, color = viewer._status_msg, ACC
    elif viewer.run.active:
        p = viewer.run.phase
        left = max(0.0, viewer.run.total_duration - viewer.run.elapsed_effective)
        txt = (
            f"EXPERIMENTO  fase {p.key} [{p.hipotese}] {p.titulo}  |  "
            f"quadro {viewer.run.frame}  |  restam {left:5.1f}s  |  "
            f"backend={getattr(viewer.r, 'backend_name', 'cpu-sw')}"
        )
        color = OK
    elif viewer.run.finished:
        txt = (
            f"CONCLUÍDO — {viewer.run.frame} quadros gravados em "
            f"'{viewer.outdir}'. ESPAÇO reinicia."
        )
        color = ACC
    elif viewer.view_mode == 3 and viewer.run.last_diff is None:
        txt = "VISTA DE DIFERENÇA — Pressione G para calcular a diferença contra ray casting"
        color = WARN
    else:
        bname = getattr(viewer.r, "backend_name", "cpu-sw")
        modo_tela = "TELA CHEIA" if viewer.fullscreen else "F11: Tela Cheia"
        txt = (
            f"PRONTO [{bname}] — TAB: Backend | M: Pausar Luz | Arraste: Câmera | "
            f"ESPAÇO: Experimento | {modo_tela}"
        )
        color = FG
    viewer.screen.blit(
        viewer.font.render(txt, True, color),
        (12, view_h + (STATUS_H - viewer.font.get_height()) // 2),
    )
