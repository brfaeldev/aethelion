"""Componentes reutilizáveis de UI imediata (botões, linhas de ciclo/passo, scroll e hitboxes)."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .theme import BTN_BG, BTN_HOVER, FG, PANEL_DIV


class UIPanelCanvas:
    """Gerencia desenho de controles clicáveis, hitboxes e rolagem vertical do painel."""

    def __init__(self):
        self.scroll: int = 0
        self.content_h: int = 0
        self.hitboxes: list[tuple[object, Callable]] = []
        self.pg = None
        self.screen = None
        self.font = None
        self.font_b = None
        self.font_s = None

    def bind(self, pg, screen, font, font_b, font_s) -> None:
        """Associa o contexto de janela e fontes do quadro atual."""
        self.pg = pg
        self.screen = screen
        self.font = font
        self.font_b = font_b
        self.font_s = font_s

    def clear_hitboxes(self) -> None:
        self.hitboxes.clear()

    def handle_click(self, mx: int, my: int) -> bool:
        for rect, cb in self.hitboxes:
            if rect.collidepoint(mx, my):
                cb()
                return True
        return False

    def handle_wheel(self, wheel_y: int, view_h: int) -> None:
        max_sc = max(0, self.content_h - view_h)
        self.scroll = int(np.clip(self.scroll - wheel_y * 28, 0, max_sc))

    def clamp_scroll(self, view_h: int) -> None:
        max_scroll = max(0, self.content_h - view_h)
        if self.scroll > max_scroll:
            self.scroll = max_scroll

    def text(
        self,
        x: int,
        y: int,
        txt: str,
        color=FG,
        font=None,
        gap: int = 2,
    ) -> int:
        f = font or self.font
        self.screen.blit(f.render(txt, True, color), (x, y))
        return y + f.get_height() + gap

    def button(
        self,
        rect,
        label: str,
        cb: Callable,
        bg=BTN_BG,
        fg=FG,
        font=None,
    ) -> None:
        pg, screen = self.pg, self.screen
        f = font or self.font_s
        mpos = pg.mouse.get_pos()
        hover = rect.collidepoint(mpos)
        pg.draw.rect(screen, BTN_HOVER if hover else bg, rect, border_radius=4)
        pg.draw.rect(screen, PANEL_DIV, rect, 1, border_radius=4)
        surf = f.render(label, True, fg)
        screen.blit(
            surf,
            (
                rect.x + (rect.w - surf.get_width()) // 2,
                rect.y + (rect.h - surf.get_height()) // 2,
            ),
        )
        self.hitboxes.append((rect, cb))

    def row_cycle(
        self,
        x: int,
        y: int,
        w: int,
        label: str,
        val_str: str,
        cb: Callable,
        color=FG,
    ) -> int:
        h = self.font.get_height() + 4
        self.screen.blit(self.font.render(f"{label:<13}:", True, FG), (x, y + 2))
        bw = max(120, w - 125)
        rect = self.pg.Rect(x + 122, y, bw, h)
        self.button(rect, str(val_str), cb, fg=color, font=self.font_s)
        return y + h + 2

    def row_step(
        self,
        x: int,
        y: int,
        w: int,
        label: str,
        val_str: str,
        cb_minus: Callable,
        cb_plus: Callable,
        color=FG,
    ) -> int:
        pg, screen, font, font_s = self.pg, self.screen, self.font, self.font_s
        h = font.get_height() + 4
        screen.blit(font.render(f"{label:<13}:", True, FG), (x, y + 2))
        bw_step = 24
        val_w = max(66, w - 125 - bw_step * 2 - 8)
        vx = x + 122
        self.button(pg.Rect(vx, y, bw_step, h), "-", cb_minus)
        lbl = font_s.render(str(val_str), True, color)
        screen.blit(
            lbl,
            (vx + bw_step + 4 + max(0, (val_w - lbl.get_width()) // 2), y + 3),
        )
        self.button(
            pg.Rect(vx + bw_step + 4 + val_w + 4, y, bw_step, h), "+", cb_plus
        )
        return y + h + 2
