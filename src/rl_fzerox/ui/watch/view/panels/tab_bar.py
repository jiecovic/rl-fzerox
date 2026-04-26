# src/rl_fzerox/ui/watch/view/panels/tab_bar.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.panels.tabs import PANEL_TABS
from rl_fzerox.ui.watch.view.panels.text import _font_line_height
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
    MouseRect,
    PygameModule,
    PygameSurface,
    ViewerFonts,
)


def _draw_panel_tabs(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    selected_index: int,
) -> tuple[int, tuple[MouseRect | None, ...]]:
    return _draw_text_tabs(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=x,
        y=y,
        width=width,
        labels=PANEL_TABS.labels,
        selected_index=selected_index,
        hint_text=_panel_tab_hint(selected_index),
    )


def _draw_text_tabs(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    labels: tuple[str, ...],
    selected_index: int,
    hint_text: str | None = None,
) -> tuple[int, tuple[MouseRect | None, ...]]:
    gap = max(2, LAYOUT.inline_value_gap // 4)
    tab_height = _font_line_height(fonts.small) + 10
    tab_y = y
    current_x = x
    baseline_y = tab_y + tab_height
    pygame.draw.line(
        screen,
        PALETTE.panel_border,
        (x, baseline_y),
        (x + width, baseline_y),
        width=1,
    )
    tab_rects: list[MouseRect | None] = []
    normalized_index = selected_index % len(labels) if labels else 0
    for index, label in enumerate(labels):
        active = index == normalized_index
        label_surface = fonts.small.render(
            label,
            True,
            PALETTE.text_primary if active else PALETTE.text_muted,
        )
        tab_width = max(32, label_surface.get_width() + 12)
        rect = pygame.Rect(current_x, tab_y, tab_width, tab_height)
        pygame.draw.rect(
            screen,
            PALETTE.panel_background if active else PALETTE.app_background,
            rect,
        )
        pygame.draw.rect(
            screen,
            PALETTE.text_accent if active else PALETTE.panel_border,
            rect,
            width=1,
        )
        if active:
            pygame.draw.line(
                screen,
                PALETTE.panel_background,
                (rect.left + 1, baseline_y),
                (rect.right - 1, baseline_y),
                width=2,
            )
        screen.blit(
            label_surface,
            (
                rect.x + max(0, (rect.width - label_surface.get_width()) // 2),
                rect.y + max(0, (rect.height - label_surface.get_height()) // 2),
            ),
        )
        tab_rects.append((rect.x, rect.y, rect.width, rect.height))
        current_x += tab_width + gap
        if current_x > x + width:
            tab_rects.extend((None,) * (len(labels) - len(tab_rects)))
            break
    if hint_text:
        hint_surface = fonts.small.render(
            hint_text,
            True,
            PALETTE.text_muted,
        )
        hint_x = x + width - hint_surface.get_width()
        if hint_x > current_x:
            screen.blit(
                hint_surface,
                (hint_x, tab_y + max(0, (tab_height - hint_surface.get_height()) // 2)),
            )
    while len(tab_rects) < len(labels):
        tab_rects.append(None)
    return tab_y + tab_height, tuple(tab_rects)


def _panel_tab_hint(selected_index: int) -> str:
    tab_number = PANEL_TABS.normalize(selected_index) + 1
    return f"Tab {tab_number}/{PANEL_TABS.count}"
