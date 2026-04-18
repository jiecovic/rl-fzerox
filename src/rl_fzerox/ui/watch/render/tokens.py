# src/rl_fzerox/ui/watch/render/tokens.py
from __future__ import annotations

from rl_fzerox.ui.watch.layout import LAYOUT, PALETTE, FlagViz, ViewerFonts


def _draw_flag_viz(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    flag_viz: FlagViz,
) -> int:
    label_surface = fonts.small.render("Flags", True, PALETTE.text_muted)
    screen.blit(label_surface, (x, y))
    y += label_surface.get_height() + LAYOUT.line_gap

    pill_height = 0
    for row in flag_viz.rows:
        row_x = x
        for token in row:
            pill_rect = _draw_pill(
                pygame=pygame,
                screen=screen,
                font=fonts.small,
                x=row_x,
                y=y,
                label=token.label,
                active=token.active,
                active_text_color=PALETTE.text_primary,
                active_fill_color=PALETTE.flag_active_background,
                active_border_color=PALETTE.flag_active_border,
            )
            pill_height = pill_rect.height
            row_x += pill_rect.width + LAYOUT.flag_token_gap
        y += pill_height + LAYOUT.line_gap

    return y


def _pill_width(font, label: str) -> int:
    return font.render(label, True, PALETTE.text_primary).get_width() + (
        2 * LAYOUT.flag_token_pad_x
    )


def _pill_height(font) -> int:
    return font.render("Ag", True, PALETTE.text_primary).get_height() + (
        2 * LAYOUT.flag_token_pad_y
    )


def _draw_pill(
    *,
    pygame,
    screen,
    font,
    x: int,
    y: int,
    label: str,
    active: bool,
    active_text_color,
    active_fill_color,
    active_border_color,
):
    text_color = active_text_color if active else PALETTE.text_muted
    fill_color = active_fill_color if active else PALETTE.flag_inactive_background
    border_color = active_border_color if active else PALETTE.flag_inactive_border
    text_surface = font.render(label, True, text_color)
    pill_rect = pygame.Rect(
        x,
        y,
        text_surface.get_width() + (2 * LAYOUT.flag_token_pad_x),
        text_surface.get_height() + (2 * LAYOUT.flag_token_pad_y),
    )
    pygame.draw.rect(
        screen,
        fill_color,
        pill_rect,
        border_radius=pill_rect.height // 2,
    )
    pygame.draw.rect(
        screen,
        border_color,
        pill_rect,
        width=1,
        border_radius=pill_rect.height // 2,
    )
    screen.blit(
        text_surface,
        (x + LAYOUT.flag_token_pad_x, y + LAYOUT.flag_token_pad_y),
    )
    return pill_rect
