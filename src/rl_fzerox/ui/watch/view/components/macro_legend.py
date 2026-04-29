# src/rl_fzerox/ui/watch/view/components/macro_legend.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import (
    PygameModule,
    PygameRect,
    PygameSurface,
    RenderFont,
    ViewerFonts,
)


@dataclass(frozen=True)
class MacroLegendHint:
    """One keyboard shortcut shown in the watch HUD legend."""

    keys: str
    action: str
    controller: str | None = None


@dataclass(frozen=True)
class MacroLegendGroup:
    """One logical group of HUD keyboard hints."""

    title: str
    hints: tuple[MacroLegendHint, ...]


@dataclass(frozen=True)
class _MacroLegendStyle:
    padding_x: int = 10
    padding_y: int = 8
    row_gap: int = 5
    hint_gap: int = 6
    key_gap: int = 4
    key_pad_x: int = 6
    key_pad_y: int = 2
    action_pad_right: int = 2
    group_gap: int = 7
    radius: int = 8
    fill: Color = (12, 17, 23)
    border: Color = (42, 54, 66)
    key_fill: Color = (23, 31, 39)
    key_border: Color = (80, 99, 116)
    title: Color = PALETTE.text_muted
    key_text: Color = PALETTE.text_primary
    controller_text: Color = PALETTE.text_accent
    action_text: Color = PALETTE.text_muted


VIEWER_HOTKEY_HINTS: tuple[MacroLegendHint, ...] = (
    MacroLegendHint("Esc", "close"),
    MacroLegendHint("P", "pause"),
    MacroLegendHint("N", "step"),
    MacroLegendHint("R", "reset"),
    MacroLegendHint("K", "save"),
    MacroLegendHint("M", "manual"),
    MacroLegendHint("D", "policy"),
    MacroLegendHint("Tab / 1-6", "tabs"),
    MacroLegendHint("+/-", "speed"),
    MacroLegendHint("0", "realtime"),
)
MANUAL_CONTROL_HINTS: tuple[MacroLegendHint, ...] = (
    MacroLegendHint("Arrow keys", "steer/pitch", controller="stick X/Y"),
    MacroLegendHint("Z", "accelerate", controller="A"),
    MacroLegendHint("X", "air brake", controller="C-down"),
    MacroLegendHint("Space", "boost", controller="B"),
    MacroLegendHint("A", "lean left", controller="Z"),
    MacroLegendHint("S", "lean right", controller="R"),
    MacroLegendHint("Enter", "start", controller="Start"),
)
_LEGEND_SEPARATOR = "›"
MACRO_LEGEND_GROUPS: tuple[MacroLegendGroup, ...] = (
    MacroLegendGroup("General", VIEWER_HOTKEY_HINTS),
    MacroLegendGroup("Manual mode", MANUAL_CONTROL_HINTS),
)
MACRO_LEGEND_HINTS: tuple[MacroLegendHint, ...] = (
    *VIEWER_HOTKEY_HINTS,
    *MANUAL_CONTROL_HINTS,
)
_MACRO_LEGEND_STYLE = _MacroLegendStyle()


def _macro_legend_height(*, fonts: ViewerFonts, width: int) -> int:
    if width <= 0:
        return 0

    style = _MACRO_LEGEND_STYLE
    content_width = width - (2 * style.padding_x)
    if content_width <= 0:
        return 0

    group_heights = tuple(
        _macro_legend_group_height(font=fonts.small, group=group, width=content_width)
        for group in MACRO_LEGEND_GROUPS
    )
    visible_group_heights = tuple(height for height in group_heights if height > 0)
    if not visible_group_heights:
        return 0

    return (2 * style.padding_y) + sum(visible_group_heights) + (
        (len(visible_group_heights) - 1) * style.group_gap
    )


def _draw_macro_legend(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
) -> int:
    height = _macro_legend_height(fonts=fonts, width=width)
    if height <= 0:
        return y

    style = _MACRO_LEGEND_STYLE
    rect = pygame.Rect(x, y, width, height)
    _draw_macro_legend_panel(pygame=pygame, screen=screen, rect=rect)

    content_x = x + style.padding_x
    content_y = y + style.padding_y
    content_width = width - (2 * style.padding_x)
    row_y = content_y
    for group in MACRO_LEGEND_GROUPS:
        group_height = _macro_legend_group_height(
            font=fonts.small,
            group=group,
            width=content_width,
        )
        if group_height <= 0:
            continue
        title_surface = fonts.small.render(group.title, True, style.title)
        screen.blit(title_surface, (content_x, row_y))
        row_y += title_surface.get_height() + LAYOUT.line_gap
        for row in _macro_legend_rows(font=fonts.small, width=content_width, hints=group.hints):
            row_x = content_x
            for hint in row:
                hint_rect = _draw_macro_hint(
                    pygame=pygame,
                    screen=screen,
                    font=fonts.small,
                    x=row_x,
                    y=row_y,
                    hint=hint,
                )
                row_x = hint_rect.right + style.hint_gap
            row_y += _macro_hint_height(font=fonts.small) + style.row_gap
        row_y += style.group_gap - style.row_gap

    return rect.bottom


def _macro_legend_group_height(
    *,
    font: RenderFont,
    group: MacroLegendGroup,
    width: int,
) -> int:
    rows = _macro_legend_rows(font=font, width=width, hints=group.hints)
    if not rows:
        return 0

    title_height = font.render(group.title, True, _MACRO_LEGEND_STYLE.title).get_height()
    row_height = _macro_hint_height(font=font)
    return (
        title_height
        + LAYOUT.line_gap
        + (len(rows) * row_height)
        + ((len(rows) - 1) * _MACRO_LEGEND_STYLE.row_gap)
    )


def _macro_legend_rows(
    *,
    font: RenderFont,
    width: int,
    hints: tuple[MacroLegendHint, ...] = MACRO_LEGEND_HINTS,
) -> tuple[tuple[MacroLegendHint, ...], ...]:
    style = _MACRO_LEGEND_STYLE
    rows: list[list[MacroLegendHint]] = []
    current_row: list[MacroLegendHint] = []
    current_width = 0
    for hint in hints:
        hint_width = _macro_hint_width(font=font, hint=hint)
        next_width = hint_width if not current_row else current_width + style.hint_gap + hint_width
        if current_row and next_width > width:
            rows.append(current_row)
            current_row = [hint]
            current_width = hint_width
        else:
            current_row.append(hint)
            current_width = next_width

    if current_row:
        rows.append(current_row)
    return tuple(tuple(row) for row in rows)


def _draw_macro_legend_panel(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rect: PygameRect,
) -> None:
    style = _MACRO_LEGEND_STYLE
    pygame.draw.rect(screen, (4, 6, 9), rect.move(0, 2), border_radius=style.radius)
    pygame.draw.rect(screen, style.fill, rect, border_radius=style.radius)
    pygame.draw.rect(screen, style.border, rect, width=1, border_radius=style.radius)


def _draw_macro_hint(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    font: RenderFont,
    x: int,
    y: int,
    hint: MacroLegendHint,
) -> PygameRect:
    style = _MACRO_LEGEND_STYLE
    key_surface = font.render(hint.keys, True, style.key_text)
    action_surface = font.render(_macro_hint_action_text(hint), True, style.action_text)
    key_rect = pygame.Rect(
        x,
        y,
        key_surface.get_width() + (2 * style.key_pad_x),
        key_surface.get_height() + (2 * style.key_pad_y),
    )
    pygame.draw.rect(screen, style.key_fill, key_rect, border_radius=5)
    pygame.draw.rect(screen, style.key_border, key_rect, width=1, border_radius=5)
    screen.blit(key_surface, (key_rect.left + style.key_pad_x, key_rect.top + style.key_pad_y))

    action_x = key_rect.right + style.key_gap
    action_y = key_rect.centery - (action_surface.get_height() // 2)
    if hint.controller is not None:
        controller_surface = font.render(hint.controller, True, style.controller_text)
        arrow_surface = font.render(_LEGEND_SEPARATOR, True, style.action_text)
        screen.blit(controller_surface, (action_x, action_y))
        arrow_x = action_x + controller_surface.get_width() + style.key_gap
        screen.blit(arrow_surface, (arrow_x, action_y))
        action_x = arrow_x + arrow_surface.get_width() + style.key_gap
    screen.blit(action_surface, (action_x, action_y))
    return pygame.Rect(
        x,
        y,
        _macro_hint_width(font=font, hint=hint),
        key_rect.height,
    )


def _macro_hint_action_text(hint: MacroLegendHint) -> str:
    return hint.action


def _macro_hint_width(*, font: RenderFont, hint: MacroLegendHint) -> int:
    style = _MACRO_LEGEND_STYLE
    key_width = font.render(hint.keys, True, style.key_text).get_width() + (2 * style.key_pad_x)
    action_width = font.render(_macro_hint_action_text(hint), True, style.action_text).get_width()
    if hint.controller is not None:
        controller_width = font.render(hint.controller, True, style.controller_text).get_width()
        arrow_width = font.render(_LEGEND_SEPARATOR, True, style.action_text).get_width()
        action_width += controller_width + arrow_width + (2 * style.key_gap)
    return key_width + style.key_gap + action_width + style.action_pad_right


def _macro_hint_height(*, font: RenderFont) -> int:
    style = _MACRO_LEGEND_STYLE
    return font.render("Ag", True, style.key_text).get_height() + (2 * style.key_pad_y)
