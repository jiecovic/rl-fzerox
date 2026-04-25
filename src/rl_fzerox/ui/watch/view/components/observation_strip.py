# src/rl_fzerox/ui/watch/view/components/observation_strip.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.components.cockpit import _draw_control_viz
from rl_fzerox.ui.watch.view.components.game_view import _glass_overlay_surface
from rl_fzerox.ui.watch.view.panels.format import (
    _format_observation_summary,
    _observation_minimap_layer,
    _observation_preview_grid,
    _observation_stack_size,
)
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import (
    ControlViz,
    PygameModule,
    PygameRect,
    PygameSurface,
    RenderFont,
    ViewerFonts,
    ViewerHitboxes,
)


@dataclass(frozen=True)
class _ObservationGlassStyle:
    shadow: Color = (2, 3, 4)
    fill: Color = (7, 10, 12)
    edge: Color = (40, 50, 62)
    highlight: Color = (86, 101, 116)
    preview_border: Color = PALETTE.text_warning
    preview_border_width: int = 2


_OBSERVATION_GLASS_STYLE = _ObservationGlassStyle()


def _draw_control_viz_below_game(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    game_display_size: tuple[int, int],
    control_viz: ControlViz,
) -> tuple[ViewerHitboxes, int]:
    x = LAYOUT.preview_padding
    y = game_display_size[1] + LAYOUT.preview_gap
    width = game_display_size[0] - (2 * LAYOUT.preview_padding)
    if width <= 0 or y >= screen.get_height():
        return ViewerHitboxes(), y

    bottom, deterministic_toggle_rect = _draw_control_viz(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=x,
        y=y,
        width=width,
        control_viz=control_viz,
    )
    return ViewerHitboxes(deterministic_toggle=deterministic_toggle_rect), bottom


def _draw_observation_preview_in_rect(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    surface: PygameSurface,
    x: int,
    y: int,
    width: int,
    height: int,
    observation_shape: tuple[int, ...],
    info: dict[str, object],
) -> None:
    if width <= 0 or height <= 0:
        return

    bottom = y + height
    title_surface = fonts.section.render("Policy Obs", True, PALETTE.text_primary)
    subtitle_surface = fonts.small.render(
        _format_observation_summary(observation_shape, info=info),
        True,
        PALETTE.text_muted,
    )
    screen.blit(title_surface, (x, y))
    y += title_surface.get_height() + LAYOUT.preview_title_gap
    screen.blit(subtitle_surface, (x, y))
    y += subtitle_surface.get_height() + LAYOUT.section_rule_gap

    preview_width, preview_height = surface.get_size()
    glass_padding = 8
    labels = _observation_tile_label_texts(observation_shape, info=info)
    label_gap = 5 if labels else 0
    label_row_height = _observation_label_row_height(font=fonts.body, labels=labels)
    max_preview_width = width - (2 * glass_padding)
    max_preview_height = bottom - y - (2 * glass_padding) - label_gap - label_row_height
    if max_preview_width <= 0 or max_preview_height <= 0:
        return

    scale = min(1.0, max_preview_width / preview_width, max_preview_height / preview_height)
    if scale <= 0:
        return

    scaled_size = (
        max(1, round(preview_width * scale)),
        max(1, round(preview_height * scale)),
    )
    glass_size = (
        scaled_size[0] + (2 * glass_padding),
        scaled_size[1] + label_gap + label_row_height + (2 * glass_padding),
    )
    glass_x = x + max(0, (width - glass_size[0]) // 2)
    glass_rect = pygame.Rect(glass_x, y, *glass_size)
    _draw_observation_glass_box(pygame=pygame, screen=screen, rect=glass_rect)

    preview_y = glass_rect.top + glass_padding
    preview_x = glass_rect.left + glass_padding
    preview_rect = pygame.Rect(preview_x, preview_y, *scaled_size)
    pygame.draw.rect(screen, PALETTE.panel_background, preview_rect)
    preview_surface = (
        surface
        if surface.get_size() == scaled_size
        else pygame.transform.scale(surface, scaled_size)
    )
    screen.blit(preview_surface, preview_rect.topleft)
    _draw_observation_tile_borders(
        pygame=pygame,
        screen=screen,
        rect=preview_rect,
        observation_shape=observation_shape,
        info=info,
    )
    _draw_outer_preview_border(pygame=pygame, screen=screen, rect=preview_rect)
    screen.blit(
        _glass_overlay_surface(pygame, glass_rect.size, 10),
        glass_rect.topleft,
    )
    _draw_observation_tile_labels(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        rect=preview_rect,
        y=preview_rect.bottom + label_gap,
        height=label_row_height,
        observation_shape=observation_shape,
        info=info,
    )


def _draw_outer_preview_border(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rect: PygameRect,
) -> None:
    style = _OBSERVATION_GLASS_STYLE
    border_rect = rect.inflate(style.preview_border_width * 2, style.preview_border_width * 2)
    pygame.draw.rect(
        screen,
        style.preview_border,
        border_rect,
        width=style.preview_border_width,
        border_radius=4,
    )


def _draw_observation_glass_box(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rect: PygameRect,
) -> None:
    style = _OBSERVATION_GLASS_STYLE
    pygame.draw.rect(screen, style.shadow, rect.move(0, 3), border_radius=13)
    pygame.draw.rect(screen, style.fill, rect, border_radius=13)
    pygame.draw.rect(screen, style.highlight, rect, width=1, border_radius=13)
    pygame.draw.rect(
        screen,
        style.edge,
        rect.inflate(-4, -4),
        width=1,
        border_radius=10,
    )


def _draw_observation_tile_labels(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    rect: PygameRect,
    y: int,
    height: int,
    observation_shape: tuple[int, ...],
    info: dict[str, object],
) -> None:
    labels = _observation_tile_label_texts(observation_shape, info=info)
    if not labels or height <= 0:
        return

    for index, label in enumerate(labels):
        tile_rect = _observation_tile_rect(pygame=pygame, rect=rect, index=index, count=len(labels))
        is_newest = index == 0
        is_minimap = label == "map"
        color = _observation_tile_label_color(is_newest=is_newest, is_minimap=is_minimap)
        _draw_observation_tile_label(
            pygame=pygame,
            screen=screen,
            font=fonts.body,
            label=label,
            center_x=tile_rect.left + (tile_rect.size[0] // 2),
            center_y=y + (height // 2),
            color=color,
            active=is_newest,
        )


def _observation_tile_time_label(*, index: int, stack_size: int) -> str:
    frame_delta = (stack_size - 1) - index
    return f"Δ{frame_delta}"


def _observation_tile_label_color(*, is_newest: bool, is_minimap: bool) -> Color:
    if is_newest:
        return PALETTE.text_accent
    if is_minimap:
        return PALETTE.text_muted
    return PALETTE.text_primary


def _draw_observation_tile_borders(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rect: PygameRect,
    observation_shape: tuple[int, ...],
    info: dict[str, object],
) -> None:
    frame_count = _observation_tile_count(observation_shape, info=info)
    if frame_count <= 1:
        return

    for index in range(frame_count):
        tile_rect = _observation_tile_rect(pygame=pygame, rect=rect, index=index, count=frame_count)
        is_newest = index == 0
        border_color = PALETTE.text_accent if is_newest else PALETTE.panel_border
        pygame.draw.rect(screen, border_color, tile_rect, width=2 if is_newest else 1)


def _observation_tile_count(
    observation_shape: tuple[int, ...],
    *,
    info: dict[str, object],
) -> int:
    stack_size = _observation_stack_size(observation_shape, info=info)
    return stack_size + (1 if _observation_minimap_layer(info) else 0)


def _observation_tile_label_texts(
    observation_shape: tuple[int, ...],
    *,
    info: dict[str, object],
) -> tuple[str, ...]:
    stack_size = _observation_stack_size(observation_shape, info=info)
    frame_count = _observation_tile_count(observation_shape, info=info)
    if frame_count <= 1:
        return ()
    return tuple(
        "map"
        if index >= stack_size
        else _observation_tile_time_label(index=(stack_size - 1) - index, stack_size=stack_size)
        for index in range(frame_count)
    )


def _observation_label_row_height(
    *,
    font: RenderFont,
    labels: tuple[str, ...],
) -> int:
    if not labels:
        return 0

    label_padding_y = 3
    return max(font.render(label, True, PALETTE.text_primary).get_height() for label in labels) + (
        2 * label_padding_y
    )


def _observation_tile_rect(
    *,
    pygame: PygameModule,
    rect: PygameRect,
    index: int,
    count: int,
) -> PygameRect:
    columns, rows = _observation_preview_grid(count)
    row, column = divmod(index, columns)
    tile_left = rect.left + round((column * rect.size[0]) / columns)
    tile_right = rect.left + round(((column + 1) * rect.size[0]) / columns)
    tile_top = rect.top + round((row * rect.size[1]) / rows)
    tile_bottom = rect.top + round(((row + 1) * rect.size[1]) / rows)
    return pygame.Rect(
        tile_left,
        tile_top,
        tile_right - tile_left,
        tile_bottom - tile_top,
    )


def _draw_observation_tile_label(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    font: RenderFont,
    label: str,
    center_x: int,
    center_y: int,
    color: Color,
    active: bool,
) -> None:
    text_surface = font.render(label, True, color)
    pad_x = 7
    pad_y = 3
    label_rect = pygame.Rect(
        center_x - ((text_surface.get_width() + (2 * pad_x)) // 2),
        center_y - ((text_surface.get_height() + (2 * pad_y)) // 2),
        text_surface.get_width() + (2 * pad_x),
        text_surface.get_height() + (2 * pad_y),
    )
    border_color = color if active else PALETTE.panel_border
    pygame.draw.rect(screen, PALETTE.panel_background, label_rect, border_radius=5)
    pygame.draw.rect(screen, border_color, label_rect, width=1, border_radius=5)
    text_x = label_rect.left + pad_x
    text_y = label_rect.top + pad_y
    outline_surface = font.render(label, True, (0, 0, 0))
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        screen.blit(outline_surface, (text_x + dx, text_y + dy))
    screen.blit(text_surface, (text_x, text_y))
