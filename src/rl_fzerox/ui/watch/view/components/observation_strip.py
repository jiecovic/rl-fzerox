# src/rl_fzerox/ui/watch/view/components/observation_strip.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rl_fzerox.ui.watch.view.components.cockpit import (
    _control_viz_panel_height,
    _draw_control_viz,
)
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


class _RectLike(Protocol):
    left: int
    top: int
    bottom: int
    size: tuple[int, int]
    topleft: tuple[int, int]

    def move(self, x: int, y: int) -> _RectLike: ...

    def inflate(self, x: int, y: int) -> _RectLike: ...


class _ScreenLike(Protocol):
    def get_height(self) -> int: ...

    def blit(self, source: object, dest: tuple[int, int]) -> object: ...


class _SurfaceLike(Protocol):
    def get_size(self) -> tuple[int, int]: ...


class _PygameDrawLike(Protocol):
    def rect(
        self,
        surface: _ScreenLike,
        color: Color,
        rect: _RectLike,
        width: int = 0,
        border_radius: int = 0,
    ) -> object: ...


class _PygameTransformLike(Protocol):
    def scale(self, surface: object, size: tuple[int, int]) -> object: ...


class _PygameLike(Protocol):
    draw: _PygameDrawLike
    transform: _PygameTransformLike

    def Rect(self, left: int, top: int, width: int, height: int) -> _RectLike: ...


def _draw_observation_preview_below_game(
    *,
    pygame: _PygameLike,
    screen: _ScreenLike,
    fonts: ViewerFonts,
    surface: _SurfaceLike,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
    info: dict[str, object],
    control_viz: ControlViz,
) -> ViewerHitboxes:
    x = LAYOUT.preview_padding
    y = game_display_size[1] + LAYOUT.preview_gap
    width = game_display_size[0] - (2 * LAYOUT.preview_padding)
    height = screen.get_height() - y - LAYOUT.preview_padding
    if width <= 0 or height <= 0:
        return ViewerHitboxes()

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
    frame_count = _observation_tile_count(observation_shape, info=info)
    label_rail_height = 0
    if frame_count > 1:
        label_rail_height = fonts.body.render("0", True, PALETTE.text_primary).get_height() + 8
    glass_padding = 8
    label_gap = 5 if frame_count > 1 else 0
    chrome_height = (2 * glass_padding) + label_rail_height + label_gap
    control_gap = 10
    control_height = _control_viz_panel_height(width)
    available_height = (
        screen.get_height()
        - y
        - chrome_height
        - control_gap
        - control_height
        - LAYOUT.preview_padding
    )
    scale = min(width / preview_width, available_height / preview_height)
    if scale <= 0:
        return ViewerHitboxes()

    scaled_size = (
        max(1, round(preview_width * scale)),
        max(1, round(preview_height * scale)),
    )
    glass_size = (
        scaled_size[0] + (2 * glass_padding),
        scaled_size[1] + chrome_height,
    )
    glass_x = x + max(0, (width - glass_size[0]) // 2)
    glass_rect = pygame.Rect(glass_x, y, *glass_size)
    _draw_observation_glass_box(pygame=pygame, screen=screen, rect=glass_rect)

    preview_x = glass_rect.left + glass_padding
    label_y = glass_rect.top + glass_padding
    preview_y = label_y + label_rail_height + label_gap
    preview_rect = pygame.Rect(preview_x, preview_y, *scaled_size)
    _draw_observation_tile_labels(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        label_y=label_y,
        rect=preview_rect,
        observation_shape=observation_shape,
        info=info,
    )
    pygame.draw.rect(screen, PALETTE.panel_background, preview_rect)
    preview_surface = pygame.transform.scale(surface, scaled_size)
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
    _, deterministic_toggle_rect = _draw_control_viz(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=x,
        y=glass_rect.bottom + control_gap,
        width=width,
        control_viz=control_viz,
    )
    return ViewerHitboxes(deterministic_toggle=deterministic_toggle_rect)


def _draw_outer_preview_border(
    *,
    pygame: _PygameLike,
    screen: _ScreenLike,
    rect: _RectLike,
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
    pygame: _PygameLike,
    screen: _ScreenLike,
    rect: _RectLike,
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
    pygame: _PygameLike,
    screen: _ScreenLike,
    fonts: ViewerFonts,
    label_y: int,
    rect: _RectLike,
    observation_shape: tuple[int, ...],
    info: dict[str, object],
) -> None:
    stack_size = _observation_stack_size(observation_shape, info=info)
    frame_count = _observation_tile_count(observation_shape, info=info)
    if frame_count <= 1:
        return

    columns, _ = _observation_preview_grid(frame_count)
    newest_index = stack_size - 1
    for index in range(frame_count):
        _, column = divmod(index, columns)
        tile_left = rect.left + round((column * rect.size[0]) / columns)
        tile_right = rect.left + round(((column + 1) * rect.size[0]) / columns)
        tile_center_x = tile_left + ((tile_right - tile_left) // 2)
        is_newest = index == newest_index
        is_minimap = index >= stack_size
        color = _observation_tile_label_color(is_newest=is_newest, is_minimap=is_minimap)
        _draw_observation_tile_label(
            pygame=pygame,
            screen=screen,
            font=fonts.body,
            label=(
                "map"
                if is_minimap
                else _observation_tile_time_label(index=index, stack_size=stack_size)
            ),
            center_x=tile_center_x,
            y=label_y,
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
    pygame: _PygameLike,
    screen: _ScreenLike,
    rect: _RectLike,
    observation_shape: tuple[int, ...],
    info: dict[str, object],
) -> None:
    stack_size = _observation_stack_size(observation_shape, info=info)
    frame_count = _observation_tile_count(observation_shape, info=info)
    if frame_count <= 1:
        return

    columns, rows = _observation_preview_grid(frame_count)
    newest_index = stack_size - 1
    for index in range(frame_count):
        row, column = divmod(index, columns)
        tile_left = rect.left + round((column * rect.size[0]) / columns)
        tile_right = rect.left + round(((column + 1) * rect.size[0]) / columns)
        tile_top = rect.top + round((row * rect.size[1]) / rows)
        tile_bottom = rect.top + round(((row + 1) * rect.size[1]) / rows)
        tile_rect = pygame.Rect(
            tile_left,
            tile_top,
            tile_right - tile_left,
            tile_bottom - tile_top,
        )
        is_newest = index == newest_index
        border_color = PALETTE.text_accent if is_newest else PALETTE.panel_border
        pygame.draw.rect(screen, border_color, tile_rect, width=2 if is_newest else 1)


def _observation_tile_count(
    observation_shape: tuple[int, ...],
    *,
    info: dict[str, object],
) -> int:
    stack_size = _observation_stack_size(observation_shape, info=info)
    return stack_size + (1 if _observation_minimap_layer(info) else 0)


def _draw_observation_tile_label(
    *,
    pygame: _PygameLike,
    screen: _ScreenLike,
    font: RenderFont,
    label: str,
    center_x: int,
    y: int,
    color: Color,
    active: bool,
) -> None:
    text_surface = font.render(label, True, color)
    pad_x = 7
    pad_y = 3
    label_rect = pygame.Rect(
        center_x - ((text_surface.get_width() + (2 * pad_x)) // 2),
        y,
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
