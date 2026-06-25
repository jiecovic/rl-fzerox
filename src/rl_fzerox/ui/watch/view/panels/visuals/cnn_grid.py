# src/rl_fzerox/ui/watch/view/panels/visuals/cnn_grid.py
"""Grid overlays for tiled CNN activation images.

The activation capture packs channels into one RGB image. These helpers draw
channel separators and hatch the unused tiles that appear when the channel
count does not fill the final grid row.
"""

from __future__ import annotations

from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PygameModule, PygameSurface


def _draw_grid_overlay(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    x: int,
    y: int,
    size: tuple[int, int],
    grid_shape: tuple[int, int],
    used_tiles: int,
) -> None:
    rows, columns = grid_shape
    width, height = size
    _draw_unused_tile_hatches(
        pygame=pygame,
        screen=screen,
        x=x,
        y=y,
        rects=_unused_tile_rects(size=size, grid_shape=grid_shape, used_tiles=used_tiles),
    )
    for line_x in _separator_positions(width, columns):
        absolute_x = x + line_x
        pygame.draw.line(
            screen,
            PALETTE.cnn_grid_separator,
            (absolute_x, y),
            (absolute_x, y + height - 1),
            width=1,
        )
    for line_y in _separator_positions(height, rows):
        absolute_y = y + line_y
        pygame.draw.line(
            screen,
            PALETTE.cnn_grid_separator,
            (x, absolute_y),
            (x + width - 1, absolute_y),
            width=1,
        )
    pygame.draw.rect(
        screen,
        PALETTE.cnn_grid_separator,
        (x, y, width, height),
        width=1,
    )


def _draw_unused_tile_hatches(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    x: int,
    y: int,
    rects: tuple[tuple[int, int, int, int], ...],
) -> None:
    previous_clip = screen.get_clip()
    try:
        for rel_x, rel_y, width, height in rects:
            tile_rect = pygame.Rect(x + rel_x, y + rel_y, width, height)
            screen.set_clip(tile_rect)
            for offset in range(-height, width, 6):
                pygame.draw.line(
                    screen,
                    PALETTE.cnn_grid_unused_hatch,
                    (tile_rect.left + offset, tile_rect.bottom - 1),
                    (tile_rect.left + offset + height - 1, tile_rect.top),
                    width=1,
                )
    finally:
        screen.set_clip(previous_clip)


def _unused_tile_rects(
    *,
    size: tuple[int, int],
    grid_shape: tuple[int, int],
    used_tiles: int,
) -> tuple[tuple[int, int, int, int], ...]:
    rows, columns = grid_shape
    width, height = size
    if rows <= 0 or columns <= 0:
        return ()
    first_unused = max(0, min(used_tiles, rows * columns))
    rects: list[tuple[int, int, int, int]] = []
    for index in range(first_unused, rows * columns):
        row, column = divmod(index, columns)
        left, right = _partition_bounds(width, columns, column)
        top, bottom = _partition_bounds(height, rows, row)
        if right > left and bottom > top:
            rects.append((left, top, right - left, bottom - top))
    return tuple(rects)


def _separator_positions(length: int, parts: int) -> tuple[int, ...]:
    if length <= 1 or parts <= 1:
        return ()
    return tuple(
        position
        for index in range(1, parts)
        if 0 < (position := round(index * length / parts)) < length
    )


def _partition_bounds(length: int, parts: int, index: int) -> tuple[int, int]:
    if length <= 0 or parts <= 0:
        return (0, 0)
    return (
        round(index * length / parts),
        round((index + 1) * length / parts),
    )
