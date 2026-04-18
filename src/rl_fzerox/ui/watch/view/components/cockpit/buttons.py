# src/rl_fzerox/ui/watch/view/components/cockpit/buttons.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.components.cockpit.style import (
    BUTTON_ACTIVE_TEXT,
    BUTTON_FACE_FILL,
    BUTTON_FACE_HIGHLIGHT,
    BUTTON_FACE_INNER,
    BUTTON_SHADOW,
    LEAN_ACTIVE_BORDER,
    LEAN_ACTIVE_FILL,
)
from rl_fzerox.ui.watch.view.components.effects import (
    BOOST_EDGE_GLOW,
    GLASS_EDGE_GLOW,
    GLASS_SHADOW,
    GLASS_SHEEN,
)
from rl_fzerox.ui.watch.view.components.effects import (
    blend_color as _blend_color,
)
from rl_fzerox.ui.watch.view.components.effects import (
    draw_alpha_circle as _draw_alpha_circle,
)
from rl_fzerox.ui.watch.view.components.effects import (
    draw_alpha_polygon as _draw_alpha_polygon,
)
from rl_fzerox.ui.watch.view.components.effects import (
    offset_points as _offset_points,
)
from rl_fzerox.ui.watch.view.screen.theme import PALETTE


def _draw_lean_button(
    *,
    pygame,
    screen,
    font,
    x: int,
    y: int,
    width: int,
    direction: int,
    active: bool,
) -> None:
    from rl_fzerox.ui.watch.view.components.tokens import _pill_height

    height = _pill_height(font) + 2
    rect = pygame.Rect(x, y, width, height)
    points = _lean_button_points(rect, direction=direction)
    inner_points = _lean_button_points(rect.inflate(-5, -5), direction=direction)
    pygame.draw.polygon(screen, BUTTON_SHADOW, _offset_points(points, dx=1, dy=2))

    fill_color = LEAN_ACTIVE_FILL if active else BUTTON_FACE_FILL
    border_color = LEAN_ACTIVE_BORDER if active else PALETTE.flag_inactive_border
    text_color = BUTTON_ACTIVE_TEXT if active else PALETTE.text_muted
    pygame.draw.polygon(screen, fill_color, points)
    pygame.draw.polygon(screen, BUTTON_FACE_INNER if not active else LEAN_ACTIVE_FILL, inner_points)
    _draw_alpha_polygon(
        pygame=pygame,
        screen=screen,
        points=_lean_button_sheen_points(rect, direction=direction),
        color=GLASS_SHEEN,
    )
    _draw_alpha_polygon(
        pygame=pygame,
        screen=screen,
        points=_lean_button_shadow_points(rect, direction=direction),
        color=GLASS_SHADOW,
    )
    if active:
        _draw_alpha_polygon(
            pygame=pygame,
            screen=screen,
            points=_offset_points(points, dx=0, dy=0),
            color=GLASS_EDGE_GLOW,
        )
        pygame.draw.polygon(screen, fill_color, inner_points)
        _draw_alpha_polygon(
            pygame=pygame,
            screen=screen,
            points=_lean_button_sheen_points(rect, direction=direction),
            color=GLASS_SHEEN,
        )
        _draw_alpha_polygon(
            pygame=pygame,
            screen=screen,
            points=_lean_button_shadow_points(rect, direction=direction),
            color=GLASS_SHADOW,
        )
    pygame.draw.polygon(screen, border_color, points, width=2 if active else 1)

    highlight_start = (rect.left + 10, rect.top + 4)
    highlight_end = (rect.right - 10, rect.top + 4)
    pygame.draw.line(
        screen,
        LEAN_ACTIVE_BORDER if active else BUTTON_FACE_HIGHLIGHT,
        highlight_start,
        highlight_end,
        width=1,
    )

    arrow_color = LEAN_ACTIVE_BORDER if active else PALETTE.text_muted
    arrow_center_y = rect.centery
    if direction < 0:
        arrow_points = (
            (rect.left + 8, arrow_center_y),
            (rect.left + 16, arrow_center_y - 5),
            (rect.left + 16, arrow_center_y + 5),
        )
        label_x_offset = 7
    else:
        arrow_points = (
            (rect.right - 8, arrow_center_y),
            (rect.right - 16, arrow_center_y - 5),
            (rect.right - 16, arrow_center_y + 5),
        )
        label_x_offset = -7
    pygame.draw.polygon(screen, arrow_color, arrow_points)

    label = "L" if direction < 0 else "R"
    label_surface = font.render(label, True, text_color)
    screen.blit(
        label_surface,
        (
            rect.centerx - (label_surface.get_width() // 2) + label_x_offset,
            rect.centery - (label_surface.get_height() // 2),
        ),
    )


def _lean_button_points(rect, *, direction: int) -> tuple[tuple[int, int], ...]:
    inset = 12
    if direction < 0:
        return (
            (rect.left, rect.centery),
            (rect.left + inset, rect.top),
            (rect.right, rect.top),
            (rect.right - 5, rect.centery),
            (rect.right, rect.bottom),
            (rect.left + inset, rect.bottom),
        )
    return (
        (rect.right, rect.centery),
        (rect.right - inset, rect.top),
        (rect.left, rect.top),
        (rect.left + 5, rect.centery),
        (rect.left, rect.bottom),
        (rect.right - inset, rect.bottom),
    )


def _lean_button_sheen_points(rect, *, direction: int) -> tuple[tuple[int, int], ...]:
    if direction < 0:
        return (
            (rect.left + 14, rect.top + 3),
            (rect.right - 5, rect.top + 3),
            (rect.right - 9, rect.centery - 3),
            (rect.left + 18, rect.centery - 3),
            (rect.left + 8, rect.centery),
        )
    return (
        (rect.right - 14, rect.top + 3),
        (rect.left + 5, rect.top + 3),
        (rect.left + 9, rect.centery - 3),
        (rect.right - 18, rect.centery - 3),
        (rect.right - 8, rect.centery),
    )


def _lean_button_shadow_points(rect, *, direction: int) -> tuple[tuple[int, int], ...]:
    if direction < 0:
        return (
            (rect.left + 8, rect.centery + 2),
            (rect.right - 8, rect.centery + 2),
            (rect.right - 3, rect.bottom - 3),
            (rect.left + 14, rect.bottom - 3),
        )
    return (
        (rect.right - 8, rect.centery + 2),
        (rect.left + 8, rect.centery + 2),
        (rect.left + 3, rect.bottom - 3),
        (rect.right - 14, rect.bottom - 3),
    )


def _draw_boost_button(
    *,
    pygame,
    screen,
    center: tuple[int, int],
    radius: int,
    level: float,
) -> None:
    level = max(0.0, min(1.0, level))
    active = level > 0.0
    manual_intensity = max(0.0, min(1.0, (level - 0.55) / 0.45))
    normal_intensity = max(0.0, min(1.0, level / 0.55))
    manual_dominance = manual_intensity**0.45
    pygame.draw.circle(screen, BUTTON_SHADOW, (center[0] + 1, center[1] + 2), radius)
    if active:
        _draw_alpha_circle(
            pygame=pygame,
            screen=screen,
            center=center,
            radius=radius + 5 + round(4 * manual_intensity),
            color=(
                round(150 + (40 * normal_intensity) - (72 * manual_dominance)),
                255,
                round(190 + (28 * normal_intensity) - (126 * manual_dominance)),
                round((34 * normal_intensity) + (88 * manual_intensity)),
            ),
        )
    if manual_intensity > 0.0:
        _draw_alpha_circle(
            pygame=pygame,
            screen=screen,
            center=center,
            radius=radius + 9,
            color=(78, 255, 64, round(78 * manual_intensity)),
        )
    bezel_color = _blend_color(BUTTON_FACE_FILL, (29, 58, 46), normal_intensity)
    border_color = _blend_color(
        PALETTE.flag_inactive_border,
        (132, 214, 172),
        normal_intensity,
    )
    led_outer = _blend_color((37, 60, 52), (92, 176, 130), normal_intensity)
    led_outer = _blend_color(led_outer, (34, 255, 44), manual_dominance)
    led_inner = _blend_color((50, 86, 70), (174, 234, 196), normal_intensity)
    led_inner = _blend_color(led_inner, (82, 255, 26), manual_dominance)
    pygame.draw.circle(screen, bezel_color, center, radius)
    pygame.draw.circle(screen, led_outer, center, max(1, radius - 4))
    pygame.draw.circle(screen, led_inner, center, max(1, radius - 7))
    if manual_intensity > 0.0:
        flash_core = _blend_color((90, 255, 52), (34, 255, 0), manual_dominance)
        pygame.draw.circle(screen, flash_core, center, max(1, radius - 9))
    _draw_alpha_circle(
        pygame=pygame,
        screen=screen,
        center=(center[0] - max(2, radius // 4), center[1] - max(2, radius // 4)),
        radius=max(3, radius // 2),
        color=GLASS_SHEEN,
    )
    _draw_alpha_circle(
        pygame=pygame,
        screen=screen,
        center=(center[0] + 1, center[1] + max(3, radius // 3)),
        radius=max(3, radius // 2),
        color=GLASS_SHADOW,
    )
    if active:
        _draw_alpha_circle(
            pygame=pygame,
            screen=screen,
            center=center,
            radius=radius + 2,
            color=(
                BOOST_EDGE_GLOW[0],
                BOOST_EDGE_GLOW[1],
                BOOST_EDGE_GLOW[2],
                round(24 + (58 * level) + (68 * manual_intensity)),
            ),
        )
        pygame.draw.circle(screen, led_outer, center, max(1, radius - 4))
        pygame.draw.circle(screen, led_inner, center, max(1, radius - 7))
        _draw_alpha_circle(
            pygame=pygame,
            screen=screen,
            center=(center[0] - max(2, radius // 4), center[1] - max(2, radius // 4)),
            radius=max(3, radius // 2),
            color=GLASS_SHEEN,
        )
    pygame.draw.circle(screen, border_color, center, radius, width=2 if level >= 0.75 else 1)
    _draw_alpha_circle(
        pygame=pygame,
        screen=screen,
        center=(center[0] - 3, center[1] - 4),
        radius=max(2, radius // 4),
        color=(255, 255, 255, 90),
    )
