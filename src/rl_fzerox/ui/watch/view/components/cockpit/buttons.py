# src/rl_fzerox/ui/watch/view/components/cockpit/buttons.py
from __future__ import annotations

import math

from rl_fzerox.ui.watch.view.components.cockpit.style import (
    BUTTON_FACE_FILL,
    BUTTON_SHADOW,
    LEAN_ACTIVE_BORDER,
    LEAN_ACTIVE_FILL,
    LEAN_CONTROL_STYLE,
)
from rl_fzerox.ui.watch.view.components.effects import (
    BOOST_EDGE_GLOW,
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

    style = LEAN_CONTROL_STYLE
    height = _pill_height(font) + style.height_extra
    rect = pygame.Rect(x, y + style.y_offset, width, height)
    face = _lean_half_moon_points(rect, direction=direction)
    inner = _lean_half_moon_points(
        rect.inflate(-style.inner_inset, -style.inner_inset),
        direction=direction,
    )
    fill_color = LEAN_ACTIVE_FILL if active else BUTTON_FACE_FILL
    border_color = style.inactive_border if not active else LEAN_ACTIVE_BORDER
    text_color = LEAN_ACTIVE_BORDER if active else style.inactive_text

    pygame.draw.polygon(
        screen,
        BUTTON_SHADOW,
        tuple(
            (point_x + style.shadow_offset[0], point_y + style.shadow_offset[1])
            for point_x, point_y in face
        ),
    )
    if active:
        _draw_alpha_polygon(
            pygame=pygame,
            screen=screen,
            points=_lean_half_moon_points(
                rect.inflate(style.glow_inflate, style.glow_inflate),
                direction=direction,
            ),
            color=(
                LEAN_ACTIVE_BORDER[0],
                LEAN_ACTIVE_BORDER[1],
                LEAN_ACTIVE_BORDER[2],
                style.active_glow_alpha,
            ),
        )
    pygame.draw.polygon(screen, fill_color, face)
    pygame.draw.polygon(screen, style.inactive_inner_fill if not active else fill_color, inner)
    _draw_alpha_polygon(
        pygame=pygame,
        screen=screen,
        points=_lean_half_moon_sheen(rect, direction=direction),
        color=style.sheen,
    )
    pygame.draw.polygon(screen, border_color, face, width=2 if active else 1)
    _draw_lean_flat_edge(
        pygame=pygame,
        screen=screen,
        rect=rect,
        direction=direction,
        color=border_color,
    )

    label = "L" if direction < 0 else "R"
    label_surface = font.render(label, True, text_color)
    label_x = rect.centerx - (label_surface.get_width() // 2)
    screen.blit(
        label_surface,
        (
            label_x,
            rect.bottom - label_surface.get_height() - style.label_bottom_padding,
        ),
    )


def _lean_half_moon_points(rect, *, direction: int) -> tuple[tuple[int, int], ...]:
    steps = LEAN_CONTROL_STYLE.curve_steps
    points: list[tuple[int, int]] = []
    for index in range(steps + 1):
        theta = math.pi * (index / steps)
        y = rect.top + round((rect.height / 2) * (1.0 - math.cos(theta)))
        curve_offset = round(rect.width * math.sin(theta))
        x = rect.right - curve_offset if direction < 0 else rect.left + curve_offset
        points.append((x, y))
    if direction < 0:
        points.append((rect.right, rect.top))
    else:
        points.append((rect.left, rect.top))
    return tuple(points)


def _lean_half_moon_sheen(rect, *, direction: int) -> tuple[tuple[int, int], ...]:
    inset = max(3, rect.width // 5)
    if direction < 0:
        return (
            (rect.right - 4, rect.top + 4),
            (rect.left + inset, rect.top + 6),
            (rect.left + inset + 2, rect.centery - 3),
            (rect.right - 5, rect.centery - 4),
        )
    return (
        (rect.left + 4, rect.top + 4),
        (rect.right - inset, rect.top + 6),
        (rect.right - inset - 2, rect.centery - 3),
        (rect.left + 5, rect.centery - 4),
    )


def _draw_lean_flat_edge(*, pygame, screen, rect, direction: int, color) -> None:
    x = rect.right if direction < 0 else rect.left
    inset = LEAN_CONTROL_STYLE.flat_edge_inset
    pygame.draw.line(screen, color, (x, rect.top + inset), (x, rect.bottom - inset), width=1)


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
