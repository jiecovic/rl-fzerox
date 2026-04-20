# src/rl_fzerox/ui/watch/view/components/cockpit/primitives.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.screen.theme import Color
from rl_fzerox.ui.watch.view.screen.types import PygameModule, PygameSurface, RenderFont


def _draw_round_marker(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    color: Color,
    center: tuple[int, int],
    radius: int,
    outline_color: Color | None,
) -> None:
    gfxdraw = getattr(pygame, "gfxdraw", None)
    if gfxdraw is not None:
        gfxdraw.filled_circle(screen, center[0], center[1], radius, color)
        gfxdraw.aacircle(screen, center[0], center[1], radius, color)
        if outline_color is not None and radius > 1:
            gfxdraw.aacircle(screen, center[0], center[1], radius, outline_color)
        return

    pygame.draw.circle(screen, color, center, radius)
    if outline_color is not None and radius > 1:
        pygame.draw.circle(screen, outline_color, center, radius, width=1)


def _draw_centered_label(
    *,
    screen: PygameSurface,
    font: RenderFont,
    label: str,
    color: Color,
    center_x: int,
    y: int,
) -> None:
    surface = font.render(label, True, color)
    screen.blit(surface, (center_x - (surface.get_width() // 2), y))
