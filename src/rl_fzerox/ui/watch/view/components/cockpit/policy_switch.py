# src/rl_fzerox/ui/watch/view/components/cockpit/policy_switch.py
from __future__ import annotations

from importlib.resources import as_file, files

from rl_fzerox.ui.watch.view.components.cockpit.style import COCKPIT_PANEL_STYLE
from rl_fzerox.ui.watch.view.screen.types import (
    MouseRect,
    PygameModule,
    PygameSurface,
    RenderFont,
)

_switch_assets: dict[str, PygameSurface] = {}


def draw_policy_mode_switch(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    font: RenderFont,
    x: int,
    y: int,
    deterministic_policy: bool | None,
) -> MouseRect | None:
    style = COCKPIT_PANEL_STYLE.policy_mode_switch
    enabled = deterministic_policy is not None
    deterministic = deterministic_policy is True
    sprite = _load_switch_asset(
        pygame=pygame,
        asset_name=_switch_asset_name(deterministic=deterministic, enabled=enabled),
    )
    screen.blit(sprite, (x, y))

    label_text = "deterministic" if deterministic else "stochastic"
    label_color = style.active_text if enabled else style.inactive_text
    label = font.render(label_text, True, label_color)
    label_x = x + style.switch_width + style.label_gap
    label_y = y + (style.height // 2) - (label.get_height() // 2)
    screen.blit(label, (label_x, label_y))

    if not enabled:
        return None
    hitbox_width = style.switch_width + style.label_gap + label.get_width()
    return (x, y, hitbox_width, style.height)


def _switch_asset_name(*, deterministic: bool, enabled: bool) -> str:
    style = COCKPIT_PANEL_STYLE.policy_mode_switch
    if not enabled:
        return style.disabled_asset
    if deterministic:
        return style.on_asset
    return style.off_asset


def _load_switch_asset(*, pygame: PygameModule, asset_name: str) -> PygameSurface:
    cached = _switch_assets.get(asset_name)
    if cached is not None:
        return cached

    resource = files(COCKPIT_PANEL_STYLE.policy_mode_switch.asset_package).joinpath(asset_name)
    with as_file(resource) as path:
        image = pygame.image.load(str(path)).convert_alpha()
    _switch_assets[asset_name] = image
    return image
