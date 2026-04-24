# src/rl_fzerox/ui/watch/view/panels/draw.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import ControllerState, FZeroXTelemetry
from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches
from rl_fzerox.ui.watch.view.components.cockpit import _draw_control_viz
from rl_fzerox.ui.watch.view.components.tokens import _draw_flag_viz
from rl_fzerox.ui.watch.view.panels.model import _build_panel_columns, _panel_tab_width
from rl_fzerox.ui.watch.view.panels.viz import _wrap_text
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import (
    MouseRect,
    PanelColumns,
    PanelLine,
    PanelSection,
    PygameModule,
    PygameRect,
    PygameSurface,
    RenderFont,
    TextSurface,
    ViewerFonts,
    ViewerHitboxes,
)


@dataclass(frozen=True, slots=True)
class SidePanelData:
    """Presentation data needed to build and draw the watch side panel."""

    episode: int
    info: dict[str, object]
    reset_info: dict[str, object]
    episode_reward: float
    paused: bool
    control_state: ControllerState
    gas_level: float
    thrust_warning_threshold: float | None
    boost_active: bool
    boost_lamp_level: float
    action_mask_branches: ActionMaskBranches
    policy_label: str | None
    policy_curriculum_stage: str | None
    policy_num_timesteps: int | None
    policy_deterministic: bool | None
    policy_action: ActionValue | None
    policy_reload_age_seconds: float | None
    policy_reload_error: str | None
    best_finish_position: int | None
    best_finish_times: dict[str, int]
    latest_finish_times: dict[str, int]
    latest_finish_deltas_ms: dict[str, int]
    track_pool_records: tuple[dict[str, object], ...]
    panel_tab_index: int
    continuous_drive_deadzone: float
    continuous_air_brake_mode: str
    continuous_air_brake_disabled: bool
    action_repeat: int
    max_episode_steps: int
    progress_frontier_stall_limit_frames: int | None
    stuck_min_speed_kph: float
    game_display_size: tuple[int, int]
    observation_shape: tuple[int, ...]
    observation_state: StateVector | None
    observation_state_feature_names: tuple[str, ...]
    telemetry: FZeroXTelemetry | None


def _draw_side_panel(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    panel_rect: PygameRect,
    data: SidePanelData,
) -> ViewerHitboxes:
    pygame.draw.rect(screen, PALETTE.panel_background, panel_rect)
    pygame.draw.line(
        screen,
        PALETTE.panel_border,
        panel_rect.topleft,
        panel_rect.bottomleft,
        width=2,
    )

    x = panel_rect.x + LAYOUT.panel_padding
    y = panel_rect.y + LAYOUT.panel_padding
    panel_width = panel_rect.width - (2 * LAYOUT.panel_padding)
    columns = _build_panel_columns(
        episode=data.episode,
        info=data.info,
        reset_info=data.reset_info,
        episode_reward=data.episode_reward,
        paused=data.paused,
        control_state=data.control_state,
        gas_level=data.gas_level,
        thrust_warning_threshold=data.thrust_warning_threshold,
        boost_active=data.boost_active,
        boost_lamp_level=data.boost_lamp_level,
        action_mask_branches=data.action_mask_branches,
        policy_label=data.policy_label,
        policy_curriculum_stage=data.policy_curriculum_stage,
        policy_num_timesteps=data.policy_num_timesteps,
        policy_deterministic=data.policy_deterministic,
        policy_action=data.policy_action,
        policy_reload_age_seconds=data.policy_reload_age_seconds,
        policy_reload_error=data.policy_reload_error,
        best_finish_position=data.best_finish_position,
        best_finish_times=data.best_finish_times,
        latest_finish_times=data.latest_finish_times,
        latest_finish_deltas_ms=data.latest_finish_deltas_ms,
        track_pool_records=data.track_pool_records,
        continuous_drive_deadzone=data.continuous_drive_deadzone,
        continuous_air_brake_mode=data.continuous_air_brake_mode,
        continuous_air_brake_disabled=data.continuous_air_brake_disabled,
        action_repeat=data.action_repeat,
        max_episode_steps=data.max_episode_steps,
        progress_frontier_stall_limit_frames=data.progress_frontier_stall_limit_frames,
        stuck_min_speed_kph=data.stuck_min_speed_kph,
        game_display_size=data.game_display_size,
        observation_shape=data.observation_shape,
        observation_state=data.observation_state,
        observation_state_feature_names=data.observation_state_feature_names,
        telemetry=data.telemetry,
    )

    selected_tab_index = data.panel_tab_index % len(_PANEL_TABS)
    selected_tab_width = _panel_tab_width(panel_width)
    selected_tab_x = panel_rect.right - LAYOUT.panel_padding - selected_tab_width

    y = _draw_panel_title(
        screen=screen,
        fonts=fonts,
        x=x,
        y=y,
        width=panel_width,
        title="F-Zero X Watch",
        subtitle=_panel_subtitle(data.policy_label),
    )
    y += LAYOUT.title_section_gap
    y, tab_rects = _draw_panel_tabs(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=selected_tab_x,
        y=y,
        width=selected_tab_width,
        selected_index=selected_tab_index,
    )
    y += LAYOUT.title_section_gap

    _draw_column(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=selected_tab_x,
        y=y,
        width=selected_tab_width,
        sections=_panel_tab_sections(columns, selected_tab_index),
    )
    return ViewerHitboxes(panel_tabs=tab_rects)


def _draw_column(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    sections: list[PanelSection],
) -> int:
    current_y = y
    for section_index, section in enumerate(sections):
        current_y = _draw_section(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=current_y,
            width=width,
            section=section,
        )
        if section_index < len(sections) - 1:
            current_y += LAYOUT.section_gap
    return current_y


def _panel_subtitle(policy_label: str | None) -> str:
    if policy_label is None:
        return "manual control"
    return f"policy: {policy_label}"


_PANEL_TABS: tuple[str, ...] = (
    "Session",
    "Game",
    "State",
)


def _panel_tab_sections(columns: PanelColumns, selected_index: int) -> list[PanelSection]:
    tabs = (columns.left, columns.middle, columns.stats)
    return tabs[selected_index % len(tabs)]


def _draw_panel_tabs(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    selected_index: int,
) -> tuple[int, tuple[MouseRect | None, ...]]:
    gap = max(2, LAYOUT.inline_value_gap // 2)
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
    for index, label in enumerate(_PANEL_TABS):
        active = index == (selected_index % len(_PANEL_TABS))
        label_surface = fonts.small.render(
            label,
            True,
            PALETTE.text_primary if active else PALETTE.text_muted,
        )
        tab_width = max(64, label_surface.get_width() + 16)
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
            tab_rects.extend((None,) * (len(_PANEL_TABS) - len(tab_rects)))
            break
    hint_surface = fonts.small.render(
        _panel_tab_hint(selected_index),
        True,
        PALETTE.text_muted,
    )
    hint_x = x + width - hint_surface.get_width()
    if hint_x > current_x:
        screen.blit(
            hint_surface,
            (hint_x, tab_y + max(0, (tab_height - hint_surface.get_height()) // 2)),
        )
    while len(tab_rects) < len(_PANEL_TABS):
        tab_rects.append(None)
    return tab_y + tab_height, tuple(tab_rects)


def _panel_tab_hint(selected_index: int) -> str:
    tab_number = (selected_index % len(_PANEL_TABS)) + 1
    return f"Tab {tab_number}/{len(_PANEL_TABS)}"


def _draw_panel_title(
    *,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    title: str,
    subtitle: str,
) -> int:
    title_surface = fonts.title.render(title, True, PALETTE.text_primary)
    fitted_subtitle = _fit_text(fonts.small, subtitle, width)
    subtitle_surface = fonts.small.render(fitted_subtitle, True, PALETTE.text_muted)
    screen.blit(title_surface, (x, y))
    subtitle_y = y + title_surface.get_height() + LAYOUT.title_gap
    screen.blit(subtitle_surface, (x, subtitle_y))
    return subtitle_y + subtitle_surface.get_height()


def _draw_section(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    section: PanelSection,
) -> int:
    section_title = fonts.section.render(section.title, True, PALETTE.text_primary)
    screen.blit(section_title, (x, y))
    y += section_title.get_height() + LAYOUT.section_title_gap
    pygame.draw.line(screen, PALETTE.panel_border, (x, y), (x + width, y), width=1)
    y += LAYOUT.section_rule_gap

    for line in section.lines:
        if line.divider:
            y = _draw_panel_divider(pygame=pygame, screen=screen, x=x, y=y, width=width)
            continue
        if line.label and line.wrap:
            y = _draw_wrapped_line(
                screen=screen,
                fonts=fonts,
                x=x,
                y=y,
                width=width,
                label=line.label,
                value=line.value,
                color=line.color,
                min_value_lines=line.min_value_lines,
            )
            continue
        if line.label:
            y = _draw_labeled_value_line(
                pygame=pygame,
                screen=screen,
                fonts=fonts,
                x=x,
                y=y,
                width=width,
                line=line,
            )
            continue

        value_surface = fonts.small.render(line.value, True, line.color)
        line_height = _font_line_height(fonts.small)
        screen.blit(value_surface, (x, _centered_text_y(y, line_height, value_surface)))
        y += line_height + LAYOUT.line_gap

    if section.control_viz is not None:
        y += LAYOUT.control_viz_gap
        y, _ = _draw_control_viz(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            width=width,
            control_viz=section.control_viz,
        )
    if section.flag_viz is not None:
        y += LAYOUT.control_viz_gap
        y = _draw_flag_viz(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            flag_viz=section.flag_viz,
        )

    return y


def _draw_panel_divider(*, pygame, screen, x: int, y: int, width: int) -> int:
    line_y = y + max(1, LAYOUT.line_gap // 2)
    pygame.draw.line(screen, PALETTE.panel_border, (x, line_y), (x + width, line_y), width=1)
    return line_y + LAYOUT.line_gap + 1


def _draw_wrapped_line(
    *,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    label: str,
    value: str,
    color,
    min_value_lines: int,
) -> int:
    label_surface = fonts.small.render(label, True, PALETTE.text_muted)
    label_height = _font_line_height(fonts.small)
    screen.blit(label_surface, (x, _centered_text_y(y, label_height, label_surface)))
    y += label_height + LAYOUT.line_gap

    wrapped_lines = _wrap_text(
        fonts.small,
        value,
        width - LAYOUT.wrapped_value_indent,
    )
    for wrapped_line in wrapped_lines:
        value_surface = fonts.small.render(wrapped_line, True, color)
        value_height = _font_line_height(fonts.small)
        screen.blit(
            value_surface,
            (x + LAYOUT.wrapped_value_indent, _centered_text_y(y, value_height, value_surface)),
        )
        y += value_height + LAYOUT.line_gap

    if len(wrapped_lines) < min_value_lines:
        blank_height = _font_line_height(fonts.small)
        y += (min_value_lines - len(wrapped_lines)) * (blank_height + LAYOUT.line_gap)

    return y


def _draw_labeled_value_line(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    line: PanelLine,
) -> int:
    label_font = fonts.record_header if line.heading else fonts.small
    label_color = PALETTE.text_primary if line.heading else PALETTE.text_muted
    label_surface = label_font.render(line.label, True, label_color)
    label_height = _font_line_height(label_font)

    value_font = fonts.small if line.heading else fonts.body
    value_height = _font_line_height(value_font)
    status_text_surface = fonts.small.render(line.status_text, True, line.color)
    status_text_height = _font_line_height(fonts.small)
    row_height = max(label_height, value_height, status_text_height)

    screen.blit(label_surface, (x, _centered_text_y(y, row_height, label_surface)))

    inline_value_space = max(
        0,
        width - label_surface.get_width() - LAYOUT.inline_value_gap,
    )

    if line.status_icon is not None:
        icon_slot_width = row_height
        center = (
            x + width - (icon_slot_width // 2),
            y + (row_height // 2),
        )
        if line.status_text:
            status_gap = max(1, LAYOUT.inline_value_gap // 2)
            status_text_space = max(0, inline_value_space - icon_slot_width - status_gap)
            fitted_status_text = _fit_text(fonts.small, line.status_text, status_text_space)
            status_text_surface = fonts.small.render(fitted_status_text, True, line.color)
            text_x = center[0] - (icon_slot_width // 2) - status_gap
            screen.blit(
                status_text_surface,
                (
                    text_x - status_text_surface.get_width(),
                    _centered_text_y(y, row_height, status_text_surface),
                ),
            )
        _draw_status_icon(
            pygame,
            screen,
            icon=line.status_icon,
            color=line.color,
            center=center,
        )
        return y + row_height + LAYOUT.line_gap

    fitted_value = _fit_text(value_font, line.value, inline_value_space)
    value_surface = value_font.render(fitted_value, True, line.color)
    value_x = x + width - value_surface.get_width()
    screen.blit(value_surface, (value_x, _centered_text_y(y, row_height, value_surface)))
    return y + row_height + LAYOUT.line_gap


def _font_line_height(font: RenderFont) -> int:
    """Use a stable row height so glyphs like 'g' do not shift the panel."""

    return font.render("Ag", True, PALETTE.text_primary).get_height()


def _centered_text_y(y: int, row_height: int, surface: TextSurface) -> int:
    return y + max(0, (row_height - surface.get_height()) // 2)


def _draw_status_icon(
    pygame: PygameModule,
    screen: PygameSurface,
    *,
    icon: str,
    color: Color,
    center: tuple[int, int],
) -> None:
    x, y = center
    if icon == "none":
        pygame.draw.circle(screen, color, center, 4, width=1)
        return
    if icon == "in_range":
        pygame.draw.line(screen, color, (x - 5, y), (x - 2, y + 3), width=2)
        pygame.draw.line(screen, color, (x - 2, y + 3), (x + 5, y - 4), width=2)
        return
    if icon == "outside":
        triangle = ((x, y - 5), (x - 5, y + 4), (x + 5, y + 4))
        pygame.draw.polygon(screen, color, triangle, width=1)
        pygame.draw.line(screen, color, (x, y - 2), (x, y + 1), width=1)
        pygame.draw.circle(screen, color, (x, y + 3), 1)


def _fit_text(font, text: str, max_width: int) -> str:
    if font.render(text, True, PALETTE.text_primary).get_width() <= max_width:
        return text

    suffix = "..."
    suffix_width = font.render(suffix, True, PALETTE.text_primary).get_width()
    if suffix_width >= max_width:
        return ""

    for end_index in range(len(text), 0, -1):
        candidate = text[:end_index] + suffix
        if font.render(candidate, True, PALETTE.text_primary).get_width() <= max_width:
            return candidate
    return suffix
