# src/rl_fzerox/ui/watch/view/panels/rendering/draw.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry, RaceControlState
from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches
from rl_fzerox.core.runtime_spec.schema import PolicyConfig, TrainConfig
from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesSnapshot
from rl_fzerox.ui.watch.runtime.cnn import CnnActivationSnapshot
from rl_fzerox.ui.watch.view.auxiliary_metrics import AuxiliaryEpisodeMetricsSnapshot
from rl_fzerox.ui.watch.view.panels.core.model import _build_panel_columns
from rl_fzerox.ui.watch.view.panels.core.tabs import PanelTabRegistry
from rl_fzerox.ui.watch.view.panels.rendering.section_renderer import _draw_column
from rl_fzerox.ui.watch.view.panels.rendering.tab_bar import (
    _draw_panel_tabs,
    _draw_text_tabs,
)
from rl_fzerox.ui.watch.view.panels.rendering.text import _fit_text
from rl_fzerox.ui.watch.view.panels.visuals.cnn import _draw_cnn_tab
from rl_fzerox.ui.watch.view.panels.visuals.live import _draw_live_tab
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
    PanelColumns,
    PanelSection,
    PygameModule,
    PygameRect,
    PygameSurface,
    RecordCourseHitbox,
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
    control_state: RaceControlState
    gas_level: float
    thrust_warning_threshold: float | None
    boost_active: bool
    boost_lamp_level: float
    action_mask_branches: ActionMaskBranches
    policy_label: str | None
    policy_curriculum_stage: str | None
    policy_num_timesteps: int | None
    policy_experience_frames: int | None
    policy_deterministic: bool | None
    manual_control_enabled: bool
    policy_action: ActionValue | None
    policy_reload_age_seconds: float | None
    policy_reload_error: str | None
    cnn_activations: CnnActivationSnapshot | None
    best_finish_position: int | None
    best_finish_ranks: dict[str, int]
    best_finish_rank_times: dict[str, int]
    best_finish_rank_setups: dict[str, dict[str, str | int]]
    best_finish_times: dict[str, int]
    best_finish_time_ranks: dict[str, int]
    best_finish_time_setups: dict[str, dict[str, str | int]]
    latest_finish_times: dict[str, int]
    latest_finish_deltas_ms: dict[str, int]
    failed_track_attempts: frozenset[str]
    track_pool_records: tuple[dict[str, object], ...]
    panel_tab_index: int
    cnn_layer_tab_index: int
    record_tab_index: int
    panel_tabs: PanelTabRegistry
    continuous_drive_deadzone: float
    continuous_air_brake_axis_index: int | None
    continuous_air_brake_deadzone: float
    continuous_air_brake_full_threshold: float
    continuous_air_brake_min_duty: float
    continuous_air_brake_mode: str
    continuous_air_brake_disabled: bool
    action_repeat: int
    max_episode_steps: int
    progress_frontier_stall_limit_frames: int | None
    stuck_min_speed_kph: float
    game_display_size: tuple[int, int]
    observation_shape: tuple[int, ...] | None
    observation_state: StateVector | None
    observation_state_reference: StateVector | None
    observation_state_feature_names: tuple[str, ...]
    policy_auxiliary_state_predictions: dict[str, object] | None
    policy_auxiliary_state_targets: dict[str, object] | None
    auxiliary_episode_metrics: AuxiliaryEpisodeMetricsSnapshot | None
    live_episode_series: EpisodeLiveSeriesSnapshot | None
    telemetry: FZeroXTelemetry | None
    emulator_renderer: str
    watch_device: str
    train_config: TrainConfig | None
    policy_config: PolicyConfig | None


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
        policy_experience_frames=data.policy_experience_frames,
        policy_deterministic=data.policy_deterministic,
        manual_control_enabled=data.manual_control_enabled,
        policy_action=data.policy_action,
        policy_reload_age_seconds=data.policy_reload_age_seconds,
        policy_reload_error=data.policy_reload_error,
        best_finish_position=data.best_finish_position,
        best_finish_ranks=data.best_finish_ranks,
        best_finish_rank_times=data.best_finish_rank_times,
        best_finish_rank_setups=data.best_finish_rank_setups,
        best_finish_times=data.best_finish_times,
        best_finish_time_ranks=data.best_finish_time_ranks,
        best_finish_time_setups=data.best_finish_time_setups,
        latest_finish_times=data.latest_finish_times,
        latest_finish_deltas_ms=data.latest_finish_deltas_ms,
        failed_track_attempts=data.failed_track_attempts,
        track_pool_records=data.track_pool_records,
        continuous_drive_deadzone=data.continuous_drive_deadzone,
        continuous_air_brake_axis_index=data.continuous_air_brake_axis_index,
        continuous_air_brake_deadzone=data.continuous_air_brake_deadzone,
        continuous_air_brake_full_threshold=data.continuous_air_brake_full_threshold,
        continuous_air_brake_min_duty=data.continuous_air_brake_min_duty,
        continuous_air_brake_mode=data.continuous_air_brake_mode,
        continuous_air_brake_disabled=data.continuous_air_brake_disabled,
        action_repeat=data.action_repeat,
        max_episode_steps=data.max_episode_steps,
        progress_frontier_stall_limit_frames=data.progress_frontier_stall_limit_frames,
        stuck_min_speed_kph=data.stuck_min_speed_kph,
        game_display_size=data.game_display_size,
        observation_shape=data.observation_shape,
        observation_state=data.observation_state,
        observation_state_reference=data.observation_state_reference,
        observation_state_feature_names=data.observation_state_feature_names,
        policy_auxiliary_state_predictions=data.policy_auxiliary_state_predictions,
        policy_auxiliary_state_targets=data.policy_auxiliary_state_targets,
        auxiliary_episode_metrics=data.auxiliary_episode_metrics,
        telemetry=data.telemetry,
        emulator_renderer=data.emulator_renderer,
        watch_device=data.watch_device,
        train_config=data.train_config,
        policy_config=data.policy_config,
    )

    selected_tab_index = data.panel_tabs.normalize(data.panel_tab_index)

    y = _draw_panel_title(
        screen=screen,
        fonts=fonts,
        x=x,
        y=y,
        width=panel_width,
        title="F-Zero X Watch",
        subtitle=_panel_subtitle(
            data.policy_label,
            manual_control_enabled=data.manual_control_enabled,
        ),
    )
    y += LAYOUT.title_section_gap
    y, tab_rects = _draw_panel_tabs(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=x,
        y=y,
        width=panel_width,
        selected_index=selected_tab_index,
        panel_tabs=data.panel_tabs,
    )
    y += LAYOUT.title_section_gap

    record_tab_rects: tuple[tuple[int, int, int, int] | None, ...] = ()
    cnn_layer_tab_rects: tuple[tuple[int, int, int, int] | None, ...] = ()
    record_course_hitboxes: tuple[RecordCourseHitbox, ...] = ()
    state_feature_hitboxes = ()
    selected_tab_key = data.panel_tabs.key(selected_tab_index)
    if selected_tab_key == "cnn":
        _, cnn_layer_tab_rects = _draw_cnn_tab(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            width=panel_width,
            activations=data.cnn_activations,
            selected_layer_page_index=data.cnn_layer_tab_index,
        )
    elif selected_tab_key == "live":
        _draw_live_tab(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            width=panel_width,
            height=panel_rect.bottom - y - LAYOUT.panel_padding,
            info=data.info,
            live_series=data.live_episode_series,
        )
    elif selected_tab_key == "records":
        record_sections = columns.records
        if len(record_sections) > 1:
            y, record_tab_rects = _draw_text_tabs(
                pygame=pygame,
                screen=screen,
                fonts=fonts,
                x=x,
                y=y,
                width=panel_width,
                labels=tuple(_record_tab_label(section.title) for section in record_sections),
                selected_index=data.record_tab_index,
            )
            y += LAYOUT.title_section_gap
            record_sections = _record_tab_sections(record_sections, data.record_tab_index)
        _, record_hitboxes = _draw_column(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            width=panel_width,
            sections=record_sections,
        )
        record_course_hitboxes = record_hitboxes.record_courses
    else:
        _, panel_hitboxes = _draw_column(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            width=panel_width,
            sections=_panel_tab_sections(
                columns,
                selected_tab_index,
                panel_tabs=data.panel_tabs,
            ),
        )
        state_feature_hitboxes = panel_hitboxes.state_features
    return ViewerHitboxes(
        panel_tabs=tab_rects,
        cnn_layer_tabs=cnn_layer_tab_rects,
        record_tabs=record_tab_rects,
        record_courses=record_course_hitboxes,
        state_features=state_feature_hitboxes,
    )


def _panel_subtitle(policy_label: str | None, *, manual_control_enabled: bool) -> str:
    if policy_label is None:
        return "manual control"
    if manual_control_enabled:
        return f"manual control (policy loaded: {policy_label})"
    return f"policy: {policy_label}"


def _panel_tab_sections(
    columns: PanelColumns,
    selected_index: int,
    *,
    panel_tabs: PanelTabRegistry,
) -> list[PanelSection]:
    tab_key = panel_tabs.key(selected_index)
    if tab_key == "run":
        return columns.left
    if tab_key == "details":
        return columns.middle
    if tab_key == "state":
        return columns.stats
    if tab_key == "aux":
        return columns.aux
    if tab_key == "records":
        return columns.records
    if tab_key == "career":
        return columns.career
    if tab_key == "train":
        return columns.train
    return []


def _record_tab_sections(
    sections: list[PanelSection],
    selected_index: int,
) -> list[PanelSection]:
    if not sections:
        return []
    return [sections[selected_index % len(sections)]]


def _record_tab_label(section_title: str) -> str:
    return section_title.removesuffix(" Cup")


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
