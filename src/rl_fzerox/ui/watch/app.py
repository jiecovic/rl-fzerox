# src/rl_fzerox/ui/watch/app.py
from __future__ import annotations

from collections.abc import Callable

from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.input import (
    SpeedKeyRepeat,
    ViewerInput,
    _poll_viewer_input,
    mouse_over_clickable,
)
from rl_fzerox.ui.watch.recording import ViewerRecorder, open_viewer_recorder
from rl_fzerox.ui.watch.runtime import (
    WatchWorker,
    apply_viewer_input,
    drain_snapshot_queue,
    start_watch_worker,
    wait_initial_snapshot,
)
from rl_fzerox.ui.watch.runtime.cnn import (
    DEFAULT_CNN_ACTIVATION_NORMALIZATION,
    next_cnn_activation_normalization,
)
from rl_fzerox.ui.watch.runtime.timing import (
    RateMeter,
    _resolve_render_fps,
)
from rl_fzerox.ui.watch.view.auxiliary_metrics import AuxiliaryEpisodeMetricsTracker
from rl_fzerox.ui.watch.view.panels.core.tabs import (
    WATCH_PANEL_TABS,
    PanelTabRegistry,
    panel_tabs_for_config,
)
from rl_fzerox.ui.watch.view.screen.frame import (
    _create_fonts,
    _ensure_screen,
    _watch_game_display_size,
)
from rl_fzerox.ui.watch.view.screen.render import draw_watch_frame
from rl_fzerox.ui.watch.view.screen.types import PygameModule, ViewerHitboxes

WatchWorkerFactory = Callable[[WatchAppConfig], WatchWorker]
ViewerHeartbeat = Callable[[], bool]

__all__ = ["WatchWorkerFactory", "run_viewer"]


def run_viewer(
    config: WatchAppConfig,
    *,
    worker_factory: WatchWorkerFactory = start_watch_worker,
    viewer_heartbeat: ViewerHeartbeat | None = None,
) -> None:
    """Run the watch UI while a worker process advances emulator state."""

    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "pygame is required for watching emulator output. "
            "Install with `pip install -e .[watch]`."
        ) from exc

    worker = worker_factory(config)
    panel_tabs = panel_tabs_for_config(config)
    pygame.init()
    render_clock = pygame.time.Clock()
    recorder: ViewerRecorder | None = None
    try:
        snapshot, worker_closed = wait_initial_snapshot(worker, viewer_heartbeat=viewer_heartbeat)
        target_render_fps = _resolve_render_fps(
            config.watch.render_fps,
            native_fps=snapshot.native_fps,
        )
        recorder = open_viewer_recorder(
            config=config,
            native_fps=snapshot.native_fps,
            render_fps=target_render_fps,
        )
        render_rate = RateMeter(window=60)
        game_display_size = _watch_game_display_size()
        screen = None
        fonts = _create_fonts(pygame)
        paused = False
        panel_tab_index = 0
        cnn_layer_tab_index = 0
        record_tab_index = 0
        cnn_normalization = DEFAULT_CNN_ACTIVATION_NORMALIZATION
        hitboxes = ViewerHitboxes()
        speed_repeat = SpeedKeyRepeat()
        auxiliary_metrics = AuxiliaryEpisodeMetricsTracker.from_policy_config(config.policy)
        auxiliary_metrics.observe_snapshot(snapshot)
        live_episode_series = getattr(snapshot, "live_episode_series", None)
        policy_observation_layout_shape = _initial_policy_observation_layout_shape(snapshot)

        while True:
            render_limit = 0 if target_render_fps is None else max(1, int(target_render_fps))
            render_clock.tick(render_limit)
            if viewer_heartbeat is not None and not viewer_heartbeat():
                return

            viewer_input = _poll_viewer_input(
                pygame,
                deterministic_toggle_rect=hitboxes.deterministic_toggle,
                panel_tab_rects=hitboxes.panel_tabs,
                cnn_layer_tab_rects=hitboxes.cnn_layer_tabs,
                record_tab_rects=hitboxes.record_tabs,
                record_course_hitboxes=hitboxes.record_courses,
                state_feature_hitboxes=hitboxes.state_features,
                speed_repeat=speed_repeat,
            )
            panel_tab_index = _next_panel_tab_index(
                panel_tab_index,
                viewer_input,
                panel_tabs=panel_tabs,
            )
            if viewer_input.cnn_layer_tab_index is not None:
                cnn_layer_tab_index = viewer_input.cnn_layer_tab_index
            if viewer_input.record_tab_index is not None:
                record_tab_index = viewer_input.record_tab_index
            if viewer_input.toggle_cnn_normalization:
                cnn_normalization = next_cnn_activation_normalization(cnn_normalization)
            paused = apply_viewer_input(
                worker.command_queue,
                viewer_input,
                paused=paused,
                cnn_visualization_enabled=panel_tab_index == panel_tabs.cnn_index,
                auxiliary_visualization_enabled=panel_tab_index
                in {panel_tabs.state_index, panel_tabs.aux_index},
                live_visualization_enabled=panel_tab_index == panel_tabs.live_index,
                cnn_normalization=cnn_normalization,
            )
            if viewer_input.quit_requested:
                return

            latest_snapshot, worker_closed = drain_snapshot_queue(
                worker,
                worker_closed=worker_closed,
            )
            if latest_snapshot is not None:
                snapshot = latest_snapshot
                policy_observation_layout_shape = _next_policy_observation_layout_shape(
                    policy_observation_layout_shape,
                    snapshot,
                )
                auxiliary_metrics.observe_snapshot(snapshot)
                latest_live_episode_series = getattr(snapshot, "live_episode_series", None)
                if latest_live_episode_series is not None:
                    live_episode_series = latest_live_episode_series
            elif worker_closed and not worker.process.is_alive():
                return
            elif not worker.process.is_alive():
                raise RuntimeError("watch simulation worker stopped unexpectedly")

            screen = _ensure_screen(
                pygame,
                screen,
                game_display_size,
                policy_observation_layout_shape,
                fonts=fonts,
                info=snapshot.info,
                panel_tab_index=panel_tab_index,
            )
            render_rate.tick()
            hitboxes = draw_watch_frame(
                pygame=pygame,
                screen=screen,
                fonts=fonts,
                config=config,
                snapshot=snapshot,
                paused=paused,
                render_rate=render_rate,
                target_render_fps=target_render_fps,
                auxiliary_episode_metrics=auxiliary_metrics.snapshot(),
                live_episode_series=live_episode_series,
                panel_tabs=panel_tabs,
                panel_tab_index=panel_tab_index,
                cnn_layer_tab_index=cnn_layer_tab_index,
                record_tab_index=record_tab_index,
                policy_observation_layout_shape=policy_observation_layout_shape,
            )
            if recorder is not None:
                recorder.write_surface(pygame, screen)
            _sync_mouse_cursor(pygame, hitboxes)
    except KeyboardInterrupt:
        return
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            worker.shutdown()
            pygame.quit()


def _next_panel_tab_index(
    current_index: int,
    viewer_input: ViewerInput,
    *,
    panel_tabs: PanelTabRegistry = WATCH_PANEL_TABS,
) -> int:
    selected_index = viewer_input.panel_tab_index
    if selected_index is not None:
        return max(0, min(panel_tabs.count - 1, selected_index))
    if viewer_input.panel_tab_delta == 0:
        return current_index
    return panel_tabs.normalize(current_index + viewer_input.panel_tab_delta)


def _sync_mouse_cursor(pygame: PygameModule, hitboxes: ViewerHitboxes) -> None:
    """Use a hand cursor over clickable viewer chrome when pygame supports it."""

    mouse_pos = pygame.mouse.get_pos()
    hover_clickable = mouse_over_clickable(
        mouse_pos,
        deterministic_toggle_rect=hitboxes.deterministic_toggle,
        panel_tab_rects=hitboxes.panel_tabs,
        cnn_layer_tab_rects=hitboxes.cnn_layer_tabs,
        record_tab_rects=hitboxes.record_tabs,
        record_course_hitboxes=hitboxes.record_courses,
        state_feature_hitboxes=hitboxes.state_features,
    )
    cursor = _system_cursor(
        pygame,
        "SYSTEM_CURSOR_HAND" if hover_clickable else "SYSTEM_CURSOR_ARROW",
    )
    if cursor is not None:
        try:
            pygame.mouse.set_cursor(cursor)
        except Exception as exc:
            pygame_error = getattr(pygame, "error", None)
            if not isinstance(pygame_error, type) or not isinstance(exc, pygame_error):
                raise


def _default_policy_observation_layout_shape() -> tuple[int, int, int]:
    return (72, 96, 3)


def _initial_policy_observation_layout_shape(snapshot: object) -> tuple[int, ...]:
    return (
        _snapshot_policy_observation_shape(snapshot) or _default_policy_observation_layout_shape()
    )


def _next_policy_observation_layout_shape(
    current_shape: tuple[int, ...],
    snapshot: object,
) -> tuple[int, ...]:
    return _snapshot_policy_observation_shape(snapshot) or current_shape


def _snapshot_policy_observation_shape(snapshot: object) -> tuple[int, ...] | None:
    value = getattr(snapshot, "policy_observation_shape", None)
    if isinstance(value, tuple) and all(isinstance(item, int) for item in value):
        return value
    return None


def _system_cursor(pygame: PygameModule, name: str) -> object | None:
    return getattr(pygame, name, None)
