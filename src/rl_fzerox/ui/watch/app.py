# src/rl_fzerox/ui/watch/app.py
from __future__ import annotations

from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.ui.watch.input import (
    SpeedKeyRepeat,
    ViewerInput,
    _poll_viewer_input,
    mouse_over_clickable,
)
from rl_fzerox.ui.watch.runtime import (
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
from rl_fzerox.ui.watch.view.panels.tabs import PANEL_TABS
from rl_fzerox.ui.watch.view.screen.frame import (
    _create_fonts,
    _ensure_screen,
    _watch_game_display_size,
)
from rl_fzerox.ui.watch.view.screen.render import draw_watch_frame
from rl_fzerox.ui.watch.view.screen.types import PygameModule, ViewerHitboxes

__all__ = ["run_viewer"]


def run_viewer(config: WatchAppConfig) -> None:
    """Run the watch UI while a worker process advances emulator state."""

    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "pygame is required for watching emulator output. "
            "Install with `pip install -e .[watch]`."
        ) from exc

    worker = start_watch_worker(config)
    pygame.init()
    render_clock = pygame.time.Clock()
    try:
        snapshot, worker_closed = wait_initial_snapshot(worker)
        target_render_fps = _resolve_render_fps(
            config.watch.render_fps,
            native_fps=snapshot.native_fps,
        )
        render_rate = RateMeter(window=60)
        game_display_size = _watch_game_display_size()
        screen = None
        fonts = _create_fonts(pygame)
        paused = False
        panel_tab_index = 0
        record_tab_index = 0
        cnn_normalization = DEFAULT_CNN_ACTIVATION_NORMALIZATION
        hitboxes = ViewerHitboxes()
        speed_repeat = SpeedKeyRepeat()

        while True:
            render_limit = 0 if target_render_fps is None else max(1, int(target_render_fps))
            render_clock.tick(render_limit)

            viewer_input = _poll_viewer_input(
                pygame,
                deterministic_toggle_rect=hitboxes.deterministic_toggle,
                panel_tab_rects=hitboxes.panel_tabs,
                record_tab_rects=hitboxes.record_tabs,
                record_course_hitboxes=hitboxes.record_courses,
                speed_repeat=speed_repeat,
            )
            panel_tab_index = _next_panel_tab_index(panel_tab_index, viewer_input)
            if viewer_input.record_tab_index is not None:
                record_tab_index = viewer_input.record_tab_index
            if viewer_input.toggle_cnn_normalization:
                cnn_normalization = next_cnn_activation_normalization(cnn_normalization)
            paused = apply_viewer_input(
                worker.command_queue,
                viewer_input,
                paused=paused,
                cnn_visualization_enabled=panel_tab_index == PANEL_TABS.cnn_index,
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
            elif worker_closed and not worker.process.is_alive():
                return
            elif not worker.process.is_alive():
                raise RuntimeError("watch simulation worker stopped unexpectedly")

            screen = _ensure_screen(
                pygame,
                screen,
                game_display_size,
                snapshot.observation_image.shape,
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
                panel_tab_index=panel_tab_index,
                record_tab_index=record_tab_index,
            )
            _sync_mouse_cursor(pygame, hitboxes)
    except KeyboardInterrupt:
        return
    finally:
        worker.shutdown()
        pygame.quit()


def _next_panel_tab_index(current_index: int, viewer_input: ViewerInput) -> int:
    selected_index = viewer_input.panel_tab_index
    if selected_index is not None:
        return max(0, min(PANEL_TABS.count - 1, selected_index))
    if viewer_input.panel_tab_delta == 0:
        return current_index
    return PANEL_TABS.normalize(current_index + viewer_input.panel_tab_delta)


def _sync_mouse_cursor(pygame: PygameModule, hitboxes: ViewerHitboxes) -> None:
    """Use a hand cursor over clickable viewer chrome when pygame supports it."""

    mouse_pos = pygame.mouse.get_pos()
    hover_clickable = mouse_over_clickable(
        mouse_pos,
        deterministic_toggle_rect=hitboxes.deterministic_toggle,
        panel_tab_rects=hitboxes.panel_tabs,
        record_tab_rects=hitboxes.record_tabs,
        record_course_hitboxes=hitboxes.record_courses,
    )
    cursor = _system_cursor(
        pygame,
        "SYSTEM_CURSOR_HAND" if hover_clickable else "SYSTEM_CURSOR_ARROW",
    )
    if cursor is not None:
        pygame.mouse.set_cursor(cursor)


def _system_cursor(pygame: PygameModule, name: str) -> object | None:
    return getattr(pygame, name, None)
