# src/rl_fzerox/ui/watch/app.py
from __future__ import annotations

from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.ui.watch.input import _poll_viewer_input
from rl_fzerox.ui.watch.runtime import (
    apply_viewer_input,
    drain_snapshot_queue,
    start_watch_worker,
    wait_initial_snapshot,
)
from rl_fzerox.ui.watch.runtime.timing import (
    RateMeter,
    _resolve_render_fps,
)
from rl_fzerox.ui.watch.view.screen.frame import (
    _create_fonts,
    _ensure_screen,
    _watch_game_display_size,
)
from rl_fzerox.ui.watch.view.screen.render import draw_watch_frame
from rl_fzerox.ui.watch.view.screen.types import ViewerHitboxes

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
        hitboxes = ViewerHitboxes()

        while True:
            render_limit = 0 if target_render_fps is None else max(1, int(target_render_fps))
            render_clock.tick(render_limit)

            viewer_input = _poll_viewer_input(
                pygame,
                deterministic_toggle_rect=hitboxes.deterministic_toggle,
            )
            paused = apply_viewer_input(
                worker.command_queue,
                viewer_input,
                paused=paused,
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
            )
    except KeyboardInterrupt:
        return
    finally:
        worker.shutdown()
        pygame.quit()
