# src/rl_fzerox/ui/watch/recording.py
from __future__ import annotations

import numpy as np

from rl_fzerox.apps.recording.video import (
    FfmpegRgbWriter,
    as_rgb_frame,
    resolve_ffmpeg_path,
    resolve_video_fps,
)
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.view.screen.types import PygameModule, PygameSurface


class ViewerRecorder:
    """Record the drawn watch window into one H.264 video file."""

    def __init__(
        self,
        *,
        config: WatchAppConfig,
        native_fps: float,
        render_fps: float | None,
    ) -> None:
        recording = config.watch.recording
        if recording.path is None:
            raise ValueError("watch.recording.path is required when recording is enabled")
        self._writer = FfmpegRgbWriter(
            path=recording.path.expanduser(),
            ffmpeg_path=resolve_ffmpeg_path(),
            fps=resolve_video_fps(native_fps=native_fps, override=render_fps),
        )
        self._target_size: tuple[int, int] | None = None
        self._writer.__enter__()

    def write_surface(self, pygame: PygameModule, surface: PygameSurface) -> None:
        if self._target_size is None:
            self._target_size = _surface_size(surface)
        if _surface_size(surface) != self._target_size:
            surface = pygame.transform.smoothscale(surface, self._target_size)
        # pygame returns WxHx3; ffmpeg expects contiguous HxWx3 RGB bytes.
        frame = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        self._writer.write(as_rgb_frame(frame))

    def close(self) -> None:
        self._writer.close()


def open_viewer_recorder(
    *,
    config: WatchAppConfig,
    native_fps: float,
    render_fps: float | None,
) -> ViewerRecorder | None:
    if not config.watch.recording.enabled:
        return None
    if config.watch.managed_save_game_id is not None:
        return None
    return ViewerRecorder(config=config, native_fps=native_fps, render_fps=render_fps)


def _surface_size(surface: PygameSurface) -> tuple[int, int]:
    width, height = surface.get_size()
    return int(width), int(height)
