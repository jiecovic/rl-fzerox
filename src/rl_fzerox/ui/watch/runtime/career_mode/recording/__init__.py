# src/rl_fzerox/ui/watch/runtime/career_mode/recording/__init__.py
"""Career Mode recording facade.

The facade exposes recorder entrypoints, artifact path helpers, and ffmpeg
operations that tests may monkeypatch. Internal finalizer and summary models
stay in their concrete modules.
"""

from __future__ import annotations

from rl_fzerox.apps.recording.video import (
    concat_mp4_recordings,
    remux_recording_to_mp4,
    resolve_ffmpeg_path,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording.paths import (
    career_live_recording_path,
    career_segment_recording_path,
    career_session_summary_path,
    career_session_video_path,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording.recorder import (
    CareerModeFrameRecorder,
    FrameRecorder,
    RecordingSegmentStatus,
    open_career_mode_recorder,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording.summary import write_segment_summary_files

__all__ = [
    "CareerModeFrameRecorder",
    "FrameRecorder",
    "RecordingSegmentStatus",
    "career_live_recording_path",
    "career_segment_recording_path",
    "career_session_summary_path",
    "career_session_video_path",
    "concat_mp4_recordings",
    "open_career_mode_recorder",
    "remux_recording_to_mp4",
    "resolve_ffmpeg_path",
    "write_segment_summary_files",
]
