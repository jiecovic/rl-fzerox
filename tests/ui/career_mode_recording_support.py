# tests/ui/career_mode_recording_support.py
"""Shared fakes for Career Mode recording tests.

The helpers keep writer/finalizer behavior deterministic while still exercising
file creation, audio forwarding, MP4 notice draining, and summary-file writes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.ui.watch.runtime.career_mode.recording import (
    _SegmentSummarySnapshot,
    write_segment_summary_files,
)


class _FakeWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.frames: list[RgbFrame] = []
        self.audio_samples: list[list[int]] = []
        self.closed = False

    def __enter__(self) -> _FakeWriter:
        return self

    def write(self, frame: RgbFrame) -> None:
        self.frames.append(np.array(frame, copy=True))

    def write_audio(self, samples: object) -> None:
        self.audio_samples.append(np.asarray(samples, dtype=np.int16).tolist())

    def close(self) -> None:
        self.path.write_bytes(b"video")
        self.closed = True


class _FakeFinalizer:
    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.summaries: list[_SegmentSummarySnapshot | None] = []
        self.notices: list[str] = []
        self.closed = False

    def finalize(self, path: Path, *, summary: _SegmentSummarySnapshot | None = None) -> None:
        self.paths.append(path)
        self.summaries.append(summary)
        if summary is not None:
            write_segment_summary_files(summary, video_path=path.with_suffix(".mp4"))
        self.notices.append(f"MP4 ready: {path.with_suffix('.mp4').name}")

    def drain_notices(self) -> tuple[str, ...]:
        notices = tuple(self.notices)
        self.notices.clear()
        return notices

    def close(self) -> None:
        self.closed = True
