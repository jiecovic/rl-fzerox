# src/rl_fzerox/ui/watch/runtime/career_mode/recording/summary/writer.py
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rl_fzerox.ui.watch.runtime.career_mode.recording.paths import (
    career_session_summary_path,
    segment_summary_path,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording.summary.models import (
    _SegmentSummarySnapshot,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording.summary.payloads import (
    _segment_payload_sort_key,
    _segment_summary_markdown,
    _segment_summary_payload,
    _session_summary_payload,
)


@dataclass(slots=True)
class _SessionSummaryWriter:
    source_path: Path
    live_video_path: Path | None = None
    session_video_path: Path | None = None
    segment_payloads: list[dict[str, object]] = field(default_factory=list)

    def record_live_video(self, video_path: Path) -> None:
        self.live_video_path = video_path
        self.write()

    def record_session_video(self, video_path: Path) -> None:
        self.session_video_path = video_path
        self.write()

    def record_segment(self, summary: _SegmentSummarySnapshot, *, video_path: Path) -> None:
        payload = _segment_summary_payload(summary, video_path=video_path)
        self.segment_payloads = [
            segment
            for segment in self.segment_payloads
            if segment.get("segment_index") != summary.segment_index
        ]
        self.segment_payloads.append(payload)
        self.segment_payloads.sort(key=_segment_payload_sort_key)
        self.write()

    def segment_video_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        for payload in self.segment_payloads:
            video = payload.get("video")
            if not isinstance(video, Mapping):
                continue
            path = video.get("mp4_path")
            if isinstance(path, str) and path:
                paths.append(Path(path))
        return tuple(paths)

    def write(self) -> None:
        career_session_summary_path(self.source_path).write_text(
            json.dumps(_session_summary_payload(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def write_segment_summary_files(
    summary: _SegmentSummarySnapshot,
    *,
    video_path: Path,
) -> None:
    """Write machine-readable and human-readable sidecars for one finalized segment."""

    payload = _segment_summary_payload(summary, video_path=video_path)
    json_path = segment_summary_path(video_path, ".json")
    markdown_path = segment_summary_path(video_path, ".md")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _segment_summary_markdown(payload),
        encoding="utf-8",
    )
