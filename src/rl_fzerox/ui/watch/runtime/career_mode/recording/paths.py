# src/rl_fzerox/ui/watch/runtime/career_mode/recording/paths.py
from __future__ import annotations

import re
from pathlib import Path


def career_segment_recording_path(
    path: Path,
    *,
    segment_index: int,
    label: str,
    status: str | None = None,
) -> Path:
    suffix = path.suffix or ".mkv"
    slug = _slug(label)
    if status == "failed":
        slug = f"failed-attempt-{slug}"
    return path.with_name(f"{path.stem}.segment-{segment_index:03d}-{slug}{suffix}")


def career_session_summary_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.session.json")


def career_live_recording_path(path: Path) -> Path:
    suffix = path.suffix or ".mkv"
    return path.with_name(f"{path.stem}.live{suffix}")


def career_session_video_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.session.mp4")


def segment_summary_path(video_path: Path, suffix: str) -> Path:
    return video_path.with_name(f"{video_path.stem}.summary{suffix}")


def _slug(label: str) -> str:
    parts = re.findall(r"[a-z0-9]+", label.lower())
    return "-".join(parts) if parts else "career-target"
