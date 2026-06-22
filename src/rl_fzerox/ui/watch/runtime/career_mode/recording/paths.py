# src/rl_fzerox/ui/watch/runtime/career_mode/recording/paths.py
from __future__ import annotations

import re
from pathlib import Path


def career_segment_recording_path(
    path: Path,
    *,
    segment_index: int,
    label: str,
    attempt_id: str | None = None,
    status: str | None = None,
    partial: bool = False,
) -> Path:
    suffix = path.suffix or ".mkv"
    slug = _slug(label)
    if partial:
        slug = f"partial-{slug}"
    if status == "failed":
        slug = f"failed-attempt-{slug}"
    uid = _attempt_uid(attempt_id)
    uid_part = "" if uid is None else f"-{uid}"
    return path.with_name(f"{path.stem}.segment-{segment_index:03d}{uid_part}-{slug}{suffix}")


def career_session_summary_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.session.json")


def career_live_recording_path(path: Path) -> Path:
    suffix = path.suffix or ".mkv"
    return path.with_name(f"{path.stem}.live{suffix}")


def career_session_video_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.session.mp4")


def career_debug_dir(path: Path) -> Path:
    return path.with_name(f"{path.stem}.debug")


def segment_summary_path(video_path: Path, suffix: str) -> Path:
    return video_path.with_name(f"{video_path.stem}.summary{suffix}")


def _slug(label: str) -> str:
    parts = re.findall(r"[a-z0-9]+", label.lower())
    return "-".join(parts) if parts else "career-target"


def _attempt_uid(attempt_id: str | None) -> str | None:
    if attempt_id is None:
        return None
    suffix = attempt_id.rsplit("-", 1)[-1].lower()
    if re.fullmatch(r"[0-9a-f]{8}", suffix) is None:
        return None
    return suffix
