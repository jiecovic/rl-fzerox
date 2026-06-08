# src/rl_fzerox/core/runtime_spec/track_sampling_identity.py
from __future__ import annotations

import re


def track_sampling_entry_id(
    *,
    course_id: str | None,
    runtime_course_key: str | None,
    mode: str | None,
    gp_difficulty: str | None,
    vehicle: str | None,
) -> str:
    """Return a unique materialized reset-candidate id."""

    parts = [course_id or runtime_course_key or "track"]
    if mode:
        parts.append(mode)
    if gp_difficulty:
        parts.append(gp_difficulty)
    if vehicle:
        parts.append(vehicle)
    return "_".join(_slug(part) for part in parts)


def track_sampling_course_key(
    *,
    entry_id: str,
    course_id: str | None,
    runtime_course_key: str | None,
    course_ref: str | None = None,
    course_index: int | None = None,
) -> str:
    """Return the course/slot identity used for sampling statistics."""

    if runtime_course_key:
        return runtime_course_key
    if course_id:
        return course_id
    if course_ref:
        return course_ref
    if course_index is not None:
        return f"course:{int(course_index)}"
    return entry_id


def track_sampling_reset_target_key(
    *,
    entry_id: str,
    course_id: str | None,
    runtime_course_key: str | None,
    gp_difficulty: str | None,
    course_ref: str | None = None,
    course_index: int | None = None,
) -> str:
    """Return the reset target id for one course/slot and optional difficulty."""

    course_key = track_sampling_course_key(
        entry_id=entry_id,
        course_id=course_id,
        runtime_course_key=runtime_course_key,
        course_ref=course_ref,
        course_index=course_index,
    )
    if gp_difficulty is None:
        return course_key
    return f"{course_key}#difficulty={gp_difficulty}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "track"
