# src/rl_fzerox/ui/watch/runtime/courses/commands.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rl_fzerox.ui.watch.runtime.courses.navigation import (
    WatchCourseRotation,
    sync_watch_rotation_info,
)
from rl_fzerox.ui.watch.runtime.ipc import WorkerCommandBatch


class _CourseResetEnv(Protocol):
    def set_locked_reset_course(self, course_id: str | None) -> None: ...

    def set_next_sequential_reset_course(self, course_id: str | None) -> None: ...


@dataclass(frozen=True, slots=True)
class CourseCommandResult:
    selected_reset_target_key: str | None
    locked_reset_target_key: str | None
    reset_requested: bool
    lock_state_changed: bool


def apply_course_navigation_commands(
    commands: WorkerCommandBatch,
    *,
    env: _CourseResetEnv,
    info: dict[str, object],
    reset_info: dict[str, object],
    rotation: WatchCourseRotation,
    selected_reset_target_key: str | None,
    locked_reset_target_key: str | None,
) -> CourseCommandResult:
    """Apply watch reset/navigation commands against the global course rotation."""

    next_selected_key = rotation.normalized_key(selected_reset_target_key)
    next_locked_key = _valid_locked_key(rotation, locked_reset_target_key)
    lock_state_changed = False

    if commands.jump_course_id is not None:
        target_key = _jump_target_key(
            rotation,
            commands.jump_course_id,
            selected_reset_target_key=next_selected_key,
        )
        if target_key is None:
            return CourseCommandResult(
                selected_reset_target_key=next_selected_key,
                locked_reset_target_key=next_locked_key,
                reset_requested=False,
                lock_state_changed=False,
            )
        next_selected_key = target_key
        next_locked_key = None
        env.set_locked_reset_course(None)
        env.set_next_sequential_reset_course(target_key)
        _sync_info(
            info=info,
            reset_info=reset_info,
            rotation=rotation,
            selected_reset_target_key=next_selected_key,
            locked_reset_target_key=next_locked_key,
        )
        return CourseCommandResult(
            selected_reset_target_key=next_selected_key,
            locked_reset_target_key=next_locked_key,
            reset_requested=True,
            lock_state_changed=False,
        )

    if commands.toggle_current_course_lock:
        current_key = _current_or_selected_key(
            rotation,
            info,
            selected_reset_target_key=next_selected_key,
        )
        if current_key is not None:
            next_locked_key = None if next_locked_key == current_key else current_key
            env.set_locked_reset_course(next_locked_key)
            lock_state_changed = True

    target_key = _requested_target_key(
        commands,
        rotation=rotation,
        info=info,
        selected_reset_target_key=next_selected_key,
        locked_reset_target_key=next_locked_key,
    )
    if target_key is not None:
        next_selected_key = target_key
        if next_locked_key is None:
            env.set_next_sequential_reset_course(target_key)
        else:
            next_locked_key = target_key
            env.set_locked_reset_course(target_key)
        _sync_info(
            info=info,
            reset_info=reset_info,
            rotation=rotation,
            selected_reset_target_key=next_selected_key,
            locked_reset_target_key=next_locked_key,
        )
        return CourseCommandResult(
            selected_reset_target_key=next_selected_key,
            locked_reset_target_key=next_locked_key,
            reset_requested=True,
            lock_state_changed=lock_state_changed,
        )

    if lock_state_changed:
        _sync_info(
            info=info,
            reset_info=reset_info,
            rotation=rotation,
            selected_reset_target_key=next_selected_key,
            locked_reset_target_key=next_locked_key,
        )

    return CourseCommandResult(
        selected_reset_target_key=next_selected_key,
        locked_reset_target_key=next_locked_key,
        reset_requested=False,
        lock_state_changed=lock_state_changed,
    )


def next_watch_reset_after_episode(
    *,
    rotation: WatchCourseRotation,
    info: dict[str, object],
    episode_done: bool,
    selected_reset_target_key: str | None,
    locked_reset_target_key: str | None,
) -> str | None:
    """Return the selected target for the next episode after terminal handling."""

    if locked_reset_target_key is not None:
        return rotation.normalized_key(locked_reset_target_key)
    current_key = _current_or_selected_key(
        rotation,
        info,
        selected_reset_target_key=selected_reset_target_key,
    )
    if current_key is None:
        return rotation.normalized_key(selected_reset_target_key)
    if not episode_done or not _watch_episode_completed_race(info):
        return current_key
    target = rotation.adjacent_target(current_key, offset=1)
    return None if target is None else target.key


def _requested_target_key(
    commands: WorkerCommandBatch,
    *,
    rotation: WatchCourseRotation,
    info: dict[str, object],
    selected_reset_target_key: str | None,
    locked_reset_target_key: str | None,
) -> str | None:
    current_key = locked_reset_target_key or _current_or_selected_key(
        rotation,
        info,
        selected_reset_target_key=selected_reset_target_key,
    )
    if commands.reset_mode == "current":
        return current_key
    if commands.reset_mode in {"previous", "next"}:
        offset = -1 if commands.reset_mode == "previous" else 1
        target = rotation.adjacent_target(current_key, offset=offset)
        return None if target is None else target.key
    if commands.difficulty_delta != 0:
        target = rotation.difficulty_target(current_key, offset=commands.difficulty_delta)
        return None if target is None else target.key
    return None


def _jump_target_key(
    rotation: WatchCourseRotation,
    requested_key: str,
    *,
    selected_reset_target_key: str | None,
) -> str | None:
    target = rotation.target_by_key(requested_key)
    if target is not None:
        return target.key
    selected = rotation.target_by_key(selected_reset_target_key)
    if selected is not None:
        target = rotation.target_for_course_difficulty(requested_key, selected.difficulty)
        if target is not None:
            return target.key
    return None


def _current_or_selected_key(
    rotation: WatchCourseRotation,
    info: dict[str, object],
    *,
    selected_reset_target_key: str | None,
) -> str | None:
    current = rotation.target_for_info(info)
    if current is not None:
        return current.key
    return rotation.normalized_key(selected_reset_target_key)


def _valid_locked_key(
    rotation: WatchCourseRotation,
    locked_reset_target_key: str | None,
) -> str | None:
    if rotation.target_by_key(locked_reset_target_key) is None:
        return None
    return locked_reset_target_key


def _sync_info(
    *,
    info: dict[str, object],
    reset_info: dict[str, object],
    rotation: WatchCourseRotation,
    selected_reset_target_key: str | None,
    locked_reset_target_key: str | None,
) -> None:
    sync_watch_rotation_info(
        info=info,
        reset_info=reset_info,
        rotation=rotation,
        selected_reset_target_key=selected_reset_target_key,
        locked_reset_target_key=locked_reset_target_key,
    )


def _watch_episode_completed_race(info: dict[str, object]) -> bool:
    if info.get("termination_reason") != "finished":
        return False
    laps_completed = _int_info_value(info, "race_laps_completed")
    if laps_completed is None:
        laps_completed = _int_info_value(info, "laps_completed")
    total_laps = _int_info_value(info, "total_lap_count")
    if laps_completed is None or total_laps is None:
        return False
    return total_laps > 0 and laps_completed >= total_laps


def _int_info_value(info: dict[str, object], key: str) -> int | None:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)
