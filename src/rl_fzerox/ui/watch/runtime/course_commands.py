# src/rl_fzerox/ui/watch/runtime/course_commands.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rl_fzerox.ui.watch.runtime.course_navigation import (
    adjacent_watch_course_id,
    current_watch_course_id,
    sync_locked_course_info,
)
from rl_fzerox.ui.watch.runtime.ipc import WorkerCommandBatch


class _CourseResetEnv(Protocol):
    def set_locked_reset_course(self, course_id: str | None) -> None: ...

    def set_next_sequential_reset_course(self, course_id: str | None) -> None: ...


@dataclass(frozen=True, slots=True)
class CourseCommandResult:
    locked_reset_course_id: str | None
    reset_requested: bool
    lock_state_changed: bool


def apply_course_navigation_commands(
    commands: WorkerCommandBatch,
    *,
    env: _CourseResetEnv,
    info: dict[str, object],
    reset_info: dict[str, object],
    locked_reset_course_id: str | None,
    sequential_course_ids: tuple[str, ...],
) -> CourseCommandResult:
    """Apply reset/course-lock commands that take effect on the next reset."""

    next_locked_course_id = locked_reset_course_id
    lock_state_changed = False

    if commands.jump_course_id is not None:
        next_locked_course_id = None
        env.set_locked_reset_course(None)
        env.set_next_sequential_reset_course(commands.jump_course_id)
        sync_locked_course_info(
            info=info,
            reset_info=reset_info,
            locked_reset_course_id=next_locked_course_id,
        )
        return CourseCommandResult(
            locked_reset_course_id=next_locked_course_id,
            reset_requested=True,
            lock_state_changed=False,
        )

    if commands.toggle_current_course_lock:
        next_locked_course_id, lock_state_changed = _toggle_current_course_lock(
            env=env,
            info=info,
            reset_info=reset_info,
            locked_reset_course_id=next_locked_course_id,
            sequential_course_ids=sequential_course_ids,
        )

    if commands.reset_mode == "current":
        current_course_id = current_watch_course_id(info)
        if current_course_id is not None and next_locked_course_id is None:
            env.set_next_sequential_reset_course(current_course_id)
        return CourseCommandResult(
            locked_reset_course_id=next_locked_course_id,
            reset_requested=True,
            lock_state_changed=lock_state_changed,
        )

    if commands.reset_mode in {"previous", "next"}:
        next_locked_course_id = _apply_adjacent_course_reset(
            env=env,
            info=info,
            reset_info=reset_info,
            locked_reset_course_id=next_locked_course_id,
            reset_mode=commands.reset_mode,
            sequential_course_ids=sequential_course_ids,
        )
        return CourseCommandResult(
            locked_reset_course_id=next_locked_course_id,
            reset_requested=True,
            lock_state_changed=lock_state_changed,
        )

    return CourseCommandResult(
        locked_reset_course_id=next_locked_course_id,
        reset_requested=False,
        lock_state_changed=lock_state_changed,
    )


def _toggle_current_course_lock(
    *,
    env: _CourseResetEnv,
    info: dict[str, object],
    reset_info: dict[str, object],
    locked_reset_course_id: str | None,
    sequential_course_ids: tuple[str, ...],
) -> tuple[str | None, bool]:
    current_course_id = current_watch_course_id(info)
    if current_course_id is None:
        return locked_reset_course_id, False

    if locked_reset_course_id == current_course_id:
        next_locked_course_id = None
        env.set_locked_reset_course(None)
    else:
        next_locked_course_id = current_course_id
        env.set_locked_reset_course(current_course_id)

    next_course_id = adjacent_watch_course_id(
        current_course_id=current_course_id,
        ordered_course_ids=sequential_course_ids,
        offset=1,
    )
    env.set_next_sequential_reset_course(next_course_id)
    sync_locked_course_info(
        info=info,
        reset_info=reset_info,
        locked_reset_course_id=next_locked_course_id,
    )
    return next_locked_course_id, True


def _apply_adjacent_course_reset(
    *,
    env: _CourseResetEnv,
    info: dict[str, object],
    reset_info: dict[str, object],
    locked_reset_course_id: str | None,
    reset_mode: str,
    sequential_course_ids: tuple[str, ...],
) -> str | None:
    base_course_id = locked_reset_course_id or current_watch_course_id(info)
    offset = -1 if reset_mode == "previous" else 1
    target_course_id = adjacent_watch_course_id(
        current_course_id=base_course_id,
        ordered_course_ids=sequential_course_ids,
        offset=offset,
    )
    if target_course_id is None:
        return locked_reset_course_id

    if locked_reset_course_id is None:
        env.set_next_sequential_reset_course(target_course_id)
        return locked_reset_course_id

    env.set_locked_reset_course(target_course_id)
    next_course_id = adjacent_watch_course_id(
        current_course_id=target_course_id,
        ordered_course_ids=sequential_course_ids,
        offset=1,
    )
    env.set_next_sequential_reset_course(next_course_id)
    sync_locked_course_info(
        info=info,
        reset_info=reset_info,
        locked_reset_course_id=target_course_id,
    )
    return target_course_id
