# src/rl_fzerox/ui/watch/runtime/live/reset.py
from __future__ import annotations

from typing import Protocol

from rl_fzerox.ui.watch.runtime.courses.commands import next_watch_reset_after_episode
from rl_fzerox.ui.watch.runtime.courses.navigation import WatchCourseRotation


class _SequentialResetEnv(Protocol):
    def set_next_sequential_reset_course(self, course_id: str | None) -> None: ...


def _sync_next_watch_reset_after_episode(
    *,
    env: _SequentialResetEnv,
    rotation: WatchCourseRotation,
    info: dict[str, object],
    episode_done: bool,
    selected_reset_target_key: str | None,
    locked_reset_target_key: str | None,
) -> str | None:
    if not episode_done:
        return selected_reset_target_key
    next_target_key = next_watch_reset_after_episode(
        rotation=rotation,
        info=info,
        episode_done=episode_done,
        selected_reset_target_key=selected_reset_target_key,
        locked_reset_target_key=locked_reset_target_key,
    )
    if locked_reset_target_key is None and next_target_key is not None:
        env.set_next_sequential_reset_course(next_target_key)
    return next_target_key
