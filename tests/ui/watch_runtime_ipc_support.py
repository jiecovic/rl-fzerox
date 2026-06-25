# tests/ui/watch_runtime_ipc_support.py
"""Shared test doubles for Watch runtime IPC tests.

The doubles model command queues, worker shutdown queues, process lifecycle, and
sequential reset hooks without starting a real Watch worker process.
"""

from __future__ import annotations

from queue import Empty

from rl_fzerox.core.runtime_spec.schema import TrackSamplingEntryConfig
from rl_fzerox.ui.watch.runtime.courses.navigation import WatchCourseRotation


class _CommandQueue:
    def __init__(self, commands: list[object]) -> None:
        self._commands = commands

    def get_nowait(self) -> object:
        if not self._commands:
            raise Empty
        return self._commands.pop(0)


class _ShutdownQueue:
    def __init__(self) -> None:
        self.items: list[object] = []
        self.cancelled = False
        self.closed = False

    def put(self, obj: object) -> None:
        self.items.append(obj)

    def cancel_join_thread(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True


class _InterruptingProcess:
    def __init__(self) -> None:
        self._alive = True
        self.join_calls: list[float | None] = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        if len(self.join_calls) == 1:
            raise KeyboardInterrupt

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False


class _SequentialResetEnv:
    def __init__(self) -> None:
        self.next_courses: list[str | None] = []
        self.locked_courses: list[str | None] = []

    def set_locked_reset_course(self, course_id: str | None) -> None:
        self.locked_courses.append(course_id)

    def set_next_sequential_reset_course(self, course_id: str | None) -> None:
        self.next_courses.append(course_id)


def _sample_rotation() -> WatchCourseRotation:
    return WatchCourseRotation.from_entries(
        (
            TrackSamplingEntryConfig(
                id="mute_novice",
                course_id="mute_city",
                gp_difficulty="novice",
            ),
            TrackSamplingEntryConfig(
                id="silence_novice",
                course_id="silence",
                gp_difficulty="novice",
            ),
            TrackSamplingEntryConfig(
                id="mute_expert",
                course_id="mute_city",
                gp_difficulty="expert",
            ),
            TrackSamplingEntryConfig(
                id="silence_expert",
                course_id="silence",
                gp_difficulty="expert",
            ),
        )
    )
