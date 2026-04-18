# src/rl_fzerox/apps/recording/progress.py
from __future__ import annotations

import time


class ProgressPrinter:
    """Render one compact live status line while an attempt is recording."""

    def __init__(
        self,
        *,
        interval_seconds: float,
        attempt: int,
        target_rank: int,
    ) -> None:
        self._interval_seconds = interval_seconds
        self._attempt = attempt
        self._target_rank = target_rank
        self._started_at = time.monotonic()
        self._next_print_time = 0.0
        self._printed = False

    def print(
        self,
        info: dict[str, object],
        *,
        episode_return: float,
        force: bool = False,
    ) -> None:
        if self._interval_seconds <= 0.0:
            return
        now = time.monotonic()
        if not force and now < self._next_print_time:
            return
        self._next_print_time = now + self._interval_seconds
        self._printed = True
        line = format_progress_line(
            info,
            attempt=self._attempt,
            target_rank=self._target_rank,
            episode_return=episode_return,
            effective_fps=effective_fps(info, started_at=self._started_at, now=now),
        )
        print(
            f"\r\x1b[2K{line}",
            end="",
            flush=True,
        )

    def finish(self) -> None:
        if self._printed:
            print()


def format_progress_line(
    info: dict[str, object],
    *,
    attempt: int,
    target_rank: int,
    episode_return: float,
    effective_fps: float,
) -> str:
    return (
        f"try {attempt:02d} | "
        f"step {int_info(info, 'episode_step')} | "
        f"rank {int_info(info, 'position')} | "
        f"lap {int_info(info, 'lap')} | "
        f"time {format_race_time_ms(int_info(info, 'race_time_ms'))} | "
        f"{float_info(info, 'speed_kph'):.0f} km/h | "
        f"{format_compact_number(float_info(info, 'race_distance'))} prog | "
        f"{effective_fps:.1f} frames/s | "
        f"R {episode_return:.1f} | "
        f"{format_target_rank(target_rank)}"
    )


def int_info(info: dict[str, object], key: str) -> int:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def float_info(info: dict[str, object], key: str) -> float:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def format_compact_number(value: float) -> str:
    absolute_value = abs(value)
    if absolute_value >= 1_000_000.0:
        return f"{value / 1_000_000.0:.2f}M"
    if absolute_value >= 10_000.0:
        return f"{value / 1_000.0:.1f}k"
    if absolute_value >= 1_000.0:
        return f"{value / 1_000.0:.2f}k"
    return f"{value:.0f}"


def format_target_rank(target_rank: int) -> str:
    if target_rank == 1:
        return "need rank 1"
    return f"need rank <= {target_rank}"


def format_race_time_ms(milliseconds: int) -> str:
    if milliseconds <= 0:
        return "--:--.---"
    minutes, remaining_ms = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remaining_ms, 1_000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def effective_fps(
    info: dict[str, object],
    *,
    started_at: float,
    now: float,
) -> float:
    elapsed = max(now - started_at, 1e-9)
    return float(int_info(info, "episode_step")) / elapsed


def optional_str_info(info: dict[str, object], key: str) -> str | None:
    value = info.get(key)
    if not isinstance(value, str):
        return None
    return value
