# src/rl_fzerox/ui/watch/runtime/career_mode/loop/debug.py
from __future__ import annotations

import json
import os
import re
import struct
import time
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.career_mode.controller import CareerModeController
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.runtime.career_mode.recording.paths import career_debug_dir

_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_DEBUG_ENV = "RL_FZEROX_CAREER_DEBUG"

_TRACE_FIELDS = (
    "career_mode_attempt_id",
    "career_mode_target_label",
    "career_mode_phase",
    "career_mode_policy_active",
    "career_mode_last_input",
    "career_mode_last_step",
    "career_mode_last_step_frames",
    "career_mode_last_finished_attempt_id",
    "career_mode_last_finished_attempt_status",
    "career_mode_last_finished_attempt_failure_reason",
    "career_mode_post_gp_cutscene_complete",
    "career_mode_fsm_observed_screen",
    "career_mode_fsm_terminal_result",
    "career_mode_fsm_terminal_reason",
    "career_mode_fsm_fresh_race_ready",
    "career_mode_fsm_fresh_race_ready_frames",
    "career_mode_fsm_continuing_result",
    "career_mode_fsm_awaiting_fresh_race",
    "career_mode_fsm_pending_steps",
    "career_mode_fsm_game_mode",
    "career_mode_fsm_course_index",
    "career_mode_fsm_completed_laps",
    "career_mode_fsm_total_laps",
    "career_mode_fsm_completion_fraction",
    "game_mode",
    "game_mode_raw",
    "queued_game_mode",
    "queued_game_mode_raw",
    "menu_selected_mode_raw",
    "menu_difficulty_state_raw",
    "menu_difficulty_cursor_raw",
    "menu_transition_state_raw",
    "difficulty",
    "difficulty_raw",
    "course_index",
    "track_id",
    "track_course_name",
    "race_intro_timer",
    "race_time_ms",
    "race_laps_completed",
    "total_lap_count",
    "episode_completion_fraction",
    "position",
    "career_mode_gp_final_rank",
    "gp_final_rank",
    "termination_reason",
    "finished",
    "retired",
    "crashed",
    "entered_finished",
    "entered_retired",
    "entered_crashed",
)

_CHANGE_FIELDS = (
    "career_mode_attempt_id",
    "career_mode_last_finished_attempt_id",
    "career_mode_last_finished_attempt_status",
    "career_mode_post_gp_cutscene_complete",
    "career_mode_fsm_observed_screen",
    "career_mode_fsm_terminal_result",
    "career_mode_fsm_terminal_reason",
    "career_mode_fsm_fresh_race_ready",
    "career_mode_fsm_continuing_result",
    "career_mode_fsm_awaiting_fresh_race",
    "game_mode",
    "course_index",
    "race_laps_completed",
    "total_lap_count",
    "career_mode_gp_final_rank",
    "gp_final_rank",
    "termination_reason",
    "finished",
    "retired",
    "crashed",
    "entered_finished",
    "entered_retired",
    "entered_crashed",
)


class CareerDebugFrameSource(Protocol):
    def __call__(self) -> RgbFrame: ...


@dataclass(slots=True)
class CareerModeDebugTrace:
    """Write sparse controller/native-state transitions for real-run debugging.

    The Career Mode FSM owns lifecycle decisions, but the emulator screen is the
    authority for whether those decisions match reality. This trace records both
    sides at each meaningful state change: native RAM-backed facts such as
    `game_mode`, `gp_final_rank`, and terminal flags, plus controller-derived
    phase/progress fields and a PNG of the rendered screen.
    """

    directory: Path
    screenshots: bool = True
    screenshot_limit: int = 256
    trace_path: Path = field(init=False)
    frame_dir: Path = field(init=False)
    _event_index: int = field(default=0, init=False)
    _screenshot_count: int = field(default=0, init=False)
    _last_change_key: tuple[object, ...] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.trace_path = self.directory / "trace.jsonl"
        self.frame_dir = self.directory / "frames"
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.screenshots:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    def observe(
        self,
        *,
        stage: str,
        info: dict[str, object],
        controller: CareerModeController,
        frame_source: CareerDebugFrameSource | None = None,
        event: str | None = None,
        force: bool = False,
    ) -> None:
        values = _trace_values(info)
        change_key = _change_key(values)
        if not force and change_key == self._last_change_key:
            return

        self._event_index += 1
        screenshot = self._write_screenshot(
            stage=stage,
            info=info,
            frame_source=frame_source,
        )
        entry = {
            "index": self._event_index,
            "time_utc": datetime.now(UTC).isoformat(),
            "monotonic_s": round(time.perf_counter(), 6),
            "stage": stage,
            "event": event,
            "screenshot": screenshot,
            "controller_context": controller.debug_context(info),
            "values": values,
        }
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        self._last_change_key = change_key

    def _write_screenshot(
        self,
        *,
        stage: str,
        info: dict[str, object],
        frame_source: CareerDebugFrameSource | None,
    ) -> str | None:
        if (
            not self.screenshots
            or frame_source is None
            or self._screenshot_count >= self.screenshot_limit
        ):
            return None
        frame = frame_source()
        filename = _screenshot_filename(
            index=self._event_index,
            stage=stage,
            game_mode=info.get("game_mode"),
        )
        path = self.frame_dir / filename
        _write_rgb_png(path, frame)
        self._screenshot_count += 1
        return str(path.relative_to(self.directory))


def open_career_mode_debug_trace(config: WatchAppConfig) -> CareerModeDebugTrace | None:
    if not _debug_enabled(config):
        return None
    recording_path = config.watch.recording.path
    if recording_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        directory = Path("local") / "recordings" / "career-debug" / timestamp
    else:
        directory = career_debug_dir(recording_path)
    return CareerModeDebugTrace(
        directory=directory,
        screenshots=(
            config.watch.career_debug.screenshots and config.watch.career_debug.screenshot_limit > 0
        ),
        screenshot_limit=config.watch.career_debug.screenshot_limit,
    )


def observe_career_mode_debug_trace(
    trace: CareerModeDebugTrace | None,
    *,
    stage: str,
    info: dict[str, object],
    controller: CareerModeController,
    frame_source: CareerDebugFrameSource | None = None,
    event: str | None = None,
    force: bool = False,
) -> None:
    if trace is None:
        return
    try:
        trace.observe(
            stage=stage,
            info=info,
            controller=controller,
            frame_source=frame_source,
            event=event,
            force=force,
        )
    except Exception as exc:
        print(f"Career debug trace failed: {exc}", flush=True)


def _debug_enabled(config: WatchAppConfig) -> bool:
    if config.watch.career_debug.enabled:
        return True
    return os.environ.get(_DEBUG_ENV, "").strip().lower() in _TRUE_ENV_VALUES


def _trace_values(info: dict[str, object]) -> dict[str, object]:
    values: dict[str, object] = {}
    for field_name in _TRACE_FIELDS:
        values[field_name] = _json_safe(info.get(field_name))
    return values


def _change_key(values: dict[str, object]) -> tuple[object, ...]:
    return tuple(values.get(field_name) for field_name in _CHANGE_FIELDS)


def _json_safe(value: object) -> object:
    if isinstance(value, np.bool_):
        return _numpy_bool_to_python(value)
    if isinstance(value, np.integer):
        return _numpy_integer_to_python(value)
    if isinstance(value, np.floating):
        return _numpy_float_to_python(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return str(value)


def _numpy_bool_to_python(value: np.bool_) -> bool:
    return bool(value)


def _numpy_float_to_python(value: np.floating[Any]) -> float:
    return float(value)


def _numpy_integer_to_python(value: np.integer[Any]) -> int:
    return int(value)


def _screenshot_filename(
    *,
    index: int,
    stage: str,
    game_mode: object,
) -> str:
    stage_slug = _slug(stage)
    mode_slug = _slug(str(game_mode)) if game_mode is not None else "unknown"
    return f"{index:05d}-{stage_slug}-{mode_slug}.png"


def _slug(value: str) -> str:
    parts = re.findall(r"[a-z0-9]+", value.lower())
    return "-".join(parts) if parts else "event"


def _write_rgb_png(path: Path, frame: RgbFrame) -> None:
    frame_array = np.asarray(frame, dtype=np.uint8)
    if frame_array.ndim != 3 or frame_array.shape[2] != 3:
        raise ValueError(f"expected RGB frame with shape HxWx3, got {frame_array.shape!r}")

    height, width, _ = frame_array.shape
    rows = [b"\x00" + frame_array[row].tobytes() for row in range(height)]
    compressed = zlib.compress(b"".join(rows))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum)
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum & 0xFFFFFFFF)
