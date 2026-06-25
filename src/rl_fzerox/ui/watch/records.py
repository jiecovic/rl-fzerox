# src/rl_fzerox/ui/watch/records.py
"""Watch-local per-track record and attempt-stat tracking.

Runtime workers update `TrackRecordBook` from episode info and live telemetry.
The module also owns record identity, including generated X-Cup keys and legacy
lookup keys used by older watch snapshots.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.courses import X_CUP_COURSE
from rl_fzerox.core.domain.engine import (
    ENGINE_SLIDER,
    engine_value_to_slider_step,
)
from rl_fzerox.core.domain.race import is_race_difficulty_name
from rl_fzerox.core.runtime_spec.vehicle_catalog import vehicle_by_character_index

TrackRecordKey = str
TrackFinishSetup = dict[str, str | int]


def track_record_key(info: Mapping[str, object]) -> TrackRecordKey | None:
    """Return the watch-local record key for one course/difficulty attempt."""

    base_key = base_track_record_key(info)
    if base_key is None:
        return None
    difficulty = record_difficulty(info)
    if difficulty is None:
        return base_key
    return f"{base_key}#difficulty={difficulty}"


def track_record_lookup_keys(info: Mapping[str, object]) -> tuple[TrackRecordKey, ...]:
    """Return candidate keys from most specific to legacy/plain lookup forms."""

    base_key = base_track_record_key(info)
    if base_key is None:
        return ()
    difficulty = record_difficulty(info)
    keys = [base_key if difficulty is None else f"{base_key}#difficulty={difficulty}"]
    keys.extend(_legacy_track_keys(info))
    keys.append(base_key)
    return tuple(dict.fromkeys(keys))


@dataclass(frozen=True, slots=True)
class TrackAttemptStats:
    attempts: int = 0
    finishes: int = 0
    completion_samples: int = 0
    completion_sum: float = 0.0
    best_completion: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, int | float] | None) -> TrackAttemptStats:
        return cls(
            attempts=_stats_int(values, "attempts"),
            finishes=_stats_int(values, "finishes"),
            completion_samples=_stats_int(values, "completion_samples"),
            completion_sum=_stats_float(values, "completion_sum"),
            best_completion=_stats_float(values, "best_completion"),
        )

    def as_mapping(self) -> dict[str, int | float]:
        return {
            "attempts": self.attempts,
            "finishes": self.finishes,
            "completion_samples": self.completion_samples,
            "completion_sum": self.completion_sum,
            "best_completion": self.best_completion,
        }

    def update(
        self,
        *,
        info: Mapping[str, object],
        telemetry: FZeroXTelemetry | None,
        episode_done: bool,
    ) -> TrackAttemptStats:
        if not episode_done:
            return self
        completion = _episode_completion_fraction(info, telemetry)
        completion_samples = self.completion_samples
        completion_sum = self.completion_sum
        best_completion = self.best_completion
        if completion is not None:
            completion_samples += 1
            completion_sum += completion
            best_completion = max(best_completion, completion)
        return TrackAttemptStats(
            attempts=self.attempts + 1,
            finishes=self.finishes + (1 if info.get("termination_reason") == "finished" else 0),
            completion_samples=completion_samples,
            completion_sum=completion_sum,
            best_completion=best_completion,
        )


@dataclass(frozen=True, slots=True)
class TrackRecordEntry:
    best_finish_rank: int | None = None
    best_finish_rank_time_ms: int | None = None
    best_finish_rank_setup: TrackFinishSetup | None = None
    best_finish_time_ms: int | None = None
    best_finish_time_rank: int | None = None
    best_finish_time_setup: TrackFinishSetup | None = None
    latest_finish_rank: int | None = None
    latest_finish_time_ms: int | None = None
    latest_finish_delta_ms: int | None = None
    latest_finish_setup: TrackFinishSetup | None = None
    attempt_stats: TrackAttemptStats = field(default_factory=TrackAttemptStats)
    failed_attempt: bool = False

    def update(
        self,
        *,
        info: Mapping[str, object],
        telemetry: FZeroXTelemetry | None,
        episode_done: bool,
    ) -> TrackRecordEntry:
        finish_time_ms = _successful_finish_time_ms(info, telemetry)
        finish_position = _successful_finish_position(info, telemetry)
        setup = _finish_setup(info, telemetry) if finish_time_ms is not None else None
        best_finish_time_ms = self.best_finish_time_ms
        best_finish_time_rank = self.best_finish_time_rank
        best_finish_time_setup = self.best_finish_time_setup
        latest_finish_rank = self.latest_finish_rank
        latest_finish_time_ms = self.latest_finish_time_ms
        latest_finish_delta_ms = self.latest_finish_delta_ms
        latest_finish_setup = self.latest_finish_setup
        if finish_time_ms is not None:
            latest_finish_rank = finish_position
            latest_finish_time_ms = finish_time_ms
            latest_finish_delta_ms = (
                None if best_finish_time_ms is None else finish_time_ms - best_finish_time_ms
            )
            latest_finish_setup = setup or latest_finish_setup
            if best_finish_time_ms is None or finish_time_ms < best_finish_time_ms:
                best_finish_time_ms = finish_time_ms
                best_finish_time_rank = finish_position
                best_finish_time_setup = setup or best_finish_time_setup

        best_finish_rank = self.best_finish_rank
        best_finish_rank_time_ms = self.best_finish_rank_time_ms
        best_finish_rank_setup = self.best_finish_rank_setup
        if (
            finish_time_ms is not None
            and finish_position is not None
            and _is_new_best_rank(
                current_rank=best_finish_rank,
                current_time_ms=best_finish_rank_time_ms,
                finish_position=finish_position,
                finish_time_ms=finish_time_ms,
            )
        ):
            best_finish_rank = finish_position
            best_finish_rank_time_ms = finish_time_ms
            best_finish_rank_setup = setup or best_finish_rank_setup

        failed_attempt = self.failed_attempt
        if episode_done:
            failed_attempt = info.get("termination_reason") != "finished"
        return TrackRecordEntry(
            best_finish_rank=best_finish_rank,
            best_finish_rank_time_ms=best_finish_rank_time_ms,
            best_finish_rank_setup=_copy_setup(best_finish_rank_setup),
            best_finish_time_ms=best_finish_time_ms,
            best_finish_time_rank=best_finish_time_rank,
            best_finish_time_setup=_copy_setup(best_finish_time_setup),
            latest_finish_rank=latest_finish_rank,
            latest_finish_time_ms=latest_finish_time_ms,
            latest_finish_delta_ms=latest_finish_delta_ms,
            latest_finish_setup=_copy_setup(latest_finish_setup),
            attempt_stats=self.attempt_stats.update(
                info=info,
                telemetry=telemetry,
                episode_done=episode_done,
            ),
            failed_attempt=failed_attempt,
        )


@dataclass(frozen=True, slots=True)
class TrackRecordBook:
    entries: dict[str, TrackRecordEntry] = field(default_factory=dict)
    best_finish_position: int | None = None

    @property
    def is_empty(self) -> bool:
        return self.best_finish_position is None and not self.entries

    def snapshot(self) -> TrackRecordBook:
        return TrackRecordBook(
            entries={
                key: TrackRecordEntry(
                    best_finish_rank=entry.best_finish_rank,
                    best_finish_rank_time_ms=entry.best_finish_rank_time_ms,
                    best_finish_rank_setup=_copy_setup(entry.best_finish_rank_setup),
                    best_finish_time_ms=entry.best_finish_time_ms,
                    best_finish_time_rank=entry.best_finish_time_rank,
                    best_finish_time_setup=_copy_setup(entry.best_finish_time_setup),
                    latest_finish_rank=entry.latest_finish_rank,
                    latest_finish_time_ms=entry.latest_finish_time_ms,
                    latest_finish_delta_ms=entry.latest_finish_delta_ms,
                    latest_finish_setup=_copy_setup(entry.latest_finish_setup),
                    attempt_stats=entry.attempt_stats,
                    failed_attempt=entry.failed_attempt,
                )
                for key, entry in self.entries.items()
            },
            best_finish_position=self.best_finish_position,
        )

    def update(
        self,
        info: Mapping[str, object],
        telemetry: FZeroXTelemetry | None,
        *,
        episode_done: bool,
    ) -> TrackRecordBook:
        best_finish_position = _updated_best_finish_position(
            self.best_finish_position,
            info,
            telemetry,
        )
        track_key = track_record_key(info)
        if track_key is None:
            if best_finish_position == self.best_finish_position:
                return self
            return TrackRecordBook(
                entries={key: value for key, value in self.entries.items()},
                best_finish_position=best_finish_position,
            )
        current_entry = self.entries.get(track_key, TrackRecordEntry())
        updated_entry = current_entry.update(
            info=info,
            telemetry=telemetry,
            episode_done=episode_done,
        )
        if updated_entry == current_entry and best_finish_position == self.best_finish_position:
            return self
        entries = {key: value for key, value in self.entries.items()}
        entries[track_key] = updated_entry
        return TrackRecordBook(
            entries=entries,
            best_finish_position=best_finish_position,
        )

    def entry_for(self, info: Mapping[str, object]) -> TrackRecordEntry | None:
        for track_key in track_record_lookup_keys(info):
            entry = self.entries.get(track_key)
            if entry is not None:
                return entry
        return None


def base_track_record_key(info: Mapping[str, object]) -> TrackRecordKey | None:
    if _is_generated_x_cup_record(info):
        generated_hash = info.get("track_generated_course_hash")
        if isinstance(generated_hash, str) and generated_hash:
            return f"x_cup:{generated_hash}"

    course_id = info.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        return course_id

    value = info.get("track_course_index", info.get("course_index"))
    if isinstance(value, int) and not isinstance(value, bool):
        return f"course:{value}"

    for key in ("track_id", "track_display_name"):
        value = info.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def record_difficulty(info: Mapping[str, object]) -> str | None:
    for key in (
        "track_gp_difficulty",
        "track_source_gp_difficulty",
        "gp_difficulty",
        "source_gp_difficulty",
    ):
        difficulty = _valid_difficulty(info.get(key))
        if difficulty is not None:
            return difficulty

    if info.get("track_mode") == X_CUP_COURSE.race_mode:
        for key in ("difficulty_name", "difficulty"):
            difficulty = _valid_difficulty(info.get(key))
            if difficulty is not None:
                return difficulty
    return None


def _valid_difficulty(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value if is_race_difficulty_name(value) else None


def _legacy_track_keys(info: Mapping[str, object]) -> tuple[str, ...]:
    keys: list[str] = []

    course_id = info.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        keys.append(course_id)

    course_index = info.get("track_course_index", info.get("course_index"))
    if isinstance(course_index, int) and not isinstance(course_index, bool):
        keys.append(f"course:{course_index}")

    for field_name in ("track_id", "track_display_name"):
        value = info.get(field_name)
        if isinstance(value, str) and value:
            keys.append(value)

    return tuple(keys)


def _is_generated_x_cup_record(info: Mapping[str, object]) -> bool:
    if info.get("track_generated_course_kind") == X_CUP_COURSE.generated_kind:
        return True

    for key in ("track_runtime_course_key", "track_reset_course_key", "track_course_id"):
        value = info.get(key)
        if isinstance(value, str) and value.startswith(X_CUP_COURSE.id_prefix):
            return True

    course_index = info.get("track_course_index", info.get("course_index"))
    return (
        isinstance(course_index, int)
        and not isinstance(course_index, bool)
        and (course_index == X_CUP_COURSE.course_index)
    )


def _updated_best_finish_position(
    best_finish_position: int | None,
    info: Mapping[str, object],
    telemetry: FZeroXTelemetry | None,
) -> int | None:
    finish_position = _successful_finish_position(info, telemetry)
    if finish_position is None:
        return best_finish_position
    if best_finish_position is None:
        return finish_position
    return min(best_finish_position, finish_position)


def _successful_finish_position(
    info: Mapping[str, object],
    telemetry: FZeroXTelemetry | None,
) -> int | None:
    if info.get("termination_reason") != "finished":
        return None

    raw_position: object
    if telemetry is not None:
        raw_position = telemetry.player.position
    else:
        raw_position = info.get("position")
    if isinstance(raw_position, bool) or not isinstance(raw_position, int):
        return None
    if raw_position <= 0:
        return None
    return raw_position


def _successful_finish_time_ms(
    info: Mapping[str, object],
    telemetry: FZeroXTelemetry | None,
) -> int | None:
    if info.get("termination_reason") != "finished":
        return None

    raw_time: object
    if telemetry is not None:
        raw_time = telemetry.player.race_time_ms
    else:
        raw_time = info.get("race_time_ms")
    if isinstance(raw_time, bool) or not isinstance(raw_time, int):
        return None
    if raw_time <= 0:
        return None
    return raw_time


def _episode_completion_fraction(
    info: Mapping[str, object],
    telemetry: FZeroXTelemetry | None,
) -> float | None:
    value = info.get("episode_completion_fraction")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return _clamped_fraction(float(value))
    if info.get("termination_reason") == "finished":
        return 1.0
    if telemetry is None:
        return None

    course_length = float(telemetry.course_length)
    total_lap_count = int(telemetry.total_lap_count)
    total_race_distance = course_length * total_lap_count
    if total_race_distance <= 0.0:
        return None
    return _clamped_fraction(float(telemetry.player.race_distance) / total_race_distance)


def _is_new_best_rank(
    *,
    current_rank: int | None,
    current_time_ms: int | None,
    finish_position: int,
    finish_time_ms: int,
) -> bool:
    if current_rank is None or finish_position < current_rank:
        return True
    if finish_position != current_rank:
        return False
    return current_time_ms is None or finish_time_ms < current_time_ms


def _finish_setup(
    info: Mapping[str, object],
    telemetry: FZeroXTelemetry | None,
) -> TrackFinishSetup:
    setup: TrackFinishSetup = {}
    vehicle_name, vehicle_id = _finish_vehicle(info, telemetry)
    engine_raw = _finish_engine_setting_raw(info, telemetry)
    if vehicle_name is not None:
        setup["vehicle_name"] = vehicle_name
    if vehicle_id is not None:
        setup["vehicle"] = vehicle_id
    if engine_raw is not None:
        setup["engine_setting_raw_value"] = engine_raw
    return setup


def _finish_vehicle(
    info: Mapping[str, object],
    telemetry: FZeroXTelemetry | None,
) -> tuple[str | None, str | None]:
    if telemetry is not None:
        character_index = _int_setup_value(
            getattr(telemetry.player, "machine_character_index", None)
        )
        if character_index is not None:
            vehicle = vehicle_by_character_index(character_index)
            if vehicle is not None:
                return vehicle.display_name, vehicle.id

    for key in ("track_vehicle_name", "vehicle_name"):
        value = info.get(key)
        if isinstance(value, str) and value:
            vehicle_id = _string_info(info, "track_vehicle") or _string_info(info, "vehicle")
            return value, vehicle_id
    vehicle_id = _string_info(info, "track_vehicle") or _string_info(info, "vehicle")
    return vehicle_id, vehicle_id


def _finish_engine_setting_raw(
    info: Mapping[str, object],
    telemetry: FZeroXTelemetry | None,
) -> int | None:
    if telemetry is not None:
        engine_setting = telemetry.player.engine_setting
        if math.isfinite(float(engine_setting)):
            raw_value = engine_value_to_slider_step(float(engine_setting))
            if 0 <= raw_value <= ENGINE_SLIDER.max_step:
                return raw_value

    for key in (
        "engine_setting_raw_value",
        "engine_setting_raw_value_ram",
        "track_engine_setting_raw_value",
    ):
        raw_value = _int_setup_value(info.get(key))
        if raw_value is not None and 0 <= raw_value <= ENGINE_SLIDER.max_step:
            return raw_value
    return None


def _string_info(info: Mapping[str, object], key: str) -> str | None:
    value = info.get(key)
    return value if isinstance(value, str) and value else None


def _int_setup_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            return None
        return int(round(value))
    return None


def _stats_int(values: Mapping[str, int | float] | None, key: str) -> int:
    if values is None:
        return 0
    value = values.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return max(0, int(value))
    return 0


def _stats_float(values: Mapping[str, int | float] | None, key: str) -> float:
    if values is None:
        return 0.0
    value = values.get(key)
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return max(0.0, float(value))
    return 0.0


def _copy_setup(value: Mapping[str, str | int] | None) -> TrackFinishSetup | None:
    if value is None:
        return None
    return {key: item for key, item in value.items()}


def _clamped_fraction(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
