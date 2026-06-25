# src/rl_fzerox/ui/watch/records/finish.py
"""Finish extraction helpers for Watch track-record updates.

This module converts episode info and optional live telemetry into canonical
finish position, finish time, vehicle, and engine-setting values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.engine import (
    ENGINE_SLIDER,
    engine_value_to_slider_step,
)
from rl_fzerox.core.runtime_spec.vehicle_catalog import vehicle_by_character_index
from rl_fzerox.ui.watch.records.types import TrackFinishSetup


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


def _copy_setup(value: Mapping[str, str | int] | None) -> TrackFinishSetup | None:
    if value is None:
        return None
    return {key: item for key, item in value.items()}
