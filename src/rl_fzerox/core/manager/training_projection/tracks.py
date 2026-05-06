from __future__ import annotations

from rl_fzerox.core.config.vehicle_catalog import resolve_engine_setting, vehicle_by_id
from rl_fzerox.core.domain.courses import built_in_course_ref_by_id
from rl_fzerox.core.manager.config import ManagedRunConfig


def build_track_sampling_data(config: ManagedRunConfig) -> dict[str, object]:
    return {
        "enabled": True,
        "sampling_mode": "random" if config.tracks.sampling_mode == "equal" else "step_balanced",
        "entries": _track_sampling_entries(config),
    }


def _track_sampling_entries(config: ManagedRunConfig) -> list[dict[str, object]]:
    source_vehicle_id = config.vehicle.selected_vehicle_ids[0]
    source_engine = _source_engine_setting(config)
    entries: list[dict[str, object]] = []
    for course_id in config.tracks.selected_course_ids:
        course_ref = _course_ref(course_id)
        for vehicle_id in config.vehicle.selected_vehicle_ids:
            entries.append(
                _track_sampling_entry(
                    course_id=course_id,
                    course_ref=course_ref,
                    race_mode=config.tracks.race_mode,
                    target_vehicle_id=vehicle_id,
                    source_vehicle_id=source_vehicle_id,
                    source_engine_setting_id=source_engine.id,
                    source_engine_setting_raw_value=source_engine.raw_value,
                    fixed_engine_setting_raw_value=(
                        config.vehicle.engine_setting_raw_value
                        if config.vehicle.engine_mode == "fixed"
                        else None
                    ),
                    random_engine_min_raw_value=(
                        config.vehicle.engine_setting_min_raw_value
                        if config.vehicle.engine_mode == "random_range"
                        else None
                    ),
                    random_engine_max_raw_value=(
                        config.vehicle.engine_setting_max_raw_value
                        if config.vehicle.engine_mode == "random_range"
                        else None
                    ),
                )
            )
    return entries


def _track_sampling_entry(
    *,
    course_id: str,
    course_ref: str,
    race_mode: str,
    target_vehicle_id: str,
    source_vehicle_id: str,
    source_engine_setting_id: str,
    source_engine_setting_raw_value: int,
    fixed_engine_setting_raw_value: int | None,
    random_engine_min_raw_value: int | None,
    random_engine_max_raw_value: int | None,
) -> dict[str, object]:
    vehicle = vehicle_by_id(target_vehicle_id)
    if fixed_engine_setting_raw_value is not None:
        target_engine = resolve_engine_setting(
            fixed_engine_setting_raw_value,
            context=f"manager track_sampling {course_id}/{target_vehicle_id}",
        )
        engine_id = target_engine.id
        engine_raw = target_engine.raw_value
        engine_suffix = engine_id
    else:
        if random_engine_min_raw_value is None or random_engine_max_raw_value is None:
            raise ValueError("random engine range requires both min and max raw values")
        engine_id = source_engine_setting_id
        engine_raw = source_engine_setting_raw_value
        engine_suffix = f"engine_range_{random_engine_min_raw_value}_{random_engine_max_raw_value}"

    return {
        "id": f"{course_id}_{race_mode}_{target_vehicle_id}_{engine_suffix}",
        "course_ref": course_ref,
        "mode": race_mode,
        "vehicle": target_vehicle_id,
        "vehicle_name": vehicle.display_name,
        "source_vehicle": source_vehicle_id,
        "engine_setting": engine_id,
        "engine_setting_raw_value": engine_raw,
        "source_engine_setting": source_engine_setting_id,
        "source_engine_setting_raw_value": source_engine_setting_raw_value,
        "engine_setting_min_raw_value": random_engine_min_raw_value,
        "engine_setting_max_raw_value": random_engine_max_raw_value,
    }


def _source_engine_setting(config: ManagedRunConfig):
    if config.vehicle.engine_mode == "fixed":
        raw_value = config.vehicle.engine_setting_raw_value
    else:
        raw_value = (
            config.vehicle.engine_setting_min_raw_value
            + config.vehicle.engine_setting_max_raw_value
        ) // 2
    return resolve_engine_setting(
        raw_value,
        context="manager source engine setting",
    )


def _course_ref(course_id: str) -> str:
    matches = built_in_course_ref_by_id(course_id)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one built-in course ref for {course_id!r}")
    return matches[0]
