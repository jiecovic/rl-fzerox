# src/rl_fzerox/core/manager/projection/tracks.py
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from rl_fzerox.core.domain.courses import built_in_course_ref_by_id
from rl_fzerox.core.domain.race_difficulty import default_gp_difficulty
from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.vehicle_catalog import resolve_engine_setting, vehicle_by_id


@dataclass(frozen=True, slots=True)
class GeneratedXCupCourse:
    """Stable generated-course identity used before materialization writes state."""

    course_id: str
    display_name: str
    course_hash: str
    seed: int


def build_track_sampling_data(config: ManagedRunConfig) -> dict[str, object]:
    return {
        "enabled": True,
        "sampling_mode": _runtime_track_sampling_mode(config),
        "entries": _track_sampling_entries(config),
        "step_balance_update_episodes": config.tracks.step_balance_update_episodes,
        "step_balance_ema_alpha": config.tracks.step_balance_ema_alpha,
        "step_balance_max_weight_scale": config.tracks.step_balance_max_weight_scale,
        "adaptive_step_balance_completion_weight": (
            config.tracks.adaptive_step_balance_completion_weight
        ),
        "adaptive_step_balance_target_completion": (
            config.tracks.adaptive_step_balance_target_completion
        ),
    }


def _runtime_track_sampling_mode(config: ManagedRunConfig) -> str:
    sampling_mode = config.tracks.sampling_mode
    if sampling_mode == "equal":
        return "balanced"
    return sampling_mode


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
                    gp_difficulty=(
                        config.tracks.gp_difficulty
                        if config.tracks.race_mode == "gp_race"
                        else None
                    ),
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
    if config.tracks.include_x_cup:
        for x_cup_index in range(config.tracks.x_cup_course_count):
            generated_course = _x_cup_generated_course(config, index=x_cup_index)
            for vehicle_id in config.vehicle.selected_vehicle_ids:
                entries.append(
                    _track_sampling_entry(
                        course_id=generated_course.course_id,
                        course_ref=None,
                        race_mode=config.tracks.race_mode,
                        gp_difficulty=config.tracks.gp_difficulty,
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
                        course_index=X_CUP_COURSE.course_index,
                        course_name=generated_course.display_name,
                        display_name=generated_course.display_name,
                        source_course_index=X_CUP_COURSE.course_index,
                        generated_course_kind=X_CUP_COURSE.generated_kind,
                        generated_course_seed=generated_course.seed,
                        generated_course_hash=generated_course.course_hash,
                        log_per_course=False,
                    )
                )
    return entries


def _track_sampling_entry(
    *,
    course_id: str,
    course_ref: str | None,
    race_mode: str,
    gp_difficulty: str | None,
    target_vehicle_id: str,
    source_vehicle_id: str,
    source_engine_setting_id: str,
    source_engine_setting_raw_value: int,
    fixed_engine_setting_raw_value: int | None,
    random_engine_min_raw_value: int | None,
    random_engine_max_raw_value: int | None,
    course_index: int | None = None,
    course_name: str | None = None,
    display_name: str | None = None,
    source_course_index: int | None = None,
    generated_course_kind: str | None = None,
    generated_course_seed: int | None = None,
    generated_course_hash: str | None = None,
    log_per_course: bool = True,
) -> dict[str, object]:
    vehicle = vehicle_by_id(target_vehicle_id)
    resolved_gp_difficulty = (
        default_gp_difficulty()
        if race_mode == "gp_race" and gp_difficulty is None
        else gp_difficulty
    )
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

    entry = {
        "id": _track_sampling_entry_id(
            course_id=course_id,
            race_mode=race_mode,
            gp_difficulty=resolved_gp_difficulty,
            target_vehicle_id=target_vehicle_id,
            engine_suffix=engine_suffix,
        ),
        "course_ref": course_ref,
        "course_id": course_id,
        "course_name": course_name,
        "course_index": course_index,
        "display_name": display_name,
        "mode": race_mode,
        "gp_difficulty": resolved_gp_difficulty,
        "vehicle": target_vehicle_id,
        "vehicle_name": vehicle.display_name,
        "source_vehicle": source_vehicle_id,
        "engine_setting": engine_id,
        "engine_setting_raw_value": engine_raw,
        "source_engine_setting": source_engine_setting_id,
        "source_engine_setting_raw_value": source_engine_setting_raw_value,
        "source_course_index": source_course_index,
        "engine_setting_min_raw_value": random_engine_min_raw_value,
        "engine_setting_max_raw_value": random_engine_max_raw_value,
        "generated_course_kind": generated_course_kind,
        "generated_course_seed": generated_course_seed,
        "generated_course_hash": generated_course_hash,
        "log_per_course": log_per_course,
    }
    return {key: value for key, value in entry.items() if value is not None}


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


def _track_sampling_entry_id(
    *,
    course_id: str,
    race_mode: str,
    gp_difficulty: str | None,
    target_vehicle_id: str,
    engine_suffix: str,
) -> str:
    if race_mode == "gp_race" and gp_difficulty is not None:
        return f"{course_id}_{race_mode}_{gp_difficulty}_{target_vehicle_id}_{engine_suffix}"
    return f"{course_id}_{race_mode}_{target_vehicle_id}_{engine_suffix}"


def _x_cup_generated_course(config: ManagedRunConfig, *, index: int) -> GeneratedXCupCourse:
    seed = _x_cup_seed(config.seed, index=index)
    course_hash = _x_cup_course_hash(
        seed=seed,
        gp_difficulty=(
            default_gp_difficulty()
            if config.tracks.gp_difficulty is None
            else config.tracks.gp_difficulty
        ),
    )
    short_hash = course_hash[: X_CUP_COURSE.display_hash_chars]
    return GeneratedXCupCourse(
        course_id=f"{X_CUP_COURSE.id_prefix}_{short_hash}",
        display_name=f"{X_CUP_COURSE.display_prefix} {short_hash}",
        course_hash=course_hash,
        seed=seed,
    )


def _x_cup_seed(master_seed: int, *, index: int) -> int:
    payload = {
        "generator_version": X_CUP_COURSE.generator_version,
        "index": index,
        "master_seed": int(master_seed),
        "source_course_index": X_CUP_COURSE.course_index,
    }
    return int.from_bytes(_sha256_json(payload)[:8], byteorder="big", signed=False)


def _x_cup_course_hash(*, seed: int, gp_difficulty: str) -> str:
    return _sha256_json(
        {
            "generator_version": X_CUP_COURSE.generator_version,
            "gp_difficulty": gp_difficulty,
            "race_mode": X_CUP_COURSE.race_mode,
            "rng_patch_timing": X_CUP_COURSE.rng_patch_timing,
            "seed": seed,
            "source_course_index": X_CUP_COURSE.course_index,
        }
    ).hex()


def _sha256_json(payload: Mapping[str, object]) -> bytes:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).digest()
