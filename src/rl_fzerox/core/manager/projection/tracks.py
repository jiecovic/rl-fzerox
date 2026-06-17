# src/rl_fzerox/core/manager/projection/tracks.py
from __future__ import annotations

from rl_fzerox.core.domain.courses import built_in_course_ref_by_id
from rl_fzerox.core.domain.race_difficulty import default_gp_difficulty
from rl_fzerox.core.domain.x_cup import (
    X_CUP_COURSE,
    generated_x_cup_course_identity,
    generated_x_cup_slot_key,
)
from rl_fzerox.core.manager.projection.engine_tuning import adaptive_engine_tuning_config
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.track_sampling_identity import track_sampling_entry_id
from rl_fzerox.core.runtime_spec.vehicle_catalog import vehicle_by_id


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
        "adaptive_step_balance_min_confidence_episodes": (
            config.tracks.adaptive_step_balance_min_confidence_episodes
        ),
        "adaptive_step_balance_confidence_scale": (
            config.tracks.adaptive_step_balance_confidence_scale
        ),
        "deficit_budget_uniform_fraction": config.tracks.deficit_budget_uniform_fraction,
        "deficit_budget_focus_sharpness": config.tracks.deficit_budget_focus_sharpness,
        "deficit_budget_ema_alpha": config.tracks.deficit_budget_ema_alpha,
        "deficit_budget_weight_update_rollouts": (
            config.tracks.deficit_budget_weight_update_rollouts
        ),
        "deficit_budget_difficulty_metric": config.tracks.deficit_budget_difficulty_metric,
        "deficit_budget_warmup_min_episodes_per_course": (
            config.tracks.deficit_budget_warmup_min_episodes_per_course
        ),
        "x_cup_rotation": {
            "enabled": config.tracks.x_cup_auto_regeneration.enabled,
            "completion_threshold": (config.tracks.x_cup_auto_regeneration.completion_threshold),
            "min_episodes": config.tracks.x_cup_auto_regeneration.min_episodes,
            "max_episodes": config.tracks.x_cup_auto_regeneration.max_episodes,
            "ema_alpha": config.tracks.x_cup_auto_regeneration.ema_alpha,
        },
        "engine_tuning": _engine_tuning(config),
    }


def _runtime_track_sampling_mode(config: ManagedRunConfig) -> str:
    sampling_mode = config.tracks.sampling_mode
    if sampling_mode == "equal":
        return "balanced"
    return sampling_mode


def _track_sampling_entries(config: ManagedRunConfig) -> list[dict[str, object]]:
    source_vehicle_id = config.vehicle.selected_vehicle_ids[0]
    source_engine = _source_engine_setting(config)
    gp_difficulties = _gp_difficulties(config)
    entries: list[dict[str, object]] = []
    for course_id in config.tracks.selected_course_ids:
        course_ref = _course_ref(course_id)
        for gp_difficulty in gp_difficulties:
            for vehicle_id in config.vehicle.selected_vehicle_ids:
                entries.append(
                    _track_sampling_entry(
                        course_id=course_id,
                        runtime_course_key=course_id,
                        course_ref=course_ref,
                        race_mode=config.tracks.race_mode,
                        gp_difficulty=gp_difficulty,
                        target_vehicle_id=vehicle_id,
                        source_vehicle_id=source_vehicle_id,
                        source_engine_setting_raw_value=source_engine,
                        fixed_engine_setting_raw_value=(
                            config.vehicle.engine_setting_raw_value
                            if config.vehicle.engine_mode == "fixed"
                            else None
                        ),
                        random_engine_min_raw_value=(
                            config.vehicle.engine_setting_min_raw_value
                            if config.vehicle.engine_mode in {"random_range", "adaptive_tuner"}
                            else None
                        ),
                        random_engine_max_raw_value=(
                            config.vehicle.engine_setting_max_raw_value
                            if config.vehicle.engine_mode in {"random_range", "adaptive_tuner"}
                            else None
                        ),
                    )
                )
    if config.tracks.include_x_cup:
        for x_cup_index in range(config.tracks.x_cup_course_count):
            for gp_difficulty in gp_difficulties:
                generated_course = generated_x_cup_course_identity(
                    master_seed=config.seed,
                    slot=x_cup_index,
                    generation=1,
                )
                for vehicle_id in config.vehicle.selected_vehicle_ids:
                    entries.append(
                        _track_sampling_entry(
                            course_id=generated_course.course_id,
                            runtime_course_key=generated_x_cup_slot_key(generated_course.slot),
                            course_ref=None,
                            race_mode=config.tracks.race_mode,
                            gp_difficulty=gp_difficulty,
                            target_vehicle_id=vehicle_id,
                            source_vehicle_id=source_vehicle_id,
                            source_engine_setting_raw_value=source_engine,
                            fixed_engine_setting_raw_value=(
                                config.vehicle.engine_setting_raw_value
                                if config.vehicle.engine_mode == "fixed"
                                else None
                            ),
                            random_engine_min_raw_value=(
                                config.vehicle.engine_setting_min_raw_value
                                if config.vehicle.engine_mode in {"random_range", "adaptive_tuner"}
                                else None
                            ),
                            random_engine_max_raw_value=(
                                config.vehicle.engine_setting_max_raw_value
                                if config.vehicle.engine_mode in {"random_range", "adaptive_tuner"}
                                else None
                            ),
                            course_index=X_CUP_COURSE.course_index,
                            course_name=generated_course.display_name,
                            display_name=generated_course.display_name,
                            source_course_index=X_CUP_COURSE.course_index,
                            generated_course_kind=X_CUP_COURSE.generated_kind,
                            generated_course_seed=generated_course.seed,
                            generated_course_hash=generated_course.course_hash,
                            generated_course_slot=generated_course.slot,
                            generated_course_generation=generated_course.generation,
                            log_per_course=False,
                        )
                    )
    return entries


def _gp_difficulties(config: ManagedRunConfig) -> tuple[str | None, ...]:
    if config.tracks.race_mode != "gp_race":
        return (None,)
    return tuple(config.tracks.gp_difficulties) or (default_gp_difficulty(),)


def _track_sampling_entry(
    *,
    course_id: str,
    runtime_course_key: str | None = None,
    course_ref: str | None,
    race_mode: str,
    gp_difficulty: str | None,
    target_vehicle_id: str,
    source_vehicle_id: str,
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
    generated_course_slot: int | None = None,
    generated_course_generation: int | None = None,
    log_per_course: bool = True,
) -> dict[str, object]:
    vehicle = vehicle_by_id(target_vehicle_id)
    resolved_gp_difficulty = (
        default_gp_difficulty()
        if race_mode == "gp_race" and gp_difficulty is None
        else gp_difficulty
    )
    if fixed_engine_setting_raw_value is not None:
        engine_raw = fixed_engine_setting_raw_value
    else:
        if random_engine_min_raw_value is None or random_engine_max_raw_value is None:
            raise ValueError("random engine range requires both min and max raw values")
        engine_raw = source_engine_setting_raw_value

    entry_id = track_sampling_entry_id(
        course_id=course_id,
        runtime_course_key=runtime_course_key,
        mode=race_mode,
        gp_difficulty=resolved_gp_difficulty,
        vehicle=target_vehicle_id,
    )
    entry = {
        "id": entry_id,
        "course_ref": course_ref,
        "course_id": course_id,
        "runtime_course_key": runtime_course_key,
        "course_name": course_name,
        "course_index": course_index,
        "display_name": display_name,
        "mode": race_mode,
        "gp_difficulty": resolved_gp_difficulty,
        "vehicle": target_vehicle_id,
        "vehicle_name": vehicle.display_name,
        "source_vehicle": source_vehicle_id,
        "engine_setting_raw_value": engine_raw,
        "source_engine_setting_raw_value": source_engine_setting_raw_value,
        "source_course_index": source_course_index,
        "engine_setting_min_raw_value": random_engine_min_raw_value,
        "engine_setting_max_raw_value": random_engine_max_raw_value,
        "generated_course_kind": generated_course_kind,
        "generated_course_seed": generated_course_seed,
        "generated_course_hash": generated_course_hash,
        "generated_course_slot": generated_course_slot,
        "generated_course_generation": generated_course_generation,
        "log_per_course": log_per_course,
    }
    return {key: value for key, value in entry.items() if value is not None}


def _source_engine_setting(config: ManagedRunConfig) -> int:
    if config.vehicle.engine_mode == "fixed":
        return config.vehicle.engine_setting_raw_value
    return (
        config.vehicle.engine_setting_min_raw_value + config.vehicle.engine_setting_max_raw_value
    ) // 2


def _engine_tuning(config: ManagedRunConfig) -> dict[str, object]:
    return adaptive_engine_tuning_config(config).model_dump(mode="python")


def _course_ref(course_id: str) -> str:
    matches = built_in_course_ref_by_id(course_id)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one built-in course ref for {course_id!r}")
    return matches[0]
