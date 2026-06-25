# src/rl_fzerox/core/envs/engine/reset/track_sampling/projection.py
"""Projection from runtime config entries to selected reset targets.

The selector uses this module to combine static track config, engine tuning
choice, baseline variants, and generated-course metadata into `SelectedTrack`.
"""
from __future__ import annotations

from rl_fzerox.core.engine_tuning import (
    EngineTuningResetSampler,
    EngineTuningSelectionMode,
)
from rl_fzerox.core.envs.engine.reset.track_sampling.engine_choice import (
    choose_engine_tuning,
    resolve_entry_engine_setting,
)
from rl_fzerox.core.envs.engine.reset.track_sampling.models import SelectedTrack
from rl_fzerox.core.runtime_spec.schema import (
    AdaptiveEngineTuningConfig,
    TrackSamplingEntryConfig,
)


def selected_track_from_entry(
    entry: TrackSamplingEntryConfig,
    *,
    sampling_mode: str,
    cycle_position: int | None = None,
    seed: int | None,
    engine_tuning_config: AdaptiveEngineTuningConfig,
    engine_tuning_sampler: EngineTuningResetSampler | None,
    engine_tuning_selection: EngineTuningSelectionMode,
) -> SelectedTrack:
    if entry.baseline_state_path is None:
        raise ValueError(
            f"track sampling entry {entry.id!r} has no materialized baseline_state_path"
        )
    engine_choice = choose_engine_tuning(
        entry,
        config=engine_tuning_config,
        sampler=engine_tuning_sampler,
        seed=seed,
        selection=engine_tuning_selection,
    )
    resolved_engine = resolve_entry_engine_setting(
        entry,
        seed=seed,
        engine_choice=engine_choice,
    )
    return SelectedTrack(
        id=entry.id,
        display_name=entry.display_name,
        course_ref=entry.course_ref,
        course_id=entry.course_id,
        runtime_course_key=entry.runtime_course_key,
        course_name=entry.course_name,
        baseline_state_path=entry.baseline_state_path,
        weight=float(entry.weight),
        course_index=None if entry.course_index is None else int(entry.course_index),
        mode=entry.mode,
        gp_difficulty=entry.gp_difficulty,
        vehicle=entry.vehicle,
        vehicle_name=entry.vehicle_name,
        engine_setting_raw_value=(
            None if resolved_engine is None else int(resolved_engine.raw_value)
        ),
        engine_setting_min_raw_value=entry.engine_setting_min_raw_value,
        engine_setting_max_raw_value=entry.engine_setting_max_raw_value,
        engine_tuning_context_key=None if engine_choice is None else engine_choice.context.key,
        engine_tuning_course_key=(
            None if engine_choice is None else engine_choice.context.course_key
        ),
        engine_tuning_vehicle_id=(
            None if engine_choice is None else engine_choice.context.vehicle_id
        ),
        engine_tuning_sampled_score=None if engine_choice is None else engine_choice.sampled_score,
        engine_tuning_mean_score=None if engine_choice is None else engine_choice.mean_score,
        engine_tuning_uncertainty_score=(
            None if engine_choice is None else engine_choice.uncertainty_score
        ),
        engine_tuning_finish_count=None if engine_choice is None else engine_choice.finish_count,
        source_vehicle=entry.source_vehicle,
        source_course_index=entry.source_course_index,
        source_gp_difficulty=entry.source_gp_difficulty,
        source_engine_setting_raw_value=entry.source_engine_setting_raw_value,
        baseline_group_id=entry.baseline_group_id,
        baseline_group_weight=(
            None if entry.baseline_group_weight is None else float(entry.baseline_group_weight)
        ),
        baseline_variant_index=entry.baseline_variant_index,
        baseline_variant_count=entry.baseline_variant_count,
        baseline_variant_seed=entry.baseline_variant_seed,
        alt_baseline_id=entry.alt_baseline_id,
        alt_baseline_label=entry.alt_baseline_label,
        alt_baseline_source_entry_id=entry.alt_baseline_source_entry_id,
        generated_course_kind=entry.generated_course_kind,
        generated_course_seed=entry.generated_course_seed,
        generated_course_hash=entry.generated_course_hash,
        generated_course_slot=entry.generated_course_slot,
        generated_course_generation=entry.generated_course_generation,
        generated_course_segment_count=entry.generated_course_segment_count,
        generated_course_length=entry.generated_course_length,
        log_per_course=entry.log_per_course,
        records=entry.records,
        sampling_mode=sampling_mode,
        cycle_position=cycle_position,
    )
