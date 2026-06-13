# src/rl_fzerox/core/manager/registry/runs/track_sampling.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from rl_fzerox.core.manager.db.models import (
    RunAltBaselineModel,
    RunTrackSamplingArtifactModel,
    RunTrackSamplingEntryModel,
    RunTrackSamplingGeneratedSlotModel,
    RunTrackSamplingRuntimeModel,
)
from rl_fzerox.core.manager.registry.common import utc_now
from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
from rl_fzerox.core.training.session.callbacks.track_sampling.alt_baselines import (
    TrackSamplingAltBaseline,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.artifacts import (
    TrackSamplingMaterializedArtifact,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def clear_run_track_sampling_state(store: ManagerStore, run_id: str) -> None:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        for artifact in session.scalars(
            select(RunTrackSamplingArtifactModel).where(
                RunTrackSamplingArtifactModel.run_id == run_id
            )
        ):
            session.delete(artifact)
        for slot in session.scalars(
            select(RunTrackSamplingGeneratedSlotModel).where(
                RunTrackSamplingGeneratedSlotModel.run_id == run_id
            )
        ):
            session.delete(slot)
        for entry in session.scalars(
            select(RunTrackSamplingEntryModel).where(RunTrackSamplingEntryModel.run_id == run_id)
        ):
            session.delete(entry)
        runtime = session.get(RunTrackSamplingRuntimeModel, run_id)
        if runtime is not None:
            session.delete(runtime)


def get_run_alt_baselines(
    store: ManagerStore,
    run_id: str,
    *,
    include_deleted: bool = False,
) -> tuple[TrackSamplingAltBaseline, ...]:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        query = select(RunAltBaselineModel).where(RunAltBaselineModel.run_id == run_id)
        if not include_deleted:
            query = query.where(
                RunAltBaselineModel.enabled.is_(True),
                RunAltBaselineModel.deleted_at.is_(None),
            )
        baselines = tuple(
            session.scalars(
                query.order_by(
                    RunAltBaselineModel.course_key,
                    RunAltBaselineModel.reset_variant_key,
                    RunAltBaselineModel.created_at,
                    RunAltBaselineModel.id,
                )
            )
        )
    return tuple(_alt_baseline_from_model(baseline) for baseline in baselines)


def upsert_run_alt_baseline(
    store: ManagerStore,
    *,
    baseline: TrackSamplingAltBaseline,
) -> None:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        values = _alt_baseline_values(baseline)
        stmt = sqlite_insert(RunAltBaselineModel).values(values)
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=("id",),
                set_={key: stmt.excluded[key] for key in values if key != "id"},
            )
        )


def delete_run_alt_baseline(
    store: ManagerStore,
    *,
    run_id: str,
    baseline_id: str,
    deleted_at: str | None = None,
) -> bool:
    del deleted_at
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        baseline = session.get(RunAltBaselineModel, baseline_id)
        if baseline is None or baseline.run_id != run_id:
            return False
        state_path = Path(baseline.state_path)
        session.delete(baseline)

    state_path.unlink(missing_ok=True)
    return True


def clear_run_alt_baselines(store: ManagerStore, run_id: str) -> int:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        baselines = tuple(
            session.scalars(select(RunAltBaselineModel).where(RunAltBaselineModel.run_id == run_id))
        )
        state_paths = tuple(Path(baseline.state_path) for baseline in baselines)
        for baseline in baselines:
            session.delete(baseline)

    for state_path in state_paths:
        state_path.unlink(missing_ok=True)
    return len(baselines)


def get_run_track_sampling_state(
    store: ManagerStore,
    run_id: str,
) -> TrackSamplingRuntimeState | None:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        runtime = session.get(RunTrackSamplingRuntimeModel, run_id)
        entries = tuple(
            session.scalars(
                select(RunTrackSamplingEntryModel)
                .where(RunTrackSamplingEntryModel.run_id == run_id)
                .order_by(RunTrackSamplingEntryModel.course_key)
            )
        )
    if runtime is None or not entries:
        return None
    return TrackSamplingRuntimeState(
        sampling_mode=runtime.sampling_mode,
        action_repeat=runtime.action_repeat,
        update_episodes=runtime.update_episodes,
        ema_alpha=runtime.ema_alpha,
        max_weight_scale=runtime.max_weight_scale,
        adaptive_completion_weight=runtime.adaptive_completion_weight,
        adaptive_target_completion=runtime.adaptive_target_completion,
        adaptive_min_confidence_episodes=runtime.adaptive_min_confidence_episodes,
        adaptive_confidence_scale=runtime.adaptive_confidence_scale,
        update_count=runtime.update_count,
        episodes_since_update=runtime.episodes_since_update,
        entries=tuple(_track_sampling_entry_from_model(entry) for entry in entries),
    )


def upsert_run_track_sampling_state(
    store: ManagerStore,
    *,
    run_id: str,
    state: TrackSamplingRuntimeState,
    updated_at: str | None = None,
) -> None:
    store._ensure_schema_initialized()
    saved_at = updated_at or utc_now()
    with store._orm_session() as session:
        runtime_values = _runtime_values(run_id=run_id, state=state, updated_at=saved_at)
        runtime_insert = sqlite_insert(RunTrackSamplingRuntimeModel).values(runtime_values)
        session.execute(
            runtime_insert.on_conflict_do_update(
                index_elements=("run_id",),
                set_={
                    key: runtime_insert.excluded[key] for key in runtime_values if key != "run_id"
                },
            )
        )

        entry_values = tuple(
            _track_sampling_entry_values(run_id=run_id, entry=entry) for entry in state.entries
        )
        if entry_values:
            entry_insert = sqlite_insert(RunTrackSamplingEntryModel).values(entry_values)
            session.execute(
                entry_insert.on_conflict_do_update(
                    index_elements=("run_id", "course_key"),
                    set_={
                        key: entry_insert.excluded[key]
                        for key in entry_values[0]
                        if key not in {"run_id", "course_key"}
                    },
                )
            )

        stale_entries = delete(RunTrackSamplingEntryModel).where(
            RunTrackSamplingEntryModel.run_id == run_id
        )
        current_keys = [entry.course_key for entry in state.entries]
        if current_keys:
            stale_entries = stale_entries.where(
                ~RunTrackSamplingEntryModel.course_key.in_(current_keys)
            )
        session.execute(stale_entries)


def get_run_track_sampling_artifacts(
    store: ManagerStore,
    run_id: str,
) -> tuple[TrackSamplingMaterializedArtifact, ...]:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        artifacts = tuple(
            session.scalars(
                select(RunTrackSamplingArtifactModel)
                .where(RunTrackSamplingArtifactModel.run_id == run_id)
                .order_by(
                    RunTrackSamplingArtifactModel.course_key,
                    RunTrackSamplingArtifactModel.reset_variant_key,
                )
            )
        )
    return tuple(_track_sampling_artifact_from_model(artifact) for artifact in artifacts)


def replace_run_track_sampling_artifacts(
    store: ManagerStore,
    *,
    run_id: str,
    artifacts: tuple[TrackSamplingMaterializedArtifact, ...],
) -> None:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        session.execute(
            delete(RunTrackSamplingArtifactModel).where(
                RunTrackSamplingArtifactModel.run_id == run_id
            )
        )
        artifact_values = tuple(
            _track_sampling_artifact_values(run_id=run_id, artifact=artifact)
            for artifact in artifacts
        )
        if not artifact_values:
            return
        session.execute(sqlite_insert(RunTrackSamplingArtifactModel).values(artifact_values))


def get_run_generated_x_cup_slots(
    store: ManagerStore,
    run_id: str,
) -> tuple[GeneratedXCupSlot, ...]:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        slots = tuple(
            session.scalars(
                select(RunTrackSamplingGeneratedSlotModel)
                .where(RunTrackSamplingGeneratedSlotModel.run_id == run_id)
                .order_by(RunTrackSamplingGeneratedSlotModel.slot)
            )
        )
    return tuple(_generated_x_cup_slot_from_model(slot) for slot in slots)


def replace_run_generated_x_cup_slots(
    store: ManagerStore,
    *,
    run_id: str,
    slots: tuple[GeneratedXCupSlot, ...],
    updated_at: str | None = None,
) -> None:
    store._ensure_schema_initialized()
    saved_at = updated_at or utc_now()
    with store._orm_session() as session:
        session.execute(
            delete(RunTrackSamplingGeneratedSlotModel).where(
                RunTrackSamplingGeneratedSlotModel.run_id == run_id
            )
        )
        slot_values = tuple(
            _generated_x_cup_slot_values(run_id=run_id, slot=slot, updated_at=saved_at)
            for slot in slots
        )
        if not slot_values:
            return
        session.execute(sqlite_insert(RunTrackSamplingGeneratedSlotModel).values(slot_values))


def _runtime_values(
    *,
    run_id: str,
    state: TrackSamplingRuntimeState,
    updated_at: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "sampling_mode": state.sampling_mode,
        "action_repeat": state.action_repeat,
        "update_episodes": state.update_episodes,
        "ema_alpha": state.ema_alpha,
        "max_weight_scale": state.max_weight_scale,
        "adaptive_completion_weight": state.adaptive_completion_weight,
        "adaptive_target_completion": state.adaptive_target_completion,
        "adaptive_min_confidence_episodes": state.adaptive_min_confidence_episodes,
        "adaptive_confidence_scale": state.adaptive_confidence_scale,
        "update_count": state.update_count,
        "episodes_since_update": state.episodes_since_update,
        "updated_at": updated_at,
    }


def _track_sampling_entry_from_model(
    entry: RunTrackSamplingEntryModel,
) -> TrackSamplingRuntimeEntry:
    return TrackSamplingRuntimeEntry(
        track_id=entry.track_id,
        course_key=entry.course_key,
        label=entry.label,
        base_weight=entry.base_weight,
        current_weight=entry.current_weight,
        completed_frames=entry.completed_frames,
        episode_count=entry.episode_count,
        finished_episode_count=entry.finished_episode_count,
        success_sample_count=entry.success_sample_count,
        ema_episode_frames=entry.ema_episode_frames,
        ema_completion_fraction=entry.ema_completion_fraction,
        generation_episode_count=entry.generation_episode_count,
        generation_finished_episode_count=entry.generation_finished_episode_count,
        generation_success_sample_count=entry.generation_success_sample_count,
        generation_ema_completion_fraction=entry.generation_ema_completion_fraction,
        generated_course_slot=entry.generated_course_slot,
        generated_course_generation=entry.generated_course_generation,
        generated_course_id=entry.generated_course_id,
        generated_course_name=entry.generated_course_name,
        generated_course_hash=entry.generated_course_hash,
        generated_course_seed=_optional_int(entry.generated_course_seed),
        generated_course_segment_count=entry.generated_course_segment_count,
        generated_course_length=entry.generated_course_length,
    )


def _alt_baseline_from_model(
    baseline: RunAltBaselineModel,
) -> TrackSamplingAltBaseline:
    return TrackSamplingAltBaseline(
        id=baseline.id,
        run_id=baseline.run_id,
        course_key=baseline.course_key,
        reset_variant_key=baseline.reset_variant_key,
        source_entry_id=baseline.source_entry_id,
        label=baseline.label,
        state_path=_stored_path(baseline.state_path),
        weight=baseline.weight,
        enabled=baseline.enabled,
        created_at=baseline.created_at,
        updated_at=baseline.updated_at,
        deleted_at=baseline.deleted_at,
    )


def _alt_baseline_values(baseline: TrackSamplingAltBaseline) -> dict[str, object]:
    return {
        "id": baseline.id,
        "run_id": baseline.run_id,
        "course_key": baseline.course_key,
        "reset_variant_key": baseline.reset_variant_key,
        "source_entry_id": baseline.source_entry_id,
        "label": baseline.label,
        "state_path": str(baseline.state_path),
        "weight": float(baseline.weight),
        "enabled": bool(baseline.enabled),
        "created_at": baseline.created_at,
        "updated_at": baseline.updated_at,
        "deleted_at": baseline.deleted_at,
    }


def _track_sampling_entry_values(
    *,
    run_id: str,
    entry: TrackSamplingRuntimeEntry,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "course_key": entry.course_key,
        "track_id": entry.track_id,
        "label": entry.label,
        "base_weight": entry.base_weight,
        "current_weight": entry.current_weight,
        "completed_frames": entry.completed_frames,
        "episode_count": entry.episode_count,
        "finished_episode_count": entry.finished_episode_count,
        "success_sample_count": entry.success_sample_count,
        "ema_episode_frames": entry.ema_episode_frames,
        "ema_completion_fraction": entry.ema_completion_fraction,
        "generation_episode_count": entry.generation_episode_count,
        "generation_finished_episode_count": entry.generation_finished_episode_count,
        "generation_success_sample_count": entry.generation_success_sample_count,
        "generation_ema_completion_fraction": entry.generation_ema_completion_fraction,
        "generated_course_slot": entry.generated_course_slot,
        "generated_course_generation": entry.generated_course_generation,
        "generated_course_id": entry.generated_course_id,
        "generated_course_name": entry.generated_course_name,
        "generated_course_hash": entry.generated_course_hash,
        "generated_course_seed": _optional_seed_text(entry.generated_course_seed),
        "generated_course_segment_count": entry.generated_course_segment_count,
        "generated_course_length": entry.generated_course_length,
    }


def _track_sampling_artifact_from_model(
    artifact: RunTrackSamplingArtifactModel,
) -> TrackSamplingMaterializedArtifact:
    return TrackSamplingMaterializedArtifact(
        course_key=artifact.course_key,
        reset_variant_key=artifact.reset_variant_key,
        entry_id=artifact.entry_id,
        baseline_state_path=_stored_path(artifact.baseline_state_path),
        metadata_path=_stored_path(artifact.metadata_path),
        source_course_index=artifact.source_course_index,
        source_gp_difficulty=artifact.source_gp_difficulty,
        source_vehicle=artifact.source_vehicle,
        source_engine_setting_raw_value=artifact.source_engine_setting_raw_value,
        generated_course_slot=artifact.generated_course_slot,
        generated_course_generation=artifact.generated_course_generation,
        generated_course_id=artifact.generated_course_id,
        generated_course_name=artifact.generated_course_name,
        generated_course_hash=artifact.generated_course_hash,
        generated_course_seed=_optional_int(artifact.generated_course_seed),
        generated_course_segment_count=artifact.generated_course_segment_count,
        generated_course_length=artifact.generated_course_length,
    )


def _track_sampling_artifact_values(
    *,
    run_id: str,
    artifact: TrackSamplingMaterializedArtifact,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "course_key": artifact.course_key,
        "reset_variant_key": artifact.reset_variant_key,
        "entry_id": artifact.entry_id,
        "baseline_state_path": str(artifact.baseline_state_path),
        "metadata_path": str(artifact.metadata_path),
        "source_course_index": artifact.source_course_index,
        "source_gp_difficulty": artifact.source_gp_difficulty,
        "source_vehicle": artifact.source_vehicle,
        "source_engine_setting_raw_value": artifact.source_engine_setting_raw_value,
        "generated_course_slot": artifact.generated_course_slot,
        "generated_course_generation": artifact.generated_course_generation,
        "generated_course_id": artifact.generated_course_id,
        "generated_course_name": artifact.generated_course_name,
        "generated_course_hash": artifact.generated_course_hash,
        "generated_course_seed": _optional_seed_text(artifact.generated_course_seed),
        "generated_course_segment_count": artifact.generated_course_segment_count,
        "generated_course_length": artifact.generated_course_length,
    }


def _generated_x_cup_slot_from_model(
    slot: RunTrackSamplingGeneratedSlotModel,
) -> GeneratedXCupSlot:
    return GeneratedXCupSlot(
        course_key=slot.course_key,
        slot=slot.slot,
        generation=slot.generation,
        course_id=slot.course_id,
        course_name=slot.course_name,
        course_hash=slot.course_hash,
        course_seed=int(slot.course_seed),
        segment_count=slot.segment_count,
        course_length=slot.course_length,
    )


def _generated_x_cup_slot_values(
    *,
    run_id: str,
    slot: GeneratedXCupSlot,
    updated_at: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "slot": slot.slot,
        "course_key": slot.course_key,
        "generation": slot.generation,
        "course_id": slot.course_id,
        "course_name": slot.course_name,
        "course_hash": slot.course_hash,
        "course_seed": _optional_seed_text(slot.course_seed),
        "segment_count": slot.segment_count,
        "course_length": slot.course_length,
        "updated_at": updated_at,
    }


def _stored_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("stored boolean value cannot be converted to optional int")
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"stored value cannot be converted to optional int: {value!r}")


def _optional_seed_text(value: int | None) -> str | None:
    return None if value is None else str(value)
