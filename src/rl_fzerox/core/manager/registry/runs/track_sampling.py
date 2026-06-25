# src/rl_fzerox/core/manager/registry/runs/track_sampling.py
"""SQLite persistence for runtime track-sampling state."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from rl_fzerox.core.manager.db.models import (
    RunTrackSamplingArtifactModel,
    RunTrackSamplingEntryModel,
    RunTrackSamplingGeneratedSlotModel,
    RunTrackSamplingRuntimeModel,
)
from rl_fzerox.core.manager.registry.common import utc_now
from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
from rl_fzerox.core.training.session.callbacks.track_sampling.artifacts import (
    TrackSamplingMaterializedArtifact,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.persistence import (
    deficit_budget_scheduler_state_json,
    load_deficit_budget_scheduler_state_json,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


# These fields use the same names and runtime types in ORM rows and domain
# dataclasses. Seeds, paths, and scheduler JSON stay explicit because their
# persisted representation intentionally differs from the domain object.
_RUNTIME_STATE_FIELDS = (
    "sampling_mode",
    "action_repeat",
    "update_episodes",
    "ema_alpha",
    "max_weight_scale",
    "adaptive_completion_weight",
    "adaptive_target_completion",
    "adaptive_min_confidence_episodes",
    "adaptive_confidence_scale",
    "deficit_budget_difficulty_metric",
    "deficit_budget_warmup_min_episodes_per_course",
    "update_count",
    "episodes_since_update",
)

_ENTRY_MIRROR_FIELDS = (
    "track_id",
    "course_key",
    "label",
    "base_weight",
    "current_weight",
    "completed_frames",
    "episode_count",
    "finished_episode_count",
    "success_sample_count",
    "completion_sample_count",
    "completion_fraction_total",
    "ema_episode_frames",
    "ema_completion_fraction",
    "ema_finish_rate",
    "current_problem_score",
    "generation_episode_count",
    "generation_finished_episode_count",
    "generation_success_sample_count",
    "generation_ema_completion_fraction",
)

_GENERATED_COURSE_MIRROR_FIELDS = (
    "generated_course_slot",
    "generated_course_generation",
    "generated_course_id",
    "generated_course_name",
    "generated_course_hash",
    "generated_course_segment_count",
    "generated_course_length",
)

_ARTIFACT_MIRROR_FIELDS = (
    "course_key",
    "reset_variant_key",
    "entry_id",
    "source_course_index",
    "source_gp_difficulty",
    "source_vehicle",
    "source_engine_setting_raw_value",
)

_GENERATED_SLOT_MIRROR_FIELDS = (
    "course_key",
    "slot",
    "generation",
    "course_id",
    "course_name",
    "course_hash",
    "segment_count",
    "course_length",
)


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
        **_values_from_attrs(runtime, _RUNTIME_STATE_FIELDS),
        entries=tuple(_track_sampling_entry_from_model(entry) for entry in entries),
        deficit_budget_scheduler=load_deficit_budget_scheduler_state_json(
            runtime.deficit_budget_scheduler_json,
        ),
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
        **_values_from_attrs(state, _RUNTIME_STATE_FIELDS),
        "deficit_budget_scheduler_json": deficit_budget_scheduler_state_json(
            state.deficit_budget_scheduler,
        ),
        "updated_at": updated_at,
    }


def _track_sampling_entry_from_model(
    entry: RunTrackSamplingEntryModel,
) -> TrackSamplingRuntimeEntry:
    return TrackSamplingRuntimeEntry(
        **_values_from_attrs(entry, (*_ENTRY_MIRROR_FIELDS, *_GENERATED_COURSE_MIRROR_FIELDS)),
        generated_course_seed=_optional_int(entry.generated_course_seed),
    )


def _track_sampling_entry_values(
    *,
    run_id: str,
    entry: TrackSamplingRuntimeEntry,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        **_values_from_attrs(entry, (*_ENTRY_MIRROR_FIELDS, *_GENERATED_COURSE_MIRROR_FIELDS)),
        "generated_course_seed": _optional_seed_text(entry.generated_course_seed),
    }


def _track_sampling_artifact_from_model(
    artifact: RunTrackSamplingArtifactModel,
) -> TrackSamplingMaterializedArtifact:
    return TrackSamplingMaterializedArtifact(
        **_values_from_attrs(artifact, _ARTIFACT_MIRROR_FIELDS),
        baseline_state_path=_stored_path(artifact.baseline_state_path),
        metadata_path=_stored_path(artifact.metadata_path),
        **_values_from_attrs(artifact, _GENERATED_COURSE_MIRROR_FIELDS),
        generated_course_seed=_optional_int(artifact.generated_course_seed),
    )


def _track_sampling_artifact_values(
    *,
    run_id: str,
    artifact: TrackSamplingMaterializedArtifact,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        **_values_from_attrs(artifact, _ARTIFACT_MIRROR_FIELDS),
        "baseline_state_path": str(artifact.baseline_state_path),
        "metadata_path": str(artifact.metadata_path),
        **_values_from_attrs(artifact, _GENERATED_COURSE_MIRROR_FIELDS),
        "generated_course_seed": _optional_seed_text(artifact.generated_course_seed),
    }


def _generated_x_cup_slot_from_model(
    slot: RunTrackSamplingGeneratedSlotModel,
) -> GeneratedXCupSlot:
    return GeneratedXCupSlot(
        **_values_from_attrs(slot, _GENERATED_SLOT_MIRROR_FIELDS),
        course_seed=int(slot.course_seed),
    )


def _generated_x_cup_slot_values(
    *,
    run_id: str,
    slot: GeneratedXCupSlot,
    updated_at: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        **_values_from_attrs(slot, _GENERATED_SLOT_MIRROR_FIELDS),
        "course_seed": _optional_seed_text(slot.course_seed),
        "updated_at": updated_at,
    }


def _stored_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _values_from_attrs(source: object, fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: getattr(source, field) for field in fields}


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
