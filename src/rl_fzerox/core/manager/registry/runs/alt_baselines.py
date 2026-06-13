# src/rl_fzerox/core/manager/registry/runs/alt_baselines.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from rl_fzerox.core.manager.db.models import RunAltBaselineModel
from rl_fzerox.core.training.session.callbacks.track_sampling.alt_baselines import (
    TrackSamplingAltBaseline,
)

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def get_run_alt_baselines(
    store: ManagerStore,
    run_id: str,
) -> tuple[TrackSamplingAltBaseline, ...]:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        baselines = tuple(
            session.scalars(
                select(RunAltBaselineModel)
                .where(
                    RunAltBaselineModel.run_id == run_id,
                    RunAltBaselineModel.enabled.is_(True),
                    RunAltBaselineModel.deleted_at.is_(None),
                )
                .order_by(
                    RunAltBaselineModel.course_key,
                    RunAltBaselineModel.reset_variant_key,
                    RunAltBaselineModel.created_at,
                    RunAltBaselineModel.id,
                )
            )
        )
    return tuple(_alt_baseline_from_model(baseline) for baseline in baselines)


def active_run_alt_baselines(
    store: ManagerStore,
    run_id: str,
) -> tuple[TrackSamplingAltBaseline, ...]:
    return tuple(baseline for baseline in get_run_alt_baselines(store, run_id) if baseline.active)


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
) -> bool:
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


def clear_run_alt_baselines_for_course(
    store: ManagerStore,
    *,
    run_id: str,
    course_key: str,
) -> int:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        baselines = tuple(
            session.scalars(
                select(RunAltBaselineModel).where(
                    RunAltBaselineModel.run_id == run_id,
                    RunAltBaselineModel.course_key == course_key,
                )
            )
        )
        state_paths = tuple(Path(baseline.state_path) for baseline in baselines)
        for baseline in baselines:
            session.delete(baseline)

    for state_path in state_paths:
        state_path.unlink(missing_ok=True)
    return len(baselines)


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


def _stored_path(value: str) -> Path:
    return Path(value).expanduser().resolve()
