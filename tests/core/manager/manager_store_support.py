# tests/core/manager/manager_store_support.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager import (
    ManagerStore,
)
from rl_fzerox.core.manager.db import manager_engine, manager_session
from rl_fzerox.core.manager.db.models import (
    ConfigSnapshotModel,
    FilesystemOperationModel,
    RunDraftModel,
    RunModel,
    RunWorkerModel,
)
from rl_fzerox.core.save_game.unlocks import FZEROX_SAVE_LAYOUT
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingMaterializedArtifact,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.artifacts import reset_variant_key

SnapshotKind = Literal["run", "draft", "template", "import"]


def _insert_stale_draft_config(
    store: ManagerStore,
    *,
    draft_id: str,
    name: str,
    config: dict[str, object],
) -> None:
    created_at = "2026-05-01T00:00:00+00:00"
    with manager_session(store.db_path) as session:
        snapshot_id = _insert_config_snapshot(
            session=session,
            snapshot_id=f"cfg_test_{draft_id}",
            kind="draft",
            raw_config_json=json.dumps(config),
            stored_config_hash="stale",
            created_at=created_at,
        )
        session.add(
            RunDraftModel(
                id=draft_id,
                name=name,
                config_snapshot_id=snapshot_id,
                source_run_id=None,
                source_artifact=None,
                source_snapshot_dir=None,
                source_num_timesteps=None,
                created_at=created_at,
                updated_at=created_at,
            )
        )


def _insert_config_snapshot(
    *,
    session: Session,
    snapshot_id: str,
    kind: SnapshotKind,
    raw_config_json: str,
    stored_config_hash: str,
    created_at: str,
) -> str:
    session.add(
        ConfigSnapshotModel(
            id=snapshot_id,
            kind=kind,
            schema_version=1,
            created_at=created_at,
            config_json=raw_config_json,
            config_hash=stored_config_hash,
        )
    )
    return snapshot_id


def _config_snapshot_json(store: ManagerStore, snapshot_id: str) -> str:
    with manager_session(store.db_path) as session:
        snapshot = session.get(ConfigSnapshotModel, snapshot_id)
        assert snapshot is not None
        return snapshot.config_json


def _table_columns(store: ManagerStore, table_name: str) -> set[str]:
    engine = manager_engine(store.db_path)
    try:
        return {str(column["name"]) for column in inspect(engine).get_columns(table_name)}
    finally:
        engine.dispose()


def _filesystem_operation_count(store: ManagerStore, *, kind: str | None = None) -> int:
    with manager_session(store.db_path) as session:
        statement = select(FilesystemOperationModel)
        if kind is not None:
            statement = statement.where(FilesystemOperationModel.kind == kind)
        return len(tuple(session.scalars(statement)))


def _set_worker_heartbeat(store: ManagerStore, *, run_id: str, heartbeat_at: str) -> None:
    with manager_session(store.db_path) as session:
        worker = session.get(RunWorkerModel, run_id)
        assert worker is not None
        worker.heartbeat_at = heartbeat_at


def _worker_exists(store: ManagerStore, run_id: str) -> bool:
    with manager_session(store.db_path) as session:
        return session.get(RunWorkerModel, run_id) is not None


def _stored_run_dir(store: ManagerStore, run_id: str) -> Path:
    with manager_session(store.db_path) as session:
        run = session.get(RunModel, run_id)
        assert run is not None
        return Path(run.run_dir)


def _write_policy_artifact(run_dir: Path, artifact: Literal["latest", "best"]) -> Path:
    policy_path = run_dir / "checkpoints" / artifact / "policy.zip"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_bytes(b"fake policy checkpoint")
    return policy_path


def _track_sampling_artifact(
    baseline_path: Path,
    *,
    difficulty: str,
    course_slot: int = 1,
    course_generation: int = 3,
) -> TrackSamplingMaterializedArtifact:
    return TrackSamplingMaterializedArtifact(
        course_key="x_cup_slot_1",
        reset_variant_key=reset_variant_key(
            mode="gp_race",
            gp_difficulty=difficulty,
            vehicle="blue_falcon",
        ),
        entry_id=f"x_cup_1234abcd_gp_race_{difficulty}_blue_falcon",
        baseline_state_path=baseline_path.expanduser().resolve(),
        metadata_path=baseline_path.with_suffix(".json").expanduser().resolve(),
        source_course_index=48,
        source_gp_difficulty=difficulty,
        source_vehicle="blue_falcon",
        source_engine_setting_raw_value=50,
        generated_course_slot=course_slot,
        generated_course_generation=course_generation,
        generated_course_id="x_cup_1234abcd",
        generated_course_name="X Cup 1234abcd",
        generated_course_hash="1234abcd",
        generated_course_seed=1234,
        generated_course_segment_count=38,
        generated_course_length=61_743.98046875,
    )


def _logical_sra(cup_progress: dict[str, int]) -> bytes:
    payload = bytearray(FZEROX_SAVE_LAYOUT.raw_sra_size)
    payload[: len(FZEROX_SAVE_LAYOUT.title)] = FZEROX_SAVE_LAYOUT.title
    for progress_offset in FZEROX_SAVE_LAYOUT.gp_progress_offsets:
        payload[progress_offset.offset] = cup_progress.get(progress_offset.cup_id, 0)
    return bytes(payload)
