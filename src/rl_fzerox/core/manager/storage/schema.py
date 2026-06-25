# src/rl_fzerox/core/manager/storage/schema.py
"""Current-schema bootstrap for the SQLite-backed manager registry."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import DeclarativeBase, Session

from rl_fzerox.core.manager.db import manager_session
from rl_fzerox.core.manager.db.models import (
    EvaluationBaselineSuiteModel,
    EvaluationModel,
    EvaluationPresetModel,
    ManagerBase,
    RunAltBaselineModel,
    RunTrackSamplingArtifactModel,
    RunTrackSamplingEntryModel,
    RunTrackSamplingGeneratedSlotModel,
    RunTrackSamplingRuntimeModel,
    SaveGameAttemptModel,
    SaveGameCourseSetupModel,
    SaveGameCupSetupModel,
    SaveGameModel,
    SchemaVersionModel,
)
from rl_fzerox.core.manager.db.repositories.configs import create_config_snapshot
from rl_fzerox.core.manager.db.repositories.runs import upsert_template
from rl_fzerox.core.manager.db.session import manager_engine
from rl_fzerox.core.manager.run_spec import default_managed_run_config
from rl_fzerox.core.manager.storage.serialization import config_hash

SCHEMA_VERSION = 40

CONFIG_OWNER_TABLES = ("runs", "run_drafts", "run_templates")
SAVE_GAME_CHILD_TABLES = (
    "save_game_attempts",
    "save_game_course_setups",
    "save_game_cup_setups",
)
RUN_CHILD_TABLES = (
    "run_alt_baselines",
    "run_commands",
    "run_events",
    "run_runtime",
    "run_track_sampling_artifacts",
    "run_track_sampling_entries",
    "run_track_sampling_generated_slots",
    "run_track_sampling_runtime",
    "run_workers",
)
MANAGER_RUNTIME_TABLES = ("viewer_leases",)
EVALUATION_TABLES = (
    "evaluations",
    "evaluation_presets",
    "evaluation_baseline_suites",
)
TRACK_SAMPLING_ENTRY_LEGACY_COLUMNS = frozenset(
    {
        "generated_entry_id",
        "generated_baseline_state_path",
    }
)


def initialize_manager_schema(db_path: Path, *, applied_at: str) -> None:
    """Create manager tables and seed the built-in first template."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = manager_engine(db_path)
    try:
        table_names = set(inspect(engine).get_table_names())
        if _has_manager_schema(table_names):
            _assert_current_schema(engine=engine, table_names=table_names)
        else:
            ManagerBase.metadata.create_all(engine)
    finally:
        engine.dispose()

    with manager_session(db_path) as session:
        for version in session.scalars(select(SchemaVersionModel)):
            session.delete(version)
        session.add(SchemaVersionModel(version=SCHEMA_VERSION, applied_at=applied_at))
        _refresh_default_evaluation_presets(session=session, updated_at=applied_at)
        _refresh_default_template(session=session, updated_at=applied_at)


def refresh_default_template(db_path: Path, *, updated_at: str) -> None:
    """Refresh the built-in template without re-running schema creation."""

    with manager_session(db_path) as session:
        _refresh_default_template(session=session, updated_at=updated_at)


def _refresh_default_template(
    *,
    session: Session,
    updated_at: str,
) -> None:
    config = default_managed_run_config()
    snapshot = create_config_snapshot(
        session,
        kind="template",
        config=config,
        created_at=updated_at,
        snapshot_id=f"cfg_template_all_cups_recurrent_ppo_{config_hash(config)[:12]}",
    )
    upsert_template(
        session,
        template_id="all_cups_recurrent_ppo",
        name="All cups recurrent PPO",
        config_snapshot_id=snapshot.id,
        created_at=updated_at,
        updated_at=updated_at,
    )


def _refresh_default_evaluation_presets(
    *,
    session: Session,
    updated_at: str,
) -> None:
    from rl_fzerox.core.manager.db.repositories.evaluations import (
        upsert_default_evaluation_presets,
    )

    upsert_default_evaluation_presets(session, now=updated_at)


def _has_manager_schema(table_names: set[str]) -> bool:
    return bool(
        table_names
        & {
            "schema_version",
            "runs",
            "run_drafts",
            "run_templates",
            "run_runtime",
        }
    )


def _assert_current_schema(
    *,
    engine: Engine,
    table_names: set[str],
) -> None:
    inspector = inspect(engine)
    for table_name in CONFIG_OWNER_TABLES:
        _assert_table_present(table_names=table_names, table_name=table_name)
        columns = _table_columns(inspector=inspector, table_name=table_name)
        if "config_json" in columns or "config_hash" in columns:
            raise RuntimeError(
                "manager DB is not current: config JSON must be stored in config_snapshots only"
            )
        _assert_required_columns(
            table_name=table_name,
            columns=columns,
            required_columns={"config_snapshot_id"},
        )
    _assert_table_present(table_names=table_names, table_name="config_snapshots")
    _assert_table_present(table_names=table_names, table_name="save_games")
    _assert_save_game_columns(inspector=inspector)
    _assert_tables_present(table_names=table_names, required_tables=SAVE_GAME_CHILD_TABLES)
    _assert_tables_present(table_names=table_names, required_tables=MANAGER_RUNTIME_TABLES)
    _assert_tables_present(table_names=table_names, required_tables=EVALUATION_TABLES)
    _assert_evaluation_columns(inspector=inspector)
    _assert_save_game_child_columns(inspector=inspector)
    _assert_run_foreign_keys(inspector=inspector, table_names=table_names)
    _assert_track_sampling_artifact_columns(inspector=inspector)
    _assert_alt_baseline_columns(inspector=inspector)
    _assert_track_sampling_runtime_columns(inspector=inspector)
    _assert_track_sampling_entry_columns(inspector=inspector)
    _assert_track_sampling_generated_slot_columns(inspector=inspector)


def _assert_save_game_columns(*, inspector: Inspector) -> None:
    _assert_model_columns(inspector=inspector, model=SaveGameModel)


def _assert_evaluation_columns(*, inspector: Inspector) -> None:
    for model in (EvaluationModel, EvaluationPresetModel, EvaluationBaselineSuiteModel):
        table_name = model.__table__.name
        columns = _table_columns(inspector=inspector, table_name=table_name)
        legacy_preset_columns = {"config_json", "source_artifact"}
        legacy_columns = legacy_preset_columns.intersection(columns)
        if table_name == "evaluation_presets" and legacy_columns:
            _raise_legacy_columns_error(
                subject="evaluation presets",
                legacy_columns=legacy_columns,
                verb="have",
            )
        _assert_required_columns(
            table_name=table_name,
            columns=columns,
            required_columns=_required_column_names(model),
        )


def _assert_save_game_child_columns(*, inspector: Inspector) -> None:
    for model in (SaveGameCourseSetupModel, SaveGameCupSetupModel, SaveGameAttemptModel):
        _assert_model_columns(inspector=inspector, model=model)


def _assert_track_sampling_entry_columns(*, inspector: Inspector) -> None:
    table_name = "run_track_sampling_entries"
    columns = _table_columns(inspector=inspector, table_name=table_name)
    legacy_columns = columns.intersection(TRACK_SAMPLING_ENTRY_LEGACY_COLUMNS)
    if legacy_columns:
        _raise_legacy_columns_error(
            subject=table_name,
            legacy_columns=legacy_columns,
        )
    _assert_required_columns(
        table_name=table_name,
        columns=columns,
        required_columns=_required_column_names(RunTrackSamplingEntryModel),
    )


def _assert_track_sampling_runtime_columns(*, inspector: Inspector) -> None:
    _assert_model_columns(inspector=inspector, model=RunTrackSamplingRuntimeModel)


def _required_column_names(model: type[DeclarativeBase]) -> set[str]:
    return {column.name for column in model.__table__.columns}


def _assert_track_sampling_artifact_columns(*, inspector: Inspector) -> None:
    _assert_model_columns(inspector=inspector, model=RunTrackSamplingArtifactModel)


def _assert_alt_baseline_columns(*, inspector: Inspector) -> None:
    _assert_model_columns(inspector=inspector, model=RunAltBaselineModel)


def _assert_track_sampling_generated_slot_columns(*, inspector: Inspector) -> None:
    _assert_model_columns(inspector=inspector, model=RunTrackSamplingGeneratedSlotModel)


def _assert_table_present(*, table_names: set[str], table_name: str) -> None:
    if table_name not in table_names:
        raise RuntimeError(f"manager DB is not current: missing {table_name}")


def _assert_tables_present(
    *,
    table_names: set[str],
    required_tables: tuple[str, ...],
) -> None:
    for table_name in required_tables:
        _assert_table_present(table_names=table_names, table_name=table_name)


def _table_columns(*, inspector: Inspector, table_name: str) -> set[str]:
    return {str(column["name"]) for column in inspector.get_columns(table_name)}


def _assert_model_columns(*, inspector: Inspector, model: type[DeclarativeBase]) -> None:
    table_name = model.__table__.name
    _assert_required_columns(
        table_name=table_name,
        columns=_table_columns(inspector=inspector, table_name=table_name),
        required_columns=_required_column_names(model),
    )


def _assert_required_columns(
    *,
    table_name: str,
    columns: set[str],
    required_columns: set[str],
) -> None:
    missing_columns = required_columns.difference(columns)
    if missing_columns:
        joined_columns = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"manager DB is not current: {table_name} is missing {joined_columns}")


def _raise_legacy_columns_error(
    *,
    subject: str,
    legacy_columns: set[str],
    verb: str = "has",
) -> None:
    joined_columns = ", ".join(sorted(legacy_columns))
    raise RuntimeError(
        f"manager DB is not current: {subject} {verb} legacy columns {joined_columns}"
    )


def _assert_run_foreign_keys(
    *,
    inspector: Inspector,
    table_names: set[str],
) -> None:
    for table_name in RUN_CHILD_TABLES:
        if table_name not in table_names:
            raise RuntimeError(f"manager DB is not current: missing {table_name}")
        for foreign_key in inspector.get_foreign_keys(table_name):
            if "run_id" not in foreign_key.get("constrained_columns", ()):
                continue
            if foreign_key.get("referred_table") != "runs":
                raise RuntimeError(
                    f"manager DB is not current: {table_name}.run_id must reference runs.id"
                )
