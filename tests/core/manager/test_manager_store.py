# tests/core/manager/test_manager_store.py
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import rl_fzerox.core.manager.artifacts.filesystem as filesystem_ops_module
import rl_fzerox.core.manager.registry.drafts.fork_sources as draft_fork_sources
import rl_fzerox.core.manager.registry.paths as registry_paths
import rl_fzerox.core.manager.registry.runs.maintenance as run_maintenance
import rl_fzerox.core.manager.store as store_module
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config, new_managed_run_id
from rl_fzerox.core.manager.artifacts.filesystem import FilesystemOperation


def test_manager_store_seeds_default_template(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")

    templates = store.list_templates()

    assert len(templates) == 1
    assert templates[0].id == "all_cups_recurrent_ppo"
    assert templates[0].config == default_managed_run_config()


def test_manager_store_initializes_schema_only_once_per_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0
    original_initialize_schema = store_module.initialize_manager_schema

    def wrapped_initialize_schema(connection: sqlite3.Connection, *, applied_at: str) -> None:
        nonlocal call_count
        call_count += 1
        original_initialize_schema(connection, applied_at=applied_at)

    monkeypatch.setattr(store_module, "initialize_manager_schema", wrapped_initialize_schema)

    store = ManagerStore(tmp_path / "runs.db")

    assert store.pending_run_command("missing-run") is None
    assert store.pending_run_command("missing-run") is None

    assert call_count == 1


def test_manager_store_refreshes_system_template_to_current_defaults(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")
    store.initialize()

    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["reward"]["energy_refill_progress_multiplier"] = 1.0

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            UPDATE run_templates
            SET config_json = ?, config_hash = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                json.dumps(stale_config),
                "stale",
                "2026-05-01T00:00:00+00:00",
                "all_cups_recurrent_ppo",
            ),
        )

    template = store.default_template()

    assert template.config == default_managed_run_config()
    assert template.config.reward.energy_refill_progress_multiplier == 3.0


def test_manager_store_saves_draft_without_filesystem_artifacts(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config().model_copy(update={"seed": 321})

    draft = store.create_draft(
        name="Prototype Run",
        config=config,
    )

    drafts = store.list_drafts()
    assert len(drafts) == 1
    assert drafts[0].id == draft.id
    assert drafts[0].name == "Prototype Run"
    assert drafts[0].config == config
    assert not (tmp_path / "managed_runs").exists()


def test_manager_store_pins_and_cleans_fork_draft_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    source_run = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "parent-run",
    )

    def fake_snapshot_fork_source(
        *,
        source_run_dir: Path,
        artifact: str,
        destination_dir: Path,
    ) -> int:
        assert source_run_dir == source_run.run_dir
        assert artifact == "latest"
        destination_dir.mkdir(parents=True, exist_ok=True)
        return 123_456

    monkeypatch.setattr(
        draft_fork_sources,
        "snapshot_fork_source",
        fake_snapshot_fork_source,
    )

    draft = store.create_draft(
        name="Pinned Fork",
        config=config,
        source_run_id=source_run.id,
        source_artifact="latest",
    )

    assert draft.source_snapshot_dir is not None
    assert draft.source_snapshot_dir.is_dir()
    assert draft.source_num_timesteps == 123_456

    assert store.delete_draft(draft.id)
    assert not draft.source_snapshot_dir.exists()


def test_manager_store_persists_run_lineage_metadata(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "parent-run",
    )

    run = store.create_run(
        run_id="child-run",
        name="Child Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "child-run",
        lineage_step_offset=816_040,
        parent_run_id="parent-run",
        source_run_id="parent-run",
        source_artifact="best",
        source_num_timesteps=816_040,
    )

    loaded = store.get_run(run.id)

    assert loaded is not None
    assert loaded.lineage_id == "parent-run"
    assert loaded.lineage_step_offset == 816_040
    assert loaded.parent_run_id == "parent-run"
    assert loaded.source_run_id == "parent-run"
    assert loaded.source_artifact == "best"
    assert loaded.source_num_timesteps == 816_040


def test_manager_store_deletes_draft(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    draft = store.create_draft(
        name="Delete Me",
        config=default_managed_run_config(),
    )

    assert store.delete_draft(draft.id)

    assert store.list_drafts() == ()


def test_manager_store_normalizes_stale_draft_configs(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["reward"] = {"manual_boost_reward": 0.5}

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            INSERT INTO run_drafts(
                id,
                name,
                config_json,
                config_hash,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "old-draft",
                "Old Draft",
                json.dumps(stale_config),
                "stale",
                "2026-05-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
            ),
        )

    draft = store.list_drafts()[0]

    assert draft.config.reward.manual_boost_reward == 0.5
    assert draft.config.reward.time_penalty_per_frame == 0.0
    assert draft.config.reward.step_reward_clip_max == 100.0


def test_manager_store_creates_current_runs_schema(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()

    with sqlite3.connect(store.db_path) as connection:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(runs)").fetchall()}

    assert columns == {
        "id",
        "name",
        "status",
        "config_json",
        "config_hash",
        "run_dir",
        "lineage_id",
        "lineage_step_offset",
        "parent_run_id",
        "source_run_id",
        "source_artifact",
        "source_snapshot_dir",
        "source_num_timesteps",
        "created_at",
        "started_at",
        "stopped_at",
    }


def test_manager_store_rejects_legacy_observation_fields(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["observation"] = {
        "frame_stack": 2,
        "minimap_layer": False,
        "preset": "crop_60x76",
        "progress_source": "segment_progress",
        "stack_mode": "rgb",
        "zero_edge_ratio": True,
        "zero_outside_track_bounds": True,
    }

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            INSERT INTO run_drafts(
                id,
                name,
                config_json,
                config_hash,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "old-observation",
                "Old Observation",
                json.dumps(stale_config),
                "stale",
                "2026-05-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
            ),
        )

    with pytest.raises(ValidationError):
        store.list_drafts()


def test_manager_store_rejects_legacy_state_modes(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["observation"]["state_components"] = [
        {"name": "vehicle_state", "mode": "include"},
        {"name": "track_position", "mode": "zero", "progress_source": "segment_progress"},
        {"name": "course_context", "mode": "exclude", "encoding": "one_hot_builtin"},
    ]
    stale_config["observation"]["state_feature_modes"] = [
        {"name": "track_position.segment_progress", "mode": "include", "dropout_prob": 0.25},
        {"name": "track_position.edge_ratio", "mode": "exclude", "dropout_prob": 0.0},
    ]

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            INSERT INTO run_drafts(
                id,
                name,
                config_json,
                config_hash,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-state-modes",
                "Legacy State Modes",
                json.dumps(stale_config),
                "stale",
                "2026-05-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
            ),
        )

    with pytest.raises(ValidationError):
        store.list_drafts()


def test_manager_store_rejects_legacy_vehicle_fields(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["vehicle"] = {
        "vehicle_id": "golden_fox",
        "engine_setting_raw_value": 65,
    }

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            INSERT INTO run_drafts(
                id,
                name,
                config_json,
                config_hash,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "old-vehicle",
                "Old Vehicle",
                json.dumps(stale_config),
                "stale",
                "2026-05-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
            ),
        )

    with pytest.raises(ValidationError):
        store.list_drafts()


def test_manager_store_normalizes_missing_action_config(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config.pop("action", None)

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            INSERT INTO run_drafts(
                id,
                name,
                config_json,
                config_hash,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "old-action",
                "Old Action",
                json.dumps(stale_config),
                "stale",
                "2026-05-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
            ),
        )

    draft = store.list_drafts()[0]

    assert draft.config.action.action_repeat == 2
    assert draft.config.action.steering_mode == "continuous"
    assert draft.config.action.drive_mode == "on_off"
    assert draft.config.action.force_full_throttle is False
    assert draft.config.action.include_air_brake is True
    assert draft.config.action.enable_air_brake is True
    assert draft.config.action.boost_unmask_max_speed_kph is None
    assert draft.config.action.boost_min_energy_fraction == 0.1
    assert draft.config.action.lean_output_mode == "three_way"
    assert draft.config.action.lean_mode == "release_cooldown"
    assert draft.config.action.lean_unmask_min_speed_kph is None
    assert draft.config.action.lean_initial_lockout_frames == 0
    assert draft.config.action.include_pitch is True
    assert draft.config.action.enable_pitch is True
    assert draft.config.action.pitch_mode == "discrete"
    assert draft.config.action.pitch_buckets == 5


def test_manager_store_rejects_legacy_progress_suspend_field(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["reward"]["suspend_progress_while_outside_track_bounds"] = False
    stale_config["reward"].pop("suspend_progress_while_outside_track_bounds")
    stale_config["reward"]["suspend_progress_while_airborne"] = True

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            INSERT INTO run_drafts(
                id,
                name,
                config_json,
                config_hash,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "old-progress-suspend",
                "Old Progress Suspend",
                json.dumps(stale_config),
                "stale",
                "2026-05-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
            ),
        )

    with pytest.raises(ValidationError):
        store.list_drafts()


def test_manager_store_creates_run_record_without_filesystem_artifacts(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()

    run = store.create_run(
        name="Started Later",
        config=config,
        managed_runs_root=tmp_path / "managed_runs",
    )

    assert store.list_runs()[0].id == run.id
    assert not run.run_dir.exists()


def test_manager_store_supports_explicit_run_dir_and_status_updates(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run_id = new_managed_run_id("Launch Me")
    run_dir = tmp_path / "runs" / f"{run_id}_0001"
    run_dir.mkdir(parents=True)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    run = store.create_run(
        run_id=run_id,
        name="Launch Me",
        config=default_managed_run_config(),
        explicit_run_dir=run_dir,
    )
    launched = store.update_run_status(
        run_id=run.id,
        status="running",
        started_at=started_at,
        stopped_at=None,
        message="worker launched",
    )

    assert launched is not None
    assert launched.run_dir == run_dir.resolve()
    assert launched.status == "running"
    assert launched.started_at == started_at
    assert store.get_run(run.id) == launched


def test_manager_store_visible_runs_exclude_created_records(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.create_run(
        name="Created Only",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "managed_runs",
    )

    assert store.list_visible_runs() == ()


def test_manager_store_allows_draft_name_used_by_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    store.create_run(name="Shared Name", config=config, managed_runs_root=tmp_path / "managed_runs")
    draft = store.create_draft(name="Shared Name", config=config)

    assert draft.name == "Shared Name"


def test_manager_store_allows_run_name_used_by_draft(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    store.create_draft(name="Shared Name", config=config)
    run = store.create_run(
        name="Shared Name",
        config=config,
        managed_runs_root=tmp_path / "managed_runs",
    )

    assert run.name == "Shared Name"


def test_manager_store_renames_run_without_mutating_config(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Old Name",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )

    renamed = store.update_run_name(run_id=run.id, name="New Name")

    assert renamed is not None
    assert renamed.name == "New Name"
    assert renamed.config == run.config


def test_manager_store_allows_duplicate_run_names(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    first = store.create_run(name="Shared Run", config=config, managed_runs_root=tmp_path / "runs")
    second = store.create_run(
        name="Shared Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
    )

    assert first.name == "Shared Run"
    assert second.name == "Shared Run"
    assert first.id != second.id


def test_manager_store_allows_renaming_run_to_existing_run_name(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    store.create_run(name="Existing Run", config=config, managed_runs_root=tmp_path / "runs")
    target = store.create_run(name="Target Run", config=config, managed_runs_root=tmp_path / "runs")

    renamed = store.update_run_name(run_id=target.id, name="Existing Run")

    assert renamed is not None
    assert renamed.name == "Existing Run"


def test_manager_store_persists_runtime_snapshots_and_metric_history(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Runtime Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    launched = store.update_run_status(
        run_id=run.id,
        status="running",
        started_at="2026-05-03T12:00:00+00:00",
        stopped_at=None,
        message="worker launched",
    )

    assert launched is not None

    store.upsert_run_runtime(
        run_id=run.id,
        total_timesteps=50_000_000,
        num_timesteps=125_000,
        progress_fraction=0.0025,
        updated_at="2026-05-03T12:05:00+00:00",
        fps=987.0,
        episode_reward_mean=4.2,
        episode_length_mean=512.0,
        approx_kl=0.014,
    )
    commanded = store.request_run_command(run_id=run.id, command="pause")

    assert commanded is not None
    assert commanded.pending_command == "pause"
    assert commanded.runtime is not None
    assert commanded.runtime.num_timesteps == 125_000
    assert commanded.runtime.fps == 987.0
    assert store.pending_run_command(run.id) == "pause"


def test_manager_store_deletes_non_running_runs_with_runtime_rows(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Delete Runtime Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    stopped = store.update_run_status(
        run_id=run.id,
        status="stopped",
        started_at="2026-05-03T12:00:00+00:00",
        stopped_at="2026-05-03T12:30:00+00:00",
        message="worker stopped",
    )

    assert stopped is not None

    store.upsert_run_runtime(
        run_id=run.id,
        total_timesteps=1_000,
        num_timesteps=500,
        progress_fraction=0.5,
        updated_at="2026-05-03T12:10:00+00:00",
        fps=321.0,
    )

    assert store.delete_run(run.id) is True
    assert store.get_run(run.id) is None


def test_manager_store_delete_run_defers_failed_filesystem_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Delete Later",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    run.run_dir.mkdir(parents=True)
    original_apply = filesystem_ops_module.apply_filesystem_operation

    def fail_delete(operation: FilesystemOperation) -> bool:
        raise OSError("filesystem busy")

    monkeypatch.setattr(filesystem_ops_module, "apply_filesystem_operation", fail_delete)

    assert store.delete_run(run.id) is True
    assert store.get_run(run.id) is None
    assert run.run_dir.exists()
    with sqlite3.connect(store.db_path) as connection:
        pending = connection.execute(
            "SELECT COUNT(*) FROM filesystem_operations",
        ).fetchone()
    assert pending is not None
    assert pending[0] == 1

    monkeypatch.setattr(filesystem_ops_module, "apply_filesystem_operation", original_apply)
    recovered = ManagerStore(store.db_path)
    recovered.initialize()

    assert not run.run_dir.exists()
    with sqlite3.connect(store.db_path) as connection:
        pending = connection.execute(
            "SELECT COUNT(*) FROM filesystem_operations",
        ).fetchone()
    assert pending is not None
    assert pending[0] == 0


def test_manager_store_reconciles_stale_dead_worker_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Lease Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    started_at = "2026-05-03T12:00:00+00:00"
    launched = store.update_run_status(
        run_id=run.id,
        status="running",
        started_at=started_at,
        stopped_at=None,
        message="worker launched",
    )

    assert launched is not None
    assert store.register_run_worker(
        run_id=run.id,
        launch_token="token-1",
        pid=12345,
        launched_at=started_at,
    )
    stale_heartbeat = (datetime.now(UTC) - timedelta(minutes=5)).isoformat(timespec="seconds")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE run_workers SET heartbeat_at = ? WHERE run_id = ?",
            (stale_heartbeat, run.id),
        )
    monkeypatch.setattr(run_maintenance, "pid_exists", lambda pid: False)

    store.reconcile_orphaned_runs()
    failed = store.get_run(run.id)

    assert failed is not None
    assert failed.status == "failed"
    with sqlite3.connect(store.db_path) as connection:
        worker_row = connection.execute(
            "SELECT 1 FROM run_workers WHERE run_id = ?",
            (run.id,),
        ).fetchone()
    assert worker_row is None


def test_manager_store_keeps_running_run_when_worker_pid_is_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Live Lease Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    started_at = "2026-05-03T12:00:00+00:00"
    launched = store.update_run_status(
        run_id=run.id,
        status="running",
        started_at=started_at,
        stopped_at=None,
        message="worker launched",
    )

    assert launched is not None
    assert store.register_run_worker(
        run_id=run.id,
        launch_token="token-1",
        pid=12345,
        launched_at=started_at,
    )
    stale_heartbeat = (datetime.now(UTC) - timedelta(minutes=5)).isoformat(timespec="seconds")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE run_workers SET heartbeat_at = ? WHERE run_id = ?",
            (stale_heartbeat, run.id),
        )
    monkeypatch.setattr(run_maintenance, "pid_exists", lambda pid: True)

    store.reconcile_orphaned_runs()
    refreshed = store.get_run(run.id)

    assert refreshed is not None
    assert refreshed.status == "running"


def test_manager_store_exposes_worker_heartbeat_on_runs(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Heartbeat Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    launched_at = datetime.now(UTC).isoformat(timespec="seconds")
    launched = store.update_run_status(
        run_id=run.id,
        status="running",
        started_at=launched_at,
        stopped_at=None,
        message="worker launched",
    )

    assert launched is not None
    assert store.register_run_worker(
        run_id=run.id,
        launch_token="token-1",
        pid=12345,
        launched_at=launched_at,
    )
    heartbeat_at = datetime.now(UTC).isoformat(timespec="seconds")
    assert store.heartbeat_run_worker(
        run_id=run.id,
        launch_token="token-1",
        heartbeat_at=heartbeat_at,
    )

    refreshed = store.get_run(run.id)

    assert refreshed is not None
    assert refreshed.worker_heartbeat_at == heartbeat_at


def test_manager_store_skips_running_runs_without_worker_lease(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Legacy Running Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    launched = store.update_run_status(
        run_id=run.id,
        status="running",
        started_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(timespec="seconds"),
        stopped_at=None,
        message="worker launched",
    )

    assert launched is not None

    store.reconcile_orphaned_runs()
    refreshed = store.get_run(run.id)

    assert refreshed is not None
    assert refreshed.status == "running"


def test_manager_store_rejects_deleting_non_leaf_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
    )
    store.create_run(
        run_id="child-run",
        name="Child Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
        lineage_id=parent.lineage_id,
        parent_run_id=parent.id,
        source_run_id=parent.id,
        source_artifact="latest",
        source_num_timesteps=123,
    )

    with pytest.raises(ValueError, match="only leaf runs can be deleted individually"):
        store.delete_run(parent.id)


def test_manager_store_deletes_full_lineage(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
    )
    child = store.create_run(
        run_id="child-run",
        name="Child Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
        lineage_id=parent.lineage_id,
        parent_run_id=parent.id,
        source_run_id=parent.id,
        source_artifact="latest",
        source_num_timesteps=456,
    )
    parent.run_dir.mkdir(parents=True)
    child.run_dir.mkdir(parents=True)

    assert store.delete_lineage(parent.lineage_id) is True
    assert store.get_run(parent.id) is None
    assert store.get_run(child.id) is None
    assert not parent.run_dir.exists()


def test_manager_store_delete_lineage_defers_failed_filesystem_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
    )
    child = store.create_run(
        run_id="child-run",
        name="Child Run",
        config=config,
        managed_runs_root=tmp_path / "runs",
        lineage_id=parent.lineage_id,
        parent_run_id=parent.id,
        source_run_id=parent.id,
        source_artifact="latest",
        source_num_timesteps=456,
    )
    parent.run_dir.mkdir(parents=True)
    child.run_dir.mkdir(parents=True)
    original_apply = filesystem_ops_module.apply_filesystem_operation

    def fail_delete(operation: FilesystemOperation) -> bool:
        raise OSError("filesystem busy")

    monkeypatch.setattr(filesystem_ops_module, "apply_filesystem_operation", fail_delete)

    assert store.delete_lineage(parent.lineage_id) is True
    assert store.get_run(parent.id) is None
    assert store.get_run(child.id) is None
    assert parent.run_dir.exists()
    with sqlite3.connect(store.db_path) as connection:
        pending = connection.execute(
            "SELECT COUNT(*) FROM filesystem_operations",
        ).fetchone()
    assert pending is not None
    assert pending[0] >= 2

    monkeypatch.setattr(filesystem_ops_module, "apply_filesystem_operation", original_apply)
    recovered = ManagerStore(store.db_path)
    recovered.initialize()

    assert not parent.run_dir.exists()
    with sqlite3.connect(store.db_path) as connection:
        pending = connection.execute(
            "SELECT COUNT(*) FROM filesystem_operations",
        ).fetchone()
    assert pending is not None
    assert pending[0] == 0


def test_manager_store_migration_replays_pending_directory_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed_runs_root = tmp_path / "local" / "runs"
    legacy_lineages_root = tmp_path / "local" / "lineages"
    monkeypatch.setattr(registry_paths, "manager_root", lambda output_root=None: managed_runs_root)
    monkeypatch.setattr(
        registry_paths,
        "manager_run_dir",
        lambda *, run_id, lineage_id, output_root=None: managed_runs_root / lineage_id / run_id,
    )

    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="legacy-run",
        name="Legacy Run",
        config=default_managed_run_config(),
        explicit_run_dir=legacy_lineages_root / "legacy-run" / "legacy-run",
        lineage_id="legacy-run",
    )
    run.run_dir.mkdir(parents=True)
    (run.run_dir / "artifact.bin").write_text("payload", encoding="utf-8")
    target_run_dir = managed_runs_root / "legacy-run" / "legacy-run"
    original_apply = filesystem_ops_module.apply_filesystem_operation
    seen_move = False

    def fail_first_move(operation: FilesystemOperation) -> bool:
        nonlocal seen_move
        if operation.kind == "move_tree" and not seen_move:
            seen_move = True
            raise OSError("filesystem busy")
        return original_apply(operation)

    monkeypatch.setattr(filesystem_ops_module, "apply_filesystem_operation", fail_first_move)

    with pytest.raises(OSError, match="filesystem busy"):
        store.migrate_lineage_layout()

    assert run.run_dir.exists()
    with sqlite3.connect(store.db_path) as connection:
        row = connection.execute(
            "SELECT run_dir FROM runs WHERE id = ?",
            (run.id,),
        ).fetchone()
        pending = connection.execute(
            "SELECT COUNT(*) FROM filesystem_operations WHERE kind = 'move_tree'",
        ).fetchone()
    assert row is not None
    assert Path(str(row[0])).resolve() == target_run_dir.resolve()
    assert pending is not None
    assert pending[0] == 1

    monkeypatch.setattr(filesystem_ops_module, "apply_filesystem_operation", original_apply)
    recovered = ManagerStore(store.db_path)
    recovered.initialize()
    refreshed = recovered.get_run(run.id)

    assert refreshed is not None
    assert refreshed.run_dir == target_run_dir.resolve()
    assert target_run_dir.exists()
    assert not run.run_dir.exists()


def test_manager_store_rejects_deleting_running_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Active Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    launched = store.update_run_status(
        run_id=run.id,
        status="running",
        started_at="2026-05-03T12:00:00+00:00",
        stopped_at=None,
        message="worker launched",
    )

    assert launched is not None

    with pytest.raises(ValueError, match="stop or pause the run before deleting it"):
        store.delete_run(run.id)
