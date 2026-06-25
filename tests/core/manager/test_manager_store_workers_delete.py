# tests/core/manager/test_manager_store_workers_delete.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

import rl_fzerox.core.manager.artifacts.filesystem as filesystem_ops_module
import rl_fzerox.core.manager.registry.runs.maintenance as run_maintenance
from rl_fzerox.core.manager import (
    ManagerStore,
    default_managed_run_config,
)
from rl_fzerox.core.manager.artifacts.filesystem import FilesystemOperation
from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingAltBaseline,
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)
from tests.core.manager.manager_store_support import (
    _filesystem_operation_count,
    _set_worker_heartbeat,
    _worker_exists,
)

SnapshotKind = Literal["run", "draft", "template", "import"]


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
    store.upsert_run_track_sampling_state(
        run_id=run.id,
        state=TrackSamplingRuntimeState(
            sampling_mode="step_balanced",
            action_repeat=2,
            update_episodes=2,
            ema_alpha=0.5,
            max_weight_scale=5.0,
            adaptive_completion_weight=0.35,
            adaptive_target_completion=0.9,
            adaptive_min_confidence_episodes=24,
            adaptive_confidence_scale=4.0,
            update_count=1,
            episodes_since_update=0,
            entries=(
                TrackSamplingRuntimeEntry(
                    track_id="mute",
                    course_key="mute",
                    label="Mute City",
                    base_weight=1.0,
                    current_weight=1.0,
                    completed_frames=100,
                    episode_count=1,
                    finished_episode_count=1,
                    success_sample_count=1,
                    ema_episode_frames=100.0,
                    ema_completion_fraction=1.0,
                ),
            ),
        ),
    )

    assert store.delete_run(run.id) is True
    assert store.get_run(run.id) is None
    assert store.get_run_track_sampling_state(run.id) is None


def test_manager_store_deletes_runs_with_generated_x_cup_slots(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Delete Generated Slot Run",
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

    store.replace_run_generated_x_cup_slots(
        run_id=run.id,
        slots=(
            GeneratedXCupSlot(
                course_key="x_cup_slot_1",
                slot=1,
                generation=7,
                course_id="x_cup_abcdef12",
                course_name="X Cup abcdef12",
                course_hash="abcdef12",
                course_seed=123456,
                segment_count=30,
                course_length=1234.5,
            ),
        ),
    )

    assert store.delete_run(run.id) is True
    assert store.get_run(run.id) is None
    assert store.get_run_generated_x_cup_slots(run.id) == ()


def test_manager_store_deletes_runs_with_alt_baselines(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Delete Alt Baseline Run",
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
    state_path = run.run_dir / "baselines" / "alt" / "alt-a.state"
    state_path.parent.mkdir(parents=True)
    state_path.write_bytes(b"state")

    assert stopped is not None

    store.upsert_run_alt_baseline(
        baseline=TrackSamplingAltBaseline(
            id="alt-a",
            run_id=run.id,
            course_key="mute_city",
            reset_variant_key="gp_race|novice|blue_falcon",
            source_entry_id="mute_city_gp_race_novice_blue_falcon",
            label="chicane approach",
            state_path=state_path,
            weight=1.0,
            enabled=True,
            created_at="2026-06-13T10:00:00+00:00",
            updated_at="2026-06-13T10:00:00+00:00",
        )
    )

    assert store.delete_run(run.id) is True
    assert store.get_run(run.id) is None
    assert store.get_run_alt_baselines(run.id) == ()
    assert not state_path.exists()


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
    assert _filesystem_operation_count(store) == 1

    monkeypatch.setattr(filesystem_ops_module, "apply_filesystem_operation", original_apply)
    recovered = ManagerStore(store.db_path)
    recovered.initialize()

    assert not run.run_dir.exists()
    assert _filesystem_operation_count(store) == 0


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
    _set_worker_heartbeat(store, run_id=run.id, heartbeat_at=stale_heartbeat)
    monkeypatch.setattr(run_maintenance, "pid_exists", lambda pid: False)

    store.reconcile_orphaned_runs()
    failed = store.get_run(run.id)

    assert failed is not None
    assert failed.status == "failed"
    assert not _worker_exists(store, run.id)


def test_manager_store_run_reads_do_not_reconcile_stale_worker_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Read Only Lease Run",
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
    _set_worker_heartbeat(store, run_id=run.id, heartbeat_at=stale_heartbeat)
    monkeypatch.setattr(run_maintenance, "pid_exists", lambda pid: False)

    fetched = store.get_run(run.id)
    summaries = store.list_visible_run_summaries()

    assert fetched is not None
    assert fetched.status == "running"
    assert tuple(summary.id for summary in summaries) == (run.id,)
    assert summaries[0].status == "running"
    assert _worker_exists(store, run.id)


def test_manager_store_keeps_fresh_worker_lease_when_pid_is_not_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Fresh Dead Lease Run",
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
    requested = store.request_run_command(run_id=run.id, command="stop")
    assert requested is not None
    assert requested.pending_command == "stop"
    fresh_heartbeat = (datetime.now(UTC) + timedelta(minutes=5)).isoformat(timespec="seconds")
    _set_worker_heartbeat(store, run_id=run.id, heartbeat_at=fresh_heartbeat)
    monkeypatch.setattr(run_maintenance, "pid_exists", lambda pid: False)

    store.reconcile_orphaned_runs()
    refreshed = store.get_run(run.id)

    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.pending_command == "stop"
    assert _worker_exists(store, run.id)


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
    _set_worker_heartbeat(store, run_id=run.id, heartbeat_at=stale_heartbeat)
    monkeypatch.setattr(run_maintenance, "pid_exists", lambda pid: True)

    store.reconcile_orphaned_runs()
    refreshed = store.get_run(run.id)

    assert refreshed is not None
    assert refreshed.status == "running"


def test_manager_store_exposes_worker_heartbeat_on_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(run_maintenance, "pid_exists", lambda pid: True)

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
