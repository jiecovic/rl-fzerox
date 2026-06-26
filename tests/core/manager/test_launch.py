# tests/core/manager/test_launch.py
from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Literal

import pytest

from rl_fzerox.apps.run_manager.launch import ManagerRunLauncher
from rl_fzerox.apps.run_manager.launching.save_games import active_career_mode_runner_pid
from rl_fzerox.apps.run_manager.launching.watch import (
    _watch_reaper,
    active_watch_pid,
    raise_if_watch_exited_early,
    validate_watch_device,
    watch_failure_detail,
)
from rl_fzerox.apps.run_manager.worker.config import _resolved_train_config
from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
from rl_fzerox.core.engine_tuning import (
    ENGINE_TUNING_STATE_VERSION,
    EngineTuningCandidateState,
    EngineTuningContext,
    EngineTuningRuntimeState,
    load_engine_tuning_runtime_state,
    save_engine_tuning_runtime_state,
)
from rl_fzerox.core.manager import (
    ManagedRun,
    ManagedRunConfig,
    ManagerStore,
    default_managed_run_config,
)
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session.artifacts import (
    engine_tuning_checkpoint_path,
    engine_tuning_model_path,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import TrackSamplingAltBaseline
from tests.core.manager.test_manager_store_checkpoints import _manifest, _payloads, _write_bundle


class _RecordingLauncher(ManagerRunLauncher):
    def __init__(self, store: ManagerStore) -> None:
        super().__init__(store)
        self.spawn_calls: list[tuple[str, bool]] = []

    def _spawn_worker(self, *, run_id: str, resume: bool) -> None:
        self.spawn_calls.append((run_id, resume))


def test_launch_allows_unsaved_fork_source(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    source_run = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "parent-run",
    )

    class _ForkRecordingLauncher(ManagerRunLauncher):
        def __init__(self, store: ManagerStore) -> None:
            super().__init__(store)
            self.fork_calls: list[tuple[str, str, str | None, object | None, bool]] = []

        def fork(
            self,
            *,
            run_id: str,
            artifact: str,
            name: str | None = None,
            config: object | None = None,
            exclude_draft_id: str | None = None,
            source_snapshot_dir: Path | None = None,
            source_num_timesteps: int | None = None,
            copy_alt_baselines: bool = True,
            engine_tuning_source_action: Literal["convert", "discard"] = "convert",
        ) -> ManagedRun:
            del (
                exclude_draft_id,
                source_num_timesteps,
                source_snapshot_dir,
                engine_tuning_source_action,
            )
            self.fork_calls.append((run_id, artifact, name, config, copy_alt_baselines))
            return source_run

    launcher = _ForkRecordingLauncher(store)

    launched = launcher.launch(
        name="Forked Child",
        config=config,
        draft_id=None,
        source_run_id=source_run.id,
        source_artifact="best",
    )

    assert launched == source_run
    assert launcher.fork_calls == [(source_run.id, "best", "Forked Child", config, True)]


def test_resume_relaunches_fork_without_local_checkpoint(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    parent_dir = tmp_path / "runs" / "parent-run"
    source_snapshot_dir = tmp_path / "runs" / "fork-source"
    source_snapshot_dir.mkdir(parents=True)
    latest_model_path = parent_dir / RUN_LAYOUT.model_artifacts.latest
    latest_policy_path = parent_dir / RUN_LAYOUT.policy_artifacts.latest
    train_config_path = parent_dir / RUN_LAYOUT.config_filename
    latest_model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_model_path.write_bytes(b"model")
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)
    latest_policy_path.write_bytes(b"policy")
    train_config_path.write_text(
        "train:\n  algorithm: maskable_hybrid_recurrent_ppo\n",
        encoding="utf-8",
    )
    latest_policy_path.with_name("policy.metadata.json").write_text(
        '{"num_timesteps": 816040}\n',
        encoding="utf-8",
    )
    store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=parent_dir,
    )

    run = store.create_run(
        run_id="fork-run",
        name="Fork Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "fork-run",
        parent_run_id="parent-run",
        source_run_id="parent-run",
        source_artifact="latest",
        source_snapshot_dir=source_snapshot_dir,
        source_num_timesteps=816_040,
    )
    store.update_run_status(
        run_id=run.id,
        status="failed",
        stopped_at="2026-05-04T14:00:00+00:00",
        message="startup failed",
    )

    launcher = _RecordingLauncher(store)

    resumed = launcher.resume(run_id=run.id)
    events = store.list_recent_run_events((run.id,))

    assert launcher.spawn_calls == [(run.id, False)]
    assert resumed.status == "running"
    assert resumed.started_at is not None
    assert "pinned fork source" in events[run.id][0].message


def test_fork_copies_active_alt_baselines(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    parent_dir = tmp_path / "runs" / "parent-run"
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=parent_dir,
    )
    _write_latest_checkpoint(parent_dir)
    source_state_path = parent_dir / "baselines" / "alt" / "alt-a.state"
    source_state_path.parent.mkdir(parents=True)
    source_state_path.write_bytes(b"alt-state")
    store.upsert_run_alt_baseline(
        baseline=TrackSamplingAltBaseline(
            id="alt-a",
            run_id=parent.id,
            course_key="mute_city",
            reset_variant_key="gp_race|novice|blue_falcon",
            source_entry_id="mute_city_gp_race_novice_blue_falcon",
            label="chicane approach",
            state_path=source_state_path,
            weight=1.0,
            enabled=True,
            created_at="2026-06-13T10:00:00+00:00",
            updated_at="2026-06-13T10:00:00+00:00",
        )
    )
    store.upsert_run_alt_baseline(
        baseline=TrackSamplingAltBaseline(
            id="alt-missing",
            run_id=parent.id,
            course_key="mute_city",
            reset_variant_key="gp_race|novice|blue_falcon",
            source_entry_id="mute_city_gp_race_novice_blue_falcon",
            label="deleted local file",
            state_path=parent_dir / "baselines" / "alt" / "missing.state",
            weight=1.0,
            enabled=True,
            created_at="2026-06-13T10:01:00+00:00",
            updated_at="2026-06-13T10:01:00+00:00",
        )
    )
    launcher = _RecordingLauncher(store)

    child = launcher.fork(run_id=parent.id, artifact="latest", name="Child Run")

    child_baselines = store.get_run_alt_baselines(child.id)
    assert launcher.spawn_calls == [(child.id, False)]
    assert len(child_baselines) == 1
    child_baseline = child_baselines[0]
    assert child_baseline.id.startswith("alt-a-fork-")
    assert child_baseline.course_key == "mute_city"
    assert child_baseline.source_entry_id == "mute_city_gp_race_novice_blue_falcon"
    assert child_baseline.label == "chicane approach"
    assert child_baseline.state_path.parent == child.run_dir / RUN_LAYOUT.baselines_dirname / "alt"
    assert child_baseline.state_path.read_bytes() == b"alt-state"


def test_fork_published_checkpoint_run_snapshot_creates_child_run(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    payloads = _payloads()
    bundle_path = tmp_path / "blue-falcon.zip"
    _write_bundle(bundle_path, manifest=_manifest(payloads), payloads=payloads)
    checkpoint = store.import_published_checkpoint_bundle(bundle_path=bundle_path)
    launcher = _RecordingLauncher(store)
    child_config = checkpoint.config.model_copy(deep=True)
    child_config.policy.auxiliary_state_enabled = True

    child = launcher.fork(
        run_id=checkpoint.run_id,
        artifact="best",
        name="Fine Tune",
        config=child_config,
    )

    assert launcher.spawn_calls == [(child.id, False)]
    assert child.parent_run_id == checkpoint.run_id
    assert child.source_run_id == checkpoint.run_id
    assert child.source_artifact == "best"
    assert child.source_snapshot_dir is not None
    assert child.source_num_timesteps == checkpoint.local_num_timesteps
    assert child.lineage_step_offset == checkpoint.lineage_num_timesteps
    assert child.lineage_id == checkpoint.id
    assert (child.source_snapshot_dir / RUN_LAYOUT.policy_artifacts.best).read_bytes() == b"policy"
    assert (child.source_snapshot_dir / RUN_LAYOUT.model_artifacts.best).read_bytes() == b"model"
    assert (child.source_snapshot_dir / RUN_LAYOUT.config_filename).is_file()

    train_config = _resolved_train_config(store=store, run=child, resume=False)

    assert train_config.train.resume_run_dir == child.source_snapshot_dir
    assert train_config.train.resume_artifact == "best"
    assert train_config.policy.auxiliary_state.enabled is True
    assert train_config.train.resume_source_auxiliary_state_enabled is False
    assert train_config.train.tensorboard_step_offset == checkpoint.lineage_num_timesteps


def test_fork_without_explicit_config_resets_action_bias_deltas(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    parent_config = default_managed_run_config().model_copy(
        update={
            "policy": default_managed_run_config().policy.model_copy(
                update={
                    "gas_on_logit": -2.0,
                    "air_brake_on_logit": 16.0,
                    "spin_idle_logit": 1.0,
                }
            )
        }
    )
    parent_dir = tmp_path / "runs" / "parent-run"
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=parent_config,
        explicit_run_dir=parent_dir,
    )
    _write_latest_checkpoint(parent_dir)
    launcher = _RecordingLauncher(store)

    child = launcher.fork(run_id=parent.id, artifact="latest", name="Child Run")

    assert child.config.policy.gas_on_logit == pytest.approx(0.0)
    assert child.config.policy.air_brake_on_logit == pytest.approx(0.0)
    assert child.config.policy.spin_idle_logit == pytest.approx(0.0)


def test_fork_with_explicit_config_preserves_action_bias_delta(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    parent_config = default_managed_run_config().model_copy(
        update={
            "policy": default_managed_run_config().policy.model_copy(
                update={"air_brake_on_logit": 16.0}
            )
        }
    )
    child_config = parent_config.model_copy(
        update={"policy": parent_config.policy.model_copy(update={"air_brake_on_logit": 1.0})}
    )
    parent_dir = tmp_path / "runs" / "parent-run"
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=parent_config,
        explicit_run_dir=parent_dir,
    )
    _write_latest_checkpoint(parent_dir)
    launcher = _RecordingLauncher(store)

    child = launcher.fork(
        run_id=parent.id,
        artifact="latest",
        name="Child Run",
        config=child_config,
    )

    assert child.config.policy.air_brake_on_logit == pytest.approx(1.0)


def test_fork_can_skip_active_alt_baselines(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    parent_dir = tmp_path / "runs" / "parent-run"
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=parent_dir,
    )
    _write_latest_checkpoint(parent_dir)
    source_state_path = parent_dir / "baselines" / "alt" / "alt-a.state"
    source_state_path.parent.mkdir(parents=True)
    source_state_path.write_bytes(b"alt-state")
    store.upsert_run_alt_baseline(
        baseline=TrackSamplingAltBaseline(
            id="alt-a",
            run_id=parent.id,
            course_key="mute_city",
            reset_variant_key="gp_race|novice|blue_falcon",
            source_entry_id="mute_city_gp_race_novice_blue_falcon",
            label="chicane approach",
            state_path=source_state_path,
            weight=1.0,
            enabled=True,
            created_at="2026-06-13T10:00:00+00:00",
            updated_at="2026-06-13T10:00:00+00:00",
        )
    )
    launcher = _RecordingLauncher(store)

    child = launcher.fork(
        run_id=parent.id,
        artifact="latest",
        name="Child Run",
        copy_alt_baselines=False,
    )

    assert launcher.spawn_calls == [(child.id, False)]
    assert store.get_run_alt_baselines(child.id) == ()


def test_fork_converts_engine_tuning_source_to_bandit_buckets(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    parent_dir = tmp_path / "runs" / "parent-run"
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=default_managed_run_config(),
        explicit_run_dir=parent_dir,
    )
    _write_latest_checkpoint(parent_dir)
    parent_policy_path = parent_dir / RUN_LAYOUT.policy_artifacts.latest
    parent_state_path = engine_tuning_checkpoint_path(parent_policy_path)
    parent_model_path = engine_tuning_model_path(parent_policy_path)
    _write_off_grid_engine_tuning_state(parent_state_path)
    parent_model_path.write_bytes(b"stale-model-sidecar")
    launcher = _RecordingLauncher(store)

    child = launcher.fork(
        run_id=parent.id,
        artifact="latest",
        name="Child Run",
        config=_bandit_engine_config(),
        engine_tuning_source_action="convert",
    )

    assert child.source_snapshot_dir is not None
    child_policy_path = child.source_snapshot_dir / RUN_LAYOUT.policy_artifacts.latest
    child_state = load_engine_tuning_runtime_state(engine_tuning_checkpoint_path(child_policy_path))
    parent_state = load_engine_tuning_runtime_state(parent_state_path)
    assert child_state is not None
    assert [candidate.engine_setting_raw_value for candidate in child_state.candidates] == [44, 84]
    assert [candidate.finish_count for candidate in child_state.candidates] == [2, 1]
    assert [candidate.episode_count for candidate in child_state.candidates] == [3, 2]
    assert [candidate.return_count for candidate in child_state.candidates] == [3, 2]
    assert [candidate.mean_completion_score for candidate in child_state.candidates] == [
        2.5 / 3,
        0.625,
    ]
    assert [candidate.mean_return_score for candidate in child_state.candidates] == [10.0, 9.0]
    assert child_state.model_state is None
    assert not engine_tuning_model_path(child_policy_path).exists()
    assert parent_state is not None
    assert [candidate.engine_setting_raw_value for candidate in parent_state.candidates] == [
        44,
        50,
        67,
        84,
    ]
    assert parent_model_path.is_file()


def test_fork_can_discard_engine_tuning_source(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    parent_dir = tmp_path / "runs" / "parent-run"
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=default_managed_run_config(),
        explicit_run_dir=parent_dir,
    )
    _write_latest_checkpoint(parent_dir)
    parent_policy_path = parent_dir / RUN_LAYOUT.policy_artifacts.latest
    parent_state_path = engine_tuning_checkpoint_path(parent_policy_path)
    parent_model_path = engine_tuning_model_path(parent_policy_path)
    _write_off_grid_engine_tuning_state(parent_state_path)
    parent_model_path.write_bytes(b"stale-model-sidecar")
    launcher = _RecordingLauncher(store)

    child = launcher.fork(
        run_id=parent.id,
        artifact="latest",
        name="Child Run",
        config=_bandit_engine_config(),
        engine_tuning_source_action="discard",
    )

    assert child.source_snapshot_dir is not None
    child_policy_path = child.source_snapshot_dir / RUN_LAYOUT.policy_artifacts.latest
    assert not engine_tuning_checkpoint_path(child_policy_path).exists()
    assert not engine_tuning_model_path(child_policy_path).exists()
    assert parent_state_path.is_file()
    assert parent_model_path.is_file()


def test_fork_can_discard_engine_tuning_source_for_non_bandit_child(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    parent_dir = tmp_path / "runs" / "parent-run"
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=default_managed_run_config(),
        explicit_run_dir=parent_dir,
    )
    _write_latest_checkpoint(parent_dir)
    parent_policy_path = parent_dir / RUN_LAYOUT.policy_artifacts.latest
    parent_state_path = engine_tuning_checkpoint_path(parent_policy_path)
    parent_model_path = engine_tuning_model_path(parent_policy_path)
    _write_off_grid_engine_tuning_state(parent_state_path)
    parent_model_path.write_bytes(b"stale-model-sidecar")
    launcher = _RecordingLauncher(store)

    child = launcher.fork(
        run_id=parent.id,
        artifact="latest",
        name="Child Run",
        config=_gaussian_process_engine_config(),
        engine_tuning_source_action="discard",
    )

    assert child.source_snapshot_dir is not None
    child_policy_path = child.source_snapshot_dir / RUN_LAYOUT.policy_artifacts.latest
    assert not engine_tuning_checkpoint_path(child_policy_path).exists()
    assert not engine_tuning_model_path(child_policy_path).exists()
    assert parent_state_path.is_file()
    assert parent_model_path.is_file()


def test_resume_rebuilds_missing_fork_source_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    parent_dir = tmp_path / "runs" / "parent-run"
    parent_dir.mkdir(parents=True)
    (tmp_path / "core.so").touch()
    (tmp_path / "rom.n64").touch()
    train_config_path = parent_dir / RUN_LAYOUT.config_filename
    latest_model_path = parent_dir / RUN_LAYOUT.model_artifacts.latest
    latest_policy_path = parent_dir / RUN_LAYOUT.policy_artifacts.latest
    latest_model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_model_path.write_bytes(b"model")
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)
    latest_policy_path.write_bytes(b"policy")
    train_config_path.write_text(
        "train:\n  algorithm: maskable_hybrid_recurrent_ppo\n",
        encoding="utf-8",
    )
    latest_policy_path.with_name("policy.metadata.json").write_text(
        '{"num_timesteps": 816040}\n',
        encoding="utf-8",
    )
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=parent_dir,
    )

    run_dir = tmp_path / "runs" / "fork-run"
    fork_source_dir = run_dir / "fork_source"
    run = store.create_run(
        run_id="fork-run",
        name="Fork Run",
        config=config,
        explicit_run_dir=run_dir,
        lineage_id=parent.lineage_id,
        parent_run_id="parent-run",
        source_run_id="parent-run",
        source_artifact="latest",
        source_snapshot_dir=fork_source_dir,
        source_num_timesteps=816_040,
        lineage_step_offset=816_040,
    )
    store.update_run_status(
        run_id=run.id,
        status="failed",
        stopped_at="2026-05-04T14:00:00+00:00",
        message="startup failed",
    )

    launcher = _RecordingLauncher(store)

    resumed = launcher.resume(run_id=run.id)
    refreshed = store.get_run(run.id)

    assert launcher.spawn_calls == [(run.id, False)]
    assert resumed.status == "running"
    assert resumed.started_at is not None
    assert refreshed is not None
    assert refreshed.source_snapshot_dir is not None
    assert (refreshed.source_snapshot_dir / RUN_LAYOUT.model_artifacts.latest).is_file()
    assert (refreshed.source_snapshot_dir / RUN_LAYOUT.config_filename).read_text(
        encoding="utf-8"
    ) == train_config_path.read_text(encoding="utf-8")
    assert refreshed.source_num_timesteps == 816_040


def test_resume_rebuilds_incomplete_existing_fork_source_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    parent_dir = tmp_path / "runs" / "parent-run"
    parent_dir.mkdir(parents=True)
    latest_model_path = parent_dir / RUN_LAYOUT.model_artifacts.latest
    latest_policy_path = parent_dir / RUN_LAYOUT.policy_artifacts.latest
    train_config_path = parent_dir / RUN_LAYOUT.config_filename
    latest_model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_model_path.write_bytes(b"model")
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)
    latest_policy_path.write_bytes(b"policy")
    train_config_path.write_text(
        "train:\n  algorithm: maskable_hybrid_recurrent_ppo\n",
        encoding="utf-8",
    )
    latest_policy_path.with_name("policy.metadata.json").write_text(
        '{"num_timesteps": 816040}\n',
        encoding="utf-8",
    )
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=config,
        explicit_run_dir=parent_dir,
    )

    run_dir = tmp_path / "runs" / "fork-run"
    fork_source_dir = run_dir / "fork_source"
    incomplete_model_path = fork_source_dir / RUN_LAYOUT.model_artifacts.latest
    incomplete_policy_path = fork_source_dir / RUN_LAYOUT.policy_artifacts.latest
    incomplete_model_path.parent.mkdir(parents=True, exist_ok=True)
    incomplete_model_path.write_bytes(b"stale-model")
    incomplete_policy_path.write_bytes(b"stale-policy")
    run = store.create_run(
        run_id="fork-run",
        name="Fork Run",
        config=config,
        explicit_run_dir=run_dir,
        lineage_id=parent.lineage_id,
        parent_run_id="parent-run",
        source_run_id="parent-run",
        source_artifact="latest",
        source_snapshot_dir=fork_source_dir,
        source_num_timesteps=816_040,
        lineage_step_offset=816_040,
    )
    store.update_run_status(
        run_id=run.id,
        status="failed",
        stopped_at="2026-05-04T14:00:00+00:00",
        message="startup failed",
    )

    launcher = _RecordingLauncher(store)

    resumed = launcher.resume(run_id=run.id)
    refreshed = store.get_run(run.id)
    events = store.list_recent_run_events((run.id,))

    assert launcher.spawn_calls == [(run.id, False)]
    assert resumed.status == "running"
    assert refreshed is not None
    assert refreshed.source_snapshot_dir is not None
    assert (refreshed.source_snapshot_dir / RUN_LAYOUT.config_filename).read_text(
        encoding="utf-8"
    ) == train_config_path.read_text(encoding="utf-8")
    assert events[run.id][1].kind == "fork_source_rebuilt"


def test_resume_requires_local_checkpoint_for_root_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    run = store.create_run(
        run_id="root-run",
        name="Root Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "root-run",
    )
    store.update_run_status(
        run_id=run.id,
        status="failed",
        stopped_at="2026-05-04T14:00:00+00:00",
        message="startup failed",
    )

    launcher = _RecordingLauncher(store)

    with pytest.raises(ValueError, match="no resumable checkpoint exists"):
        launcher.resume(run_id=run.id)


def test_resume_uses_latest_checkpoint_when_present(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    run_dir = tmp_path / "runs" / "checkpointed-run"
    run = store.create_run(
        run_id="checkpointed-run",
        name="Checkpointed Run",
        config=config,
        explicit_run_dir=run_dir,
    )
    store.update_run_status(
        run_id=run.id,
        status="stopped",
        stopped_at="2026-05-04T14:00:00+00:00",
        message="stopped cleanly",
    )
    latest_model_path = run_dir / RUN_LAYOUT.model_artifacts.latest
    latest_model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_model_path.write_bytes(b"model")

    launcher = _RecordingLauncher(store)

    resumed = launcher.resume(run_id=run.id)
    events = store.list_recent_run_events((run.id,))

    assert launcher.spawn_calls == [(run.id, True)]
    assert resumed.status == "running"
    assert "latest checkpoint" in events[run.id][0].message


def test_watch_artifact_skips_duplicate_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    run = store.create_run(
        run_id="watch-run",
        name="Watch Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "watch-run",
    )
    launcher = ManagerRunLauncher(store)

    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.resolve_model_artifact_path",
        lambda *_args, **_kwargs: run.run_dir / RUN_LAYOUT.model_artifacts.latest,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.active_watch_pid",
        lambda **_kwargs: 4321,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.resolve_watch_app_config",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate watch should not resolve config")
        ),
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate watch should not spawn a process")
        ),
    )

    status = launcher.watch_artifact(
        run_id=run.id,
        artifact="latest",
        device="cuda",
        renderer=None,
        deterministic_policy=True,
    )

    assert status == "already_running"


def test_watch_artifact_skips_duplicate_materialization_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="watch-run",
        name="Watch Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "watch-run",
    )
    launcher = ManagerRunLauncher(store)
    launch_lease_id = store.viewer_lease_id(
        kind="run_watch",
        owner_id=run.id,
        qualifier="latest:launch",
    )
    store.upsert_viewer_lease(
        lease_id=launch_lease_id,
        kind="run_watch",
        owner_id=run.id,
        pid=os.getpid(),
        qualifier="latest:launch",
    )

    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.resolve_model_artifact_path",
        lambda *_args, **_kwargs: run.run_dir / RUN_LAYOUT.model_artifacts.latest,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.resolve_watch_app_config",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate watch should not resolve config")
        ),
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate watch should not spawn a process")
        ),
    )

    status = launcher.watch_artifact(
        run_id=run.id,
        artifact="latest",
        device="cuda",
        renderer=None,
        deterministic_policy=True,
    )

    assert status == "already_running"


def test_active_watch_pid_clears_stale_viewer_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="watch-run",
        name="Watch Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "watch-run",
    )
    lease_id = store.viewer_lease_id(
        kind="run_watch",
        owner_id=run.id,
        qualifier="latest",
    )
    store.upsert_viewer_lease(
        lease_id=lease_id,
        kind="run_watch",
        owner_id=run.id,
        pid=os.getpid(),
        qualifier="latest",
    )
    assert store.heartbeat_viewer_lease(
        lease_id=lease_id,
        pid=os.getpid(),
        heartbeat_at=_stale_heartbeat_at(),
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.watch_process_matches",
        lambda **_kwargs: True,
    )

    assert (
        active_watch_pid(
            store=store,
            lease_id=lease_id,
            run_id=run.id,
            artifact="latest",
        )
        is None
    )
    assert store.get_viewer_lease(lease_id) is None


def test_watch_artifact_passes_viewer_lease_to_watch_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    run = store.create_run(
        run_id="watch-run",
        name="Watch Run",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "watch-run",
    )
    launcher = ManagerRunLauncher(store)
    log_path = tmp_path / "logs" / "watch-latest.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("stale traceback\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeProcess:
        pid = 4321

        def wait(self, timeout: float | None = None) -> int:
            if timeout is None:
                return 0
            raise subprocess.TimeoutExpired(cmd="watch", timeout=timeout or 0.0)

    def _fake_popen(
        command: list[str],
        *,
        cwd: Path,
        stdin: object,
        stdout: BinaryIO,
        stderr: object,
        start_new_session: bool,
    ) -> _FakeProcess:
        del cwd, stdin, stderr, start_new_session
        captured["command"] = command
        stdout.write(b"watch started\n")
        return _FakeProcess()

    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.resolve_model_artifact_path",
        lambda *_args, **_kwargs: run.run_dir / RUN_LAYOUT.model_artifacts.latest,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.resolve_watch_app_config",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.validate_watch_device",
        lambda _device: None,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.manager_watch_log_path",
        lambda run_id, *, artifact: log_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.subprocess.Popen",
        _fake_popen,
    )

    status = launcher.watch_artifact(
        run_id=run.id,
        artifact="latest",
        device="cuda",
        renderer="angrylion",
        deterministic_policy=False,
    )

    assert status == "started"
    lease_id = store.viewer_lease_id(
        kind="run_watch",
        owner_id=run.id,
        qualifier="latest",
    )
    assert captured["command"] == [
        sys.executable,
        "-m",
        "rl_fzerox.apps.watch",
        "--manager-db-path",
        str(store.db_path),
        "--run-id",
        run.id,
        "--artifact",
        "latest",
        "--viewer-lease-id",
        lease_id,
        "--",
        "watch.device=cuda",
        "watch.deterministic_policy=false",
        "emulator.renderer=angrylion",
    ]
    lease = store.get_viewer_lease(lease_id)
    assert lease is not None
    assert lease.kind == "run_watch"
    assert lease.owner_id == run.id
    assert lease.qualifier == "latest"
    assert lease.pid == 4321
    log_text = log_path.read_text(encoding="utf-8")
    assert "stale traceback" not in log_text
    assert "# launched_at=" in log_text
    assert "# command=" in log_text
    assert "watch started" in log_text


def test_watch_device_guard_rejects_cpu_only_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.watch.import_module",
        lambda name: _FakeTorch if name == "torch" else None,
    )

    with pytest.raises(RuntimeError, match="active PyTorch build cannot use CUDA"):
        validate_watch_device("cuda")


def test_watch_reaper_records_abnormal_exit(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="watch-run",
        name="Watch Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "watch-run",
    )
    lease_id = store.viewer_lease_id(
        kind="run_watch",
        owner_id=run.id,
        qualifier="latest",
    )
    store.upsert_viewer_lease(
        lease_id=lease_id,
        kind="run_watch",
        owner_id=run.id,
        pid=4321,
        qualifier="latest",
    )
    log_path = tmp_path / "watch.log"
    log_path.write_text("RuntimeError: CUDA error: out of memory\n", encoding="utf-8")

    class _ExitedProcess:
        pid = 4321

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 1

    _watch_reaper(
        db_path=store.db_path,
        process=_ExitedProcess(),
        lease_id=lease_id,
        run_id=run.id,
        artifact="latest",
        log_path=log_path,
    )

    assert store.get_viewer_lease(lease_id) is None
    events = store.list_recent_run_events((run.id,))[run.id]
    assert events[0].kind == "watch_failed"
    assert events[0].message == "latest watch failed: RuntimeError: CUDA error: out of memory"


def test_watch_reaper_clears_lease_without_event_on_normal_exit(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="watch-run",
        name="Watch Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "watch-run",
    )
    lease_id = store.viewer_lease_id(
        kind="run_watch",
        owner_id=run.id,
        qualifier="latest",
    )
    store.upsert_viewer_lease(
        lease_id=lease_id,
        kind="run_watch",
        owner_id=run.id,
        pid=4321,
        qualifier="latest",
    )
    log_path = tmp_path / "watch.log"
    log_path.write_text("", encoding="utf-8")

    class _ExitedProcess:
        pid = 4321

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    _watch_reaper(
        db_path=store.db_path,
        process=_ExitedProcess(),
        lease_id=lease_id,
        run_id=run.id,
        artifact="latest",
        log_path=log_path,
    )

    assert store.get_viewer_lease(lease_id) is None
    events = store.list_recent_run_events((run.id,))[run.id]
    assert [event.kind for event in events] == ["created"]


def test_watch_startup_allows_immediate_normal_close(tmp_path: Path) -> None:
    log_path = tmp_path / "watch.log"
    log_path.write_text("", encoding="utf-8")

    class _ExitedProcess:
        pid = 4321

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None
            return 0

    raise_if_watch_exited_early(process=_ExitedProcess(), log_path=log_path)


def test_watch_failure_detail_ignores_launch_log_headers(tmp_path: Path) -> None:
    log_path = tmp_path / "watch.log"
    log_path.write_text(
        "# launched_at=2026-06-08T12:00:00+00:00\n# command=python -m rl_fzerox.apps.watch\n\n",
        encoding="utf-8",
    )

    assert watch_failure_detail(log_path) is None

    log_path.write_text(
        "# launched_at=2026-06-08T12:00:00+00:00\nRuntimeError: failed for real\n",
        encoding="utf-8",
    )

    assert watch_failure_detail(log_path) == "RuntimeError: failed for real"

    log_path.write_text(
        "\n".join(
            (
                "# launched_at=2026-06-13T18:00:00+00:00",
                "RuntimeError: CUDA error: out of memory",
                "Search for `cudaErrorMemoryAllocation' in CUDA docs.",
                "Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.",
            )
        ),
        encoding="utf-8",
    )

    assert watch_failure_detail(log_path) == "RuntimeError: CUDA error: out of memory"


def test_start_career_mode_passes_viewer_lease_and_runtime_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(
        name="Unlock Save",
        save_games_root=tmp_path / "career-saves",
    )
    run = store.create_run(
        name="Unlock Policy",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    _configure_gp_cup(store, save_game_id=save_game.id, run_id=run.id, cup_id="jack")
    launcher = ManagerRunLauncher(store)
    log_path = tmp_path / "logs" / f"{save_game.id}.log"
    recording_path = Path("local/recordings/career/save-001/session/career.mkv")
    log_path.parent.mkdir(parents=True)
    log_path.write_text("stale career failure\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeProcess:
        pid = 8765

        def wait(self, timeout: float | None = None) -> int:
            if timeout is None:
                return 0
            raise subprocess.TimeoutExpired(cmd="career-mode", timeout=timeout or 0.0)

    def _fake_popen(
        command: list[str],
        *,
        cwd: Path,
        stdin: object,
        stdout: BinaryIO,
        stderr: object,
        start_new_session: bool,
    ) -> _FakeProcess:
        del cwd, stdin, stderr, start_new_session
        captured["command"] = command
        stdout.write(b"career mode started\n")
        return _FakeProcess()

    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.save_games.manager_career_mode_log_path",
        lambda save_game_id: log_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.save_games.subprocess.Popen",
        _fake_popen,
    )

    def _fake_default_recording_path(*, save_game_id: str, save_game_name: str) -> Path:
        del save_game_id, save_game_name
        return recording_path

    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.save_games.default_career_recording_path",
        _fake_default_recording_path,
    )

    status = launcher.start_career_mode(
        save_game_id=save_game.id,
        device="cpu",
        renderer="angrylion",
        attempt_seed=1234,
        deterministic_policy=False,
        recording_enabled=True,
        recording_input_hud_enabled=True,
        recording_upscale_factor=3,
        recording_path=None,
        reload_policy_between_attempts=False,
    )

    assert status == "started"
    lease_id = store.viewer_lease_id(kind="career_mode", owner_id=save_game.id)
    attempts = store.list_save_attempts(save_game.id)
    assert len(attempts) == 1
    assert attempts[0].cup_id == "jack"
    assert captured["command"] == [
        sys.executable,
        "-m",
        "rl_fzerox.apps.career_mode",
        "--manager-db-path",
        str(store.db_path),
        "--save-game-id",
        save_game.id,
        "--save-attempt-id",
        attempts[0].id,
        "--viewer-lease-id",
        lease_id,
        "--attempt-seed",
        "1234",
        "--policy-mode",
        "stochastic",
        "--",
        "watch.device=cpu",
        "watch.deterministic_policy=false",
        "emulator.renderer=angrylion",
        "watch.reload_policy_between_attempts=false",
        "watch.recording.enabled=true",
        f"watch.recording.path={recording_path}",
        "watch.recording.session_mp4_enabled=true",
        "watch.recording.keep_failed_segments=true",
        "watch.recording.upscale_factor=3",
        "watch.recording.render_input_hud=true",
    ]
    lease = store.get_viewer_lease(lease_id)
    assert lease is not None
    assert lease.kind == "career_mode"
    assert lease.owner_id == save_game.id
    assert lease.qualifier is None
    assert lease.pid == 8765
    log_text = log_path.read_text(encoding="utf-8")
    assert "stale career failure" not in log_text
    assert "# launched_at=" in log_text
    assert "# command=" in log_text
    assert "career mode started" in log_text


def test_active_career_mode_runner_pid_clears_stale_viewer_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(
        name="Unlock Save",
        save_games_root=tmp_path / "career-saves",
    )
    lease_id = store.viewer_lease_id(kind="career_mode", owner_id=save_game.id)
    store.upsert_viewer_lease(
        lease_id=lease_id,
        kind="career_mode",
        owner_id=save_game.id,
        pid=os.getpid(),
    )
    assert store.heartbeat_viewer_lease(
        lease_id=lease_id,
        pid=os.getpid(),
        heartbeat_at=_stale_heartbeat_at(),
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.save_games.career_mode_process_matches",
        lambda **_kwargs: True,
    )

    assert (
        active_career_mode_runner_pid(
            store=store,
            lease_id=lease_id,
            save_game_id=save_game.id,
        )
        is None
    )
    assert store.get_viewer_lease(lease_id) is None


def _stale_heartbeat_at() -> str:
    return (datetime.now(UTC) - timedelta(minutes=1)).isoformat(timespec="seconds")


def _write_latest_checkpoint(run_dir: Path) -> None:
    latest_model_path = run_dir / RUN_LAYOUT.model_artifacts.latest
    latest_policy_path = run_dir / RUN_LAYOUT.policy_artifacts.latest
    latest_model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_model_path.write_bytes(b"model")
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)
    latest_policy_path.write_bytes(b"policy")
    (run_dir / RUN_LAYOUT.config_filename).write_text(
        "train:\n  algorithm: maskable_hybrid_recurrent_ppo\n",
        encoding="utf-8",
    )
    latest_policy_path.with_name("policy.metadata.json").write_text(
        '{"num_timesteps": 816040}\n',
        encoding="utf-8",
    )


def _bandit_engine_config() -> ManagedRunConfig:
    config = default_managed_run_config()
    return config.model_copy(
        update={
            "vehicle": config.vehicle.model_copy(
                update={
                    "engine_mode": "adaptive_tuner",
                    "engine_setting_min_raw_value": 44,
                    "engine_setting_max_raw_value": 84,
                    "adaptive_engine_tuner_backend": "bandit",
                    "adaptive_engine_bandit_bucket_raw_values": (44, 54, 64, 74, 84),
                }
            )
        }
    )


def _gaussian_process_engine_config() -> ManagedRunConfig:
    config = default_managed_run_config()
    return config.model_copy(
        update={
            "vehicle": config.vehicle.model_copy(
                update={
                    "engine_mode": "adaptive_tuner",
                    "engine_setting_min_raw_value": 44,
                    "engine_setting_max_raw_value": 84,
                    "adaptive_engine_tuner_backend": "gaussian_process",
                }
            )
        }
    )


def _write_off_grid_engine_tuning_state(state_path: Path) -> None:
    context = EngineTuningContext(course_key="mute_city", vehicle_id="blue_falcon")
    save_engine_tuning_runtime_state(
        state_path,
        EngineTuningRuntimeState(
            version=ENGINE_TUNING_STATE_VERSION,
            update_count=2,
            candidates=(
                EngineTuningCandidateState(
                    context_key=context.key,
                    course_key=context.course_key,
                    vehicle_id=context.vehicle_id,
                    engine_setting_raw_value=44,
                    episode_count=3,
                    finish_count=2,
                    return_count=3,
                    decayed_count=2.0,
                    decayed_score_total=-200.0,
                    score_total=-200.0,
                    best_score=-90.0,
                    completion_score_total=2.5,
                    best_completion_score=1.0,
                    return_score_total=30.0,
                    best_return_score=15.0,
                    best_time_ms=90_000,
                ),
                EngineTuningCandidateState(
                    context_key=context.key,
                    course_key=context.course_key,
                    vehicle_id=context.vehicle_id,
                    engine_setting_raw_value=50,
                    finish_count=99,
                    decayed_count=99.0,
                    decayed_score_total=-7_920.0,
                    score_total=-7_920.0,
                    best_score=-70.0,
                    best_time_ms=70_000,
                ),
                EngineTuningCandidateState(
                    context_key=context.key,
                    course_key=context.course_key,
                    vehicle_id=context.vehicle_id,
                    engine_setting_raw_value=67,
                    finish_count=99,
                    decayed_count=99.0,
                    decayed_score_total=-7_920.0,
                    score_total=-7_920.0,
                    best_score=-70.0,
                    best_time_ms=70_000,
                ),
                EngineTuningCandidateState(
                    context_key=context.key,
                    course_key=context.course_key,
                    vehicle_id=context.vehicle_id,
                    engine_setting_raw_value=84,
                    episode_count=2,
                    finish_count=1,
                    return_count=2,
                    decayed_count=1.0,
                    decayed_score_total=-80.0,
                    score_total=-80.0,
                    best_score=-80.0,
                    completion_score_total=1.25,
                    best_completion_score=1.0,
                    return_score_total=18.0,
                    best_return_score=12.0,
                    best_time_ms=80_000,
                ),
            ),
        ),
    )


def _configure_gp_cup(
    store: ManagerStore,
    *,
    save_game_id: str,
    run_id: str,
    cup_id: str,
) -> None:
    store.upsert_save_cup_setup(
        save_game_id=save_game_id,
        cup_id=cup_id,
        vehicle_id="blue_falcon",
    )
    for course in sorted(BUILT_IN_COURSES, key=lambda item: item.course_index):
        if course.cup != cup_id:
            continue
        store.upsert_save_course_setup(
            save_game_id=save_game_id,
            cup_id=cup_id,
            course_id=course.id,
            policy_source_kind="run",
            policy_source_id=run_id,
            policy_artifact="best",
        )
