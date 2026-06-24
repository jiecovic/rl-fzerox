# tests/core/manager/test_worker.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.apps.run_manager.worker.cli import _mark_worker_boot_failure
from rl_fzerox.apps.run_manager.worker.config import _resolved_train_config
from rl_fzerox.core.domain.courses import X_CUP_COURSE, generated_x_cup_slot_key
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.runtime_spec.schema import (
    EmulatorConfig,
    EnvConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
from rl_fzerox.core.training.runs import save_train_run_config
from tests.core.manager.manager_store_support import _track_sampling_artifact


def test_worker_boot_failure_marks_run_failed_without_loading_config(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")
    run = store.create_run(
        run_id="boot-fail",
        name="Boot fail",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    launch_token = "worker-token"
    store.register_run_worker(
        run_id=run.id,
        launch_token=launch_token,
        pid=123,
        launched_at="2026-05-09T00:00:00+00:00",
    )

    _mark_worker_boot_failure(
        store=store,
        run_id=run.id,
        launch_token=launch_token,
        message="boot exploded",
    )

    failed = store.get_run(run.id)
    assert failed is not None
    assert failed.status == "failed"
    assert store.pending_run_command(run.id) is None
    assert (
        store.heartbeat_run_worker(
            run_id=run.id,
            launch_token=launch_token,
            heartbeat_at="2026-05-09T00:00:01+00:00",
        )
        is False
    )
    events = store.list_recent_run_events((run.id,))[run.id]
    assert events[0].kind == "failed"
    assert events[0].message == "boot exploded"


def test_worker_resume_restores_x_cup_runtime_slots_from_db(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.race_mode = "gp_race"
    config.tracks.selected_course_ids = ()
    config.tracks.include_x_cup = True
    config.tracks.x_cup_course_count = 1
    run = store.create_run(
        run_id="x-cup-runtime-resume",
        name="X Cup runtime resume",
        config=config,
        managed_runs_root=tmp_path / "runs",
    )
    slot_key = generated_x_cup_slot_key(0)
    store.replace_run_generated_x_cup_slots(
        run_id=run.id,
        slots=(
            GeneratedXCupSlot(
                course_key=slot_key,
                slot=0,
                generation=3,
                course_id="x_cup_rotated",
                course_name="X Cup rotated",
                course_hash="rotated",
                course_seed=99,
                segment_count=None,
                course_length=None,
            ),
        ),
    )

    train_config = _resolved_train_config(store=store, run=run, resume=True)

    entry = train_config.env.track_sampling.entries[0]
    assert entry.generated_course_kind == X_CUP_COURSE.generated_kind
    assert entry.id == "x_cup_rotated_gp_race_novice_blue_falcon"
    assert entry.runtime_course_key == slot_key
    assert entry.course_id == "x_cup_rotated"
    assert entry.course_name == "X Cup rotated"
    assert entry.baseline_state_path is None


def test_worker_resume_uses_saved_manifest_runtime_baselines(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.race_mode = "gp_race"
    config.tracks.selected_course_ids = ("mute_city",)
    config.tracks.gp_difficulties = ("master",)
    config.tracks.baseline_variant_count = 8
    run = store.create_run(
        run_id="runtime-manifest-resume",
        name="Runtime manifest resume",
        config=config,
        explicit_run_dir=tmp_path / "runs" / "runtime-manifest-resume",
        lineage_step_offset=123_456,
    )
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "fzerox.n64"
    baseline_path = run.run_dir / "baselines" / "old-run-seeded-baseline.state"
    core_path.touch()
    rom_path.touch()
    baseline_path.parent.mkdir(parents=True)
    manifest_entry = TrackSamplingEntryConfig(
        id="mute_city_gp_race_master_blue_falcon__variant_2",
        course_id="mute_city",
        runtime_course_key="mute_city",
        course_name="Mute City",
        course_index=0,
        mode="gp_race",
        gp_difficulty="master",
        vehicle="blue_falcon",
        baseline_state_path=baseline_path,
        baseline_variant_index=1,
        baseline_variant_count=8,
        baseline_variant_seed=14_629_459_847_334_955_741,
    )
    save_train_run_config(
        config=TrainAppConfig(
            seed=7,
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(
                track_sampling=TrackSamplingConfig(
                    enabled=True,
                    entries=(manifest_entry,),
                    baseline_variant_count=8,
                )
            ),
            train=TrainConfig(output_root=tmp_path / "runs", run_name="runtime-manifest-resume"),
        ),
        run_dir=run.run_dir,
    )

    train_config = _resolved_train_config(store=store, run=run, resume=True)

    assert len(train_config.env.track_sampling.entries) == 1
    entry = train_config.env.track_sampling.entries[0]
    assert entry.id == manifest_entry.id
    assert entry.baseline_state_path == baseline_path.resolve()
    assert entry.baseline_variant_seed == manifest_entry.baseline_variant_seed
    assert train_config.train.continue_run_dir == run.run_dir.resolve()
    assert train_config.train.resume_run_dir == run.run_dir.resolve()
    assert train_config.train.resume_mode == "full_model"
    assert train_config.train.tensorboard_step_offset == 123_456


def test_worker_resume_restores_x_cup_artifacts_from_db(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.race_mode = "gp_race"
    config.tracks.selected_course_ids = ()
    config.tracks.include_x_cup = True
    config.tracks.x_cup_course_count = 1
    run = store.create_run(
        run_id="x-cup-runtime-resume",
        name="X Cup runtime resume",
        config=config,
        managed_runs_root=tmp_path / "runs",
    )
    baseline_path = run.run_dir / "baselines" / "x_cup_1234abcd.state"
    artifact = _track_sampling_artifact(baseline_path, difficulty="novice", course_slot=0)
    store.replace_run_generated_x_cup_slots(
        run_id=run.id,
        slots=(
            GeneratedXCupSlot(
                course_key=generated_x_cup_slot_key(0),
                slot=0,
                generation=3,
                course_id="x_cup_1234abcd",
                course_name="X Cup 1234abcd",
                course_hash="1234abcd",
                course_seed=1234,
                segment_count=38,
                course_length=61_743.98046875,
            ),
        ),
    )
    store.replace_run_track_sampling_artifacts(run_id=run.id, artifacts=(artifact,))

    train_config = _resolved_train_config(store=store, run=run, resume=True)

    entry = train_config.env.track_sampling.entries[0]
    assert entry.generated_course_kind == X_CUP_COURSE.generated_kind
    assert entry.id == artifact.entry_id
    assert entry.runtime_course_key == artifact.course_key
    assert entry.baseline_state_path == artifact.baseline_state_path
    assert entry.generated_course_hash == artifact.generated_course_hash
    assert entry.generated_course_seed == artifact.generated_course_seed
