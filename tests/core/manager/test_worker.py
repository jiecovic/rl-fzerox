# tests/core/manager/test_worker.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.apps.run_manager.worker import _mark_worker_boot_failure, _resolved_train_config
from rl_fzerox.core.domain.x_cup import X_CUP_COURSE, generated_x_cup_slot_key
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)


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
    store.upsert_run_track_sampling_state(
        run_id=run.id,
        state=TrackSamplingRuntimeState(
            sampling_mode="adaptive_step_balanced",
            action_repeat=config.action.action_repeat,
            update_episodes=5,
            ema_alpha=0.1,
            max_weight_scale=5.0,
            adaptive_completion_weight=0.35,
            adaptive_target_completion=0.9,
            adaptive_min_confidence_episodes=24,
            adaptive_confidence_scale=4.0,
            update_count=1,
            episodes_since_update=0,
            entries=(
                TrackSamplingRuntimeEntry(
                    track_id=slot_key,
                    course_key=slot_key,
                    label="X Cup rotated",
                    base_weight=1.0,
                    current_weight=1.0,
                    completed_frames=100,
                    episode_count=1,
                    finished_episode_count=1,
                    success_sample_count=1,
                    ema_episode_frames=100.0,
                    ema_completion_fraction=1.0,
                    generated_course_slot=0,
                    generated_course_generation=3,
                    generated_course_id="x_cup_rotated",
                    generated_course_name="X Cup rotated",
                    generated_course_hash="rotated",
                    generated_course_seed=99,
                ),
            ),
        ),
    )

    train_config = _resolved_train_config(store=store, run=run, resume=True)

    entry = train_config.env.track_sampling.entries[0]
    assert entry.generated_course_kind == X_CUP_COURSE.generated_kind
    assert entry.runtime_course_key == slot_key
    assert entry.course_id == "x_cup_rotated"
    assert entry.course_name == "X Cup rotated"
    assert entry.baseline_state_path is None
