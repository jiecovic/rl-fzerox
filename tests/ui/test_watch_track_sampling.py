# tests/ui/test_watch_track_sampling.py
from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.runtime_spec.schema import (
    EmulatorConfig,
    EnvConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.core.training.runs import RunPaths
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)
from rl_fzerox.ui.watch.runtime import track_sampling as watch_track_sampling
from rl_fzerox.ui.watch.runtime.track_sampling import ManagedTrackSamplingRefresh


def test_managed_watch_track_sampling_refresh_restores_generated_x_cup_slot(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    db_path = tmp_path / "manager" / "runs.db"
    store = ManagerStore(db_path)
    run = store.create_run(
        run_id="run",
        name="Run",
        config=_managed_x_cup_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    run.run_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = tmp_path / "baselines" / "x_cup_new.state"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.touch()
    store.upsert_run_track_sampling_state(run_id=run.id, state=_runtime_state())

    def fake_materialize_train_run_config(
        config: TrainAppConfig,
        *,
        run_paths: RunPaths,
    ) -> TrainAppConfig:
        del run_paths
        entry = config.env.track_sampling.entries[0]
        track_sampling = config.env.track_sampling.model_copy(
            update={
                "entries": (
                    entry.model_copy(update={"baseline_state_path": baseline_path.resolve()}),
                )
            }
        )
        return config.model_copy(
            update={"env": config.env.model_copy(update={"track_sampling": track_sampling})}
        )

    monkeypatch.setattr(
        watch_track_sampling,
        "materialize_train_run_config",
        fake_materialize_train_run_config,
    )
    refresh = ManagedTrackSamplingRefresh.from_config(
        WatchAppConfig(
            emulator=EmulatorConfig(
                core_path=_touched(tmp_path / "core.so"),
                rom_path=_touched(tmp_path / "rom.n64"),
            ),
            env=EnvConfig(track_sampling=_track_sampling_config()),
            watch=WatchConfig(manager_db_path=db_path, managed_run_id=run.id),
        )
    )

    assert refresh is not None
    refreshed = refresh.refreshed_config(_track_sampling_config(), force=True)

    assert refreshed is not None
    entry = refreshed.entries[0]
    assert entry.id == "x_cup_new"
    assert entry.runtime_course_key == "x_cup_slot_1"
    assert entry.course_id == "x_cup_new"
    assert entry.course_name == "X Cup new"
    assert entry.baseline_state_path == baseline_path.resolve()
    assert entry.generated_course_hash == "newhash"
    assert entry.generated_course_seed == 1234
    assert entry.generated_course_generation == 3


def test_managed_watch_track_sampling_refresh_ignores_unchanged_slots(tmp_path: Path) -> None:
    db_path = tmp_path / "manager" / "runs.db"
    store = ManagerStore(db_path)
    run = store.create_run(
        run_id="run",
        name="Run",
        config=_managed_x_cup_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    run.run_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = tmp_path / "baselines" / "x_cup_old.state"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.touch()
    store.upsert_run_track_sampling_state(run_id=run.id, state=_runtime_state())
    config = _track_sampling_config(
        entry_id="x_cup_new",
        course_id="x_cup_new",
        course_name="X Cup new",
        course_hash="newhash",
        course_seed=1234,
        generation=3,
        baseline_path=baseline_path,
    )
    refresh = ManagedTrackSamplingRefresh.from_config(
        WatchAppConfig(
            emulator=EmulatorConfig(
                core_path=_touched(tmp_path / "core.so"),
                rom_path=_touched(tmp_path / "rom.n64"),
            ),
            env=EnvConfig(track_sampling=config),
            watch=WatchConfig(manager_db_path=db_path, managed_run_id=run.id),
        )
    )

    assert refresh is not None
    assert refresh.refreshed_config(config, force=True) is None


def _track_sampling_config(
    *,
    entry_id: str = "x_cup_old",
    course_id: str = "x_cup_old",
    course_name: str = "X Cup old",
    course_hash: str = "oldhash",
    course_seed: int = 7,
    generation: int = 0,
    baseline_path: Path | None = None,
) -> TrackSamplingConfig:
    return TrackSamplingConfig(
        enabled=True,
        entries=(
            TrackSamplingEntryConfig(
                id=entry_id,
                runtime_course_key="x_cup_slot_1",
                course_id=course_id,
                course_name=course_name,
                display_name=course_name,
                baseline_state_path=baseline_path,
                mode="gp_race",
                course_index=X_CUP_COURSE.course_index,
                generated_course_kind=X_CUP_COURSE.generated_kind,
                generated_course_seed=course_seed,
                generated_course_hash=course_hash,
                generated_course_slot=0,
                generated_course_generation=generation,
            ),
        ),
    )


def _managed_x_cup_run_config():
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.race_mode = "gp_race"
    config.tracks.selected_course_ids = ()
    config.tracks.include_x_cup = True
    config.tracks.x_cup_course_count = 1
    return config


def _runtime_state() -> TrackSamplingRuntimeState:
    return TrackSamplingRuntimeState(
        sampling_mode="adaptive_step_balanced",
        action_repeat=1,
        update_episodes=5,
        ema_alpha=0.3,
        max_weight_scale=10.0,
        adaptive_completion_weight=0.9,
        adaptive_target_completion=0.5,
        adaptive_min_confidence_episodes=24,
        adaptive_confidence_scale=4.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id="x_cup_slot_1",
                course_key="x_cup_slot_1",
                label="X Cup new",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=0,
                episode_count=0,
                finished_episode_count=0,
                success_sample_count=0,
                ema_episode_frames=None,
                ema_completion_fraction=None,
                generated_course_slot=0,
                generated_course_generation=3,
                generated_course_id="x_cup_new",
                generated_course_name="X Cup new",
                generated_course_hash="newhash",
                generated_course_seed=1234,
            ),
        ),
    )


def _touched(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path
