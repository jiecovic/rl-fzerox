# tests/ui/test_watch_track_sampling.py
from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from rl_fzerox.core.domain.courses import X_CUP_COURSE
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.runtime_spec.schema import (
    EmulatorConfig,
    EnvConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingMaterializedArtifact,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.artifacts import reset_variant_key
from rl_fzerox.ui.watch.runtime.courses.baseline import _save_managed_alt_baseline
from rl_fzerox.ui.watch.runtime.courses.navigation import WatchCourseRotation
from rl_fzerox.ui.watch.runtime.courses.sampling import (
    ManagedTrackSamplingRefresh,
    TrackSamplingRefreshStatus,
)
from rl_fzerox.ui.watch.runtime.live.track_sampling import LiveTrackSamplingState


def test_managed_watch_track_sampling_refresh_restores_generated_x_cup_slot(
    tmp_path: Path,
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
    baseline_path = run.run_dir / "baselines" / "x_cup_new.state"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_bytes(b"state")
    store.replace_run_track_sampling_artifacts(
        run_id=run.id,
        artifacts=(_x_cup_artifact(baseline_path),),
    )
    _write_x_cup_slot_state(store, run.id)

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
    assert entry.id == "x_cup_new_gp_race_novice_blue_falcon"
    assert entry.runtime_course_key == "x_cup_slot_1"
    assert entry.course_id == "x_cup_new"
    assert entry.course_name == "X Cup new"
    assert entry.baseline_state_path == baseline_path.resolve()
    assert entry.generated_course_hash == "newhash"
    assert entry.generated_course_seed == 1234
    assert entry.generated_course_generation == 3
    assert entry.generated_course_segment_count == 38
    assert entry.generated_course_length == 61_743.98046875


def test_managed_watch_track_sampling_refresh_keeps_active_x_cup_until_artifact_is_ready(
    tmp_path: Path,
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
    stale_baseline_path = tmp_path / "baselines" / "x_cup_old.state"
    stale_baseline_path.parent.mkdir(parents=True)
    stale_baseline_path.write_bytes(b"state")
    store.replace_run_track_sampling_artifacts(
        run_id=run.id,
        artifacts=(_stale_x_cup_artifact(stale_baseline_path),),
    )
    _write_x_cup_slot_state(store, run.id)
    config = _track_sampling_config()
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
    status = refresh.refresh_status(config, force=True)

    assert status.refreshed_config is None
    assert status.ready_for_reset


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
    _write_x_cup_slot_state(store, run.id)
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


def test_managed_watch_track_sampling_refresh_does_not_block_pending_x_cup_artifact(
    tmp_path: Path,
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
    old_baseline_path = tmp_path / "old.state"
    old_baseline_path.write_bytes(b"old")
    _write_x_cup_slot_state(store, run.id)
    config = _track_sampling_config(baseline_path=old_baseline_path)
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
    status = refresh.refresh_status(config, force=True)
    assert status.refreshed_config is None
    assert status.ready_for_reset


def test_managed_watch_track_sampling_refresh_remembers_pending_state_between_checks(
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
    old_baseline_path = tmp_path / "old.state"
    old_baseline_path.write_bytes(b"old")
    _write_x_cup_slot_state(store, run.id)
    config = _track_sampling_config(baseline_path=old_baseline_path)
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
    pending_status = refresh.refresh_status(config, force=True)
    assert pending_status.ready_for_reset

    def fail_store_read(_run_id: str) -> None:
        raise AssertionError("pending watch refresh should wait for the next interval")

    monkeypatch.setattr(refresh.store, "get_run_generated_x_cup_slots", fail_store_read)

    status = refresh.refresh_status(config, force=False)

    assert status.refreshed_config is None
    assert status.ready_for_reset


def test_live_track_sampling_state_refreshes_config_and_clears_stale_lock() -> None:
    current_config = _plain_track_sampling_config(course_id="mute_city")
    refreshed_config = _plain_track_sampling_config(course_id="silence")
    env = _FakeTrackSamplingEnv()
    state = _live_track_sampling_state(
        current_config,
        _FakeTrackSamplingRefresh(refreshed_config=refreshed_config),
        selected_reset_target_key="mute_city",
        locked_reset_target_key="mute_city",
    )

    assert state.refresh(env)

    assert state.active_track_sampling == refreshed_config
    assert state.selected_reset_target_key == "silence"
    assert state.locked_reset_target_key is None
    assert env.track_sampling_configs == [refreshed_config]
    assert env.locked_reset_courses == [None]


def test_live_track_sampling_state_rejects_unmaterialized_x_cup_refresh(
    tmp_path: Path,
) -> None:
    active_baseline = tmp_path / "active.state"
    active_baseline.write_bytes(b"active")
    current_config = _track_sampling_config(baseline_path=active_baseline)
    pending_config = _track_sampling_config(
        entry_id="x_cup_new",
        course_id="x_cup_new",
        course_name="X Cup new",
        course_hash="newhash",
        course_seed=1234,
        generation=3,
        baseline_path=None,
    )
    env = _FakeTrackSamplingEnv()
    state = _live_track_sampling_state(
        current_config,
        _FakeTrackSamplingRefresh(refreshed_config=pending_config),
    )

    assert not state.refresh(env, force=True)

    assert state.active_track_sampling == current_config
    assert env.track_sampling_configs == []
    assert env.preloaded_paths == []


def test_live_track_sampling_state_preloads_x_cup_before_apply(tmp_path: Path) -> None:
    active_baseline = tmp_path / "active.state"
    refreshed_baseline = tmp_path / "refreshed.state"
    active_baseline.write_bytes(b"active")
    refreshed_baseline.write_bytes(b"refreshed")
    current_config = _track_sampling_config(baseline_path=active_baseline)
    refreshed_config = _track_sampling_config(
        entry_id="x_cup_new",
        course_id="x_cup_new",
        course_name="X Cup new",
        course_hash="newhash",
        course_seed=1234,
        generation=3,
        baseline_path=refreshed_baseline,
    )
    env = _FakeTrackSamplingEnv()
    state = _live_track_sampling_state(
        current_config,
        _FakeTrackSamplingRefresh(refreshed_config=refreshed_config),
    )

    assert state.refresh(env, force=True)

    assert env.preloaded_paths == [(refreshed_baseline.resolve(),)]
    assert env.track_sampling_configs == [refreshed_config]


def test_live_track_sampling_state_preloads_active_x_cup_on_startup(tmp_path: Path) -> None:
    active_baseline = tmp_path / "active.state"
    active_baseline.write_bytes(b"active")
    current_config = _track_sampling_config(baseline_path=active_baseline)
    env = _FakeTrackSamplingEnv()
    state = _live_track_sampling_state(
        current_config,
        _FakeTrackSamplingRefresh(),
    )

    state.preload_generated_x_cup_baselines(env)

    assert env.preloaded_paths == [(active_baseline.resolve(),)]


def test_live_track_sampling_state_keeps_preloaded_x_cup_ready_if_file_disappears(
    tmp_path: Path,
) -> None:
    active_baseline = tmp_path / "active.state"
    active_baseline.write_bytes(b"active")
    current_config = _track_sampling_config(baseline_path=active_baseline)
    env = _FakeTrackSamplingEnv()
    state = _live_track_sampling_state(
        current_config,
        _FakeTrackSamplingRefresh(),
    )
    state.preload_generated_x_cup_baselines(env)
    active_baseline.unlink()

    assert state.ready_for_reset(env, force=True)


def test_live_track_sampling_state_waits_until_refresh_is_ready() -> None:
    config = _plain_track_sampling_config(course_id="mute_city")
    env = _FakeTrackSamplingEnv()
    refresh = _FakeTrackSamplingRefresh(ready_for_reset=False)
    state = _live_track_sampling_state(config, refresh)

    assert not state.ready_for_reset(env, force=True)

    assert refresh.status_forces == [True]
    assert env.track_sampling_configs == []
    assert env.locked_reset_courses == []


def test_managed_watch_track_sampling_refresh_repairs_missing_current_baseline(
    tmp_path: Path,
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
    missing_baseline_path = tmp_path / "stale" / "x_cup_new.state"
    baseline_path = run.run_dir / "baselines" / "x_cup_new.state"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_bytes(b"state")
    store.replace_run_track_sampling_artifacts(
        run_id=run.id,
        artifacts=(_x_cup_artifact(baseline_path),),
    )
    _write_x_cup_slot_state(store, run.id)
    config = _track_sampling_config(
        entry_id="x_cup_new",
        course_id="x_cup_new",
        course_name="X Cup new",
        course_hash="newhash",
        course_seed=1234,
        generation=3,
        baseline_path=missing_baseline_path,
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
    refreshed = refresh.refreshed_config(config, force=True)

    assert refreshed is not None
    assert refreshed.entries[0].baseline_state_path == baseline_path.resolve()


def test_managed_watch_save_creates_alt_baseline(tmp_path: Path) -> None:
    db_path = tmp_path / "manager" / "runs.db"
    store = ManagerStore(db_path)
    run = store.create_run(
        run_id="run",
        name="Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )

    result = _save_managed_alt_baseline(
        emulator=_FakeStateSavingEmulator(),
        manager_db_path=db_path,
        run_id=run.id,
        info={
            "track_course_key": "mute_city",
            "track_entry_id": "mute_city_gp_race_novice_blue_falcon",
            "track_mode": "gp_race",
            "track_gp_difficulty": "novice",
            "track_vehicle": "blue_falcon",
            "frame_index": 1234,
        },
    )

    assert result.handled
    assert result.saved
    assert result.state_path is not None
    assert result.state_path.read_bytes() == b"state"
    baseline = store.get_run_alt_baselines(run.id)[0]
    assert baseline.id == result.baseline_id
    assert baseline.course_key == "mute_city"
    assert baseline.source_entry_id == "mute_city_gp_race_novice_blue_falcon"
    assert baseline.label == "frame 1234"
    assert baseline.state_path == result.state_path.resolve()


def test_managed_watch_save_is_noop_for_generated_x_cup(tmp_path: Path) -> None:
    db_path = tmp_path / "manager" / "runs.db"
    store = ManagerStore(db_path)
    run = store.create_run(
        run_id="run",
        name="Run",
        config=_managed_x_cup_run_config(),
        managed_runs_root=tmp_path / "runs",
    )

    result = _save_managed_alt_baseline(
        emulator=_FakeStateSavingEmulator(),
        manager_db_path=db_path,
        run_id=run.id,
        info={
            "track_generated_course_kind": X_CUP_COURSE.generated_kind,
            "track_course_key": "x_cup_slot_1",
            "track_entry_id": "x_cup_slot_1_gp_race_novice_blue_falcon",
        },
    )

    assert result.handled
    assert not result.saved
    assert store.get_run_alt_baselines(run.id) == ()


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
                gp_difficulty="novice",
                vehicle="blue_falcon",
                generated_course_kind=X_CUP_COURSE.generated_kind,
                generated_course_seed=course_seed,
                generated_course_hash=course_hash,
                generated_course_slot=0,
                generated_course_generation=generation,
            ),
        ),
    )


def _plain_track_sampling_config(*, course_id: str) -> TrackSamplingConfig:
    return TrackSamplingConfig(
        enabled=True,
        entries=(
            TrackSamplingEntryConfig(
                id=f"{course_id}_entry",
                course_id=course_id,
                course_name=course_id,
                display_name=course_id,
                mode="time_attack",
                vehicle="blue_falcon",
            ),
        ),
    )


def _live_track_sampling_state(
    config: TrackSamplingConfig,
    refresh: _FakeTrackSamplingRefresh,
    *,
    selected_reset_target_key: str | None = None,
    locked_reset_target_key: str | None = None,
) -> LiveTrackSamplingState:
    rotation = WatchCourseRotation.from_entries(config.entries)
    return LiveTrackSamplingState(
        active_track_sampling=config,
        course_rotation=rotation,
        selected_reset_target_key=rotation.normalized_key(selected_reset_target_key),
        locked_reset_target_key=locked_reset_target_key,
        refresh_source=refresh,
    )


def _managed_x_cup_run_config():
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.race_mode = "gp_race"
    config.tracks.selected_course_ids = ()
    config.tracks.include_x_cup = True
    config.tracks.x_cup_course_count = 1
    return config


def _write_x_cup_slot_state(store: ManagerStore, run_id: str) -> None:
    store.replace_run_generated_x_cup_slots(
        run_id=run_id,
        slots=(
            GeneratedXCupSlot(
                course_key="x_cup_slot_1",
                slot=0,
                generation=3,
                course_id="x_cup_new",
                course_name="X Cup new",
                course_hash="newhash",
                course_seed=1234,
                segment_count=None,
                course_length=None,
            ),
        ),
    )


def _x_cup_artifact(baseline_path: Path) -> TrackSamplingMaterializedArtifact:
    return TrackSamplingMaterializedArtifact(
        course_key="x_cup_slot_1",
        reset_variant_key=reset_variant_key(
            mode="gp_race",
            gp_difficulty="novice",
            vehicle="blue_falcon",
        ),
        entry_id="x_cup_new_gp_race_novice_blue_falcon",
        baseline_state_path=baseline_path.resolve(),
        metadata_path=baseline_path.with_suffix(".json").resolve(),
        source_course_index=48,
        source_gp_difficulty="novice",
        source_vehicle="blue_falcon",
        source_engine_setting_raw_value=50,
        generated_course_slot=0,
        generated_course_generation=3,
        generated_course_id="x_cup_new",
        generated_course_name="X Cup new",
        generated_course_hash="newhash",
        generated_course_seed=1234,
        generated_course_segment_count=38,
        generated_course_length=61_743.98046875,
    )


def _stale_x_cup_artifact(baseline_path: Path) -> TrackSamplingMaterializedArtifact:
    return TrackSamplingMaterializedArtifact(
        course_key="x_cup_slot_1",
        reset_variant_key=reset_variant_key(
            mode="gp_race",
            gp_difficulty="novice",
            vehicle="blue_falcon",
        ),
        entry_id="x_cup_old_gp_race_novice_blue_falcon",
        baseline_state_path=baseline_path.resolve(),
        metadata_path=baseline_path.with_suffix(".json").resolve(),
        source_course_index=48,
        source_gp_difficulty="novice",
        source_vehicle="blue_falcon",
        source_engine_setting_raw_value=50,
        generated_course_slot=0,
        generated_course_generation=0,
        generated_course_id="x_cup_old",
        generated_course_name="X Cup old",
        generated_course_hash="oldhash",
        generated_course_seed=7,
        generated_course_segment_count=38,
        generated_course_length=61_743.98046875,
    )


def _touched(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


class _FakeStateSavingEmulator:
    def save_state(self, path: Path) -> None:
        path.write_bytes(b"state")


class _FakeTrackSamplingEnv:
    def __init__(self) -> None:
        self.track_sampling_configs: list[TrackSamplingConfig] = []
        self.locked_reset_courses: list[str | None] = []
        self.preloaded_paths: list[tuple[Path, ...]] = []

    def set_track_sampling_config(self, config: TrackSamplingConfig) -> None:
        self.track_sampling_configs.append(config)

    def set_locked_reset_course(self, target_key: str | None) -> None:
        self.locked_reset_courses.append(target_key)

    def preload_track_baseline_paths(self, paths: tuple[Path, ...]) -> None:
        self.preloaded_paths.append(paths)


class _FakeTrackSamplingRefresh:
    def __init__(
        self,
        *,
        refreshed_config: TrackSamplingConfig | None = None,
        ready_for_reset: bool = True,
    ) -> None:
        self._refreshed_config = refreshed_config
        self._ready_for_reset = ready_for_reset
        self.refresh_forces: list[bool] = []
        self.status_forces: list[bool] = []
        self.refresh_configs: list[TrackSamplingConfig] = []
        self.status_configs: list[TrackSamplingConfig] = []

    def refreshed_config(
        self,
        current_config: TrackSamplingConfig,
        *,
        force: bool = False,
    ) -> TrackSamplingConfig | None:
        self.refresh_configs.append(current_config)
        self.refresh_forces.append(force)
        return self._refreshed_config

    def refresh_status(
        self,
        current_config: TrackSamplingConfig,
        *,
        force: bool = False,
    ) -> TrackSamplingRefreshStatus:
        self.status_configs.append(current_config)
        self.status_forces.append(force)
        return TrackSamplingRefreshStatus(
            refreshed_config=self._refreshed_config,
            ready_for_reset=self._ready_for_reset,
        )
