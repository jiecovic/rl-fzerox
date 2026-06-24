# tests/core/manager/test_manager_store_runs_runtime.py
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sqlalchemy import event
from sqlalchemy.engine import Engine

from rl_fzerox.core.manager import (
    ManagerStore,
    default_managed_run_config,
    new_managed_run_id,
)
from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingAltBaseline,
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)
from tests.core.manager.manager_store_support import (
    _track_sampling_artifact,
)

SnapshotKind = Literal["run", "draft", "template", "import"]


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


def test_manager_store_hides_archived_runs_from_visible_views(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    archived_dir = tmp_path / "runs" / "archived"
    active_dir = tmp_path / "runs" / "active"
    (archived_dir / "tensorboard").mkdir(parents=True)
    (active_dir / "tensorboard").mkdir(parents=True)
    archived = store.create_run(
        name="Archived Run",
        config=default_managed_run_config(),
        explicit_run_dir=archived_dir,
    )
    active = store.create_run(
        name="Active Run",
        config=default_managed_run_config(),
        explicit_run_dir=active_dir,
    )
    archived_status = store.update_run_status(
        run_id=archived.id,
        status="archived",
        message="run archived",
    )
    active_status = store.update_run_status(
        run_id=active.id,
        status="stopped",
        message="run stopped",
    )

    assert archived_status is not None
    assert active_status is not None
    archived_run = store.get_run(archived.id)

    assert archived_run is not None
    assert archived_run.status == "archived"
    assert tuple(run.id for run in store.list_visible_runs()) == (active.id,)
    assert tuple(run.id for run in store.list_visible_run_summaries()) == (active.id,)
    groups = store.rebuild_tensorboard_views()
    tensorboard_targets = tuple(
        path.resolve() for path in store.tensorboard_views_root().rglob("*") if path.is_symlink()
    )

    assert sum(group.run_count for group in groups) == 1
    assert active_dir / "tensorboard" in tensorboard_targets
    assert archived_dir / "tensorboard" not in tensorboard_targets


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


def test_manager_store_persists_track_sampling_runtime_state(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Track Pool State Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    state = TrackSamplingRuntimeState(
        sampling_mode="step_balanced",
        action_repeat=2,
        update_episodes=5,
        ema_alpha=0.1,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.8,
        adaptive_target_completion=0.5,
        adaptive_min_confidence_episodes=12,
        adaptive_confidence_scale=3.0,
        update_count=7,
        episodes_since_update=2,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id="x_cup_slot_1",
                course_key="x_cup_slot_1",
                label="X Cup 1234abcd",
                base_weight=1.0,
                current_weight=2.0,
                completed_frames=1000,
                episode_count=4,
                finished_episode_count=1,
                success_sample_count=4,
                ema_episode_frames=250.0,
                ema_completion_fraction=0.75,
                generation_episode_count=2,
                generation_finished_episode_count=1,
                generation_success_sample_count=2,
                generation_ema_completion_fraction=0.8,
                generated_course_slot=1,
                generated_course_generation=3,
                generated_course_id="x_cup_1234abcd",
                generated_course_name="X Cup 1234abcd",
                generated_course_hash="1234abcd",
                generated_course_seed=12_647_406_722_013_964_192,
                generated_course_segment_count=128,
                generated_course_length=12345.0,
            ),
        ),
    )

    store.upsert_run_track_sampling_state(run_id=run.id, state=state)
    recovered = ManagerStore(store.db_path).get_run_track_sampling_state(run.id)

    assert recovered is not None
    assert recovered == state
    assert recovered.entries[0].generated_course_seed == 12_647_406_722_013_964_192

    store.clear_run_track_sampling_state(run.id)

    assert store.get_run_track_sampling_state(run.id) is None


def test_manager_store_persists_run_alt_baselines(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Alt Baseline Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    state_path = (run.run_dir / "baselines" / "alt" / "alt-a.state").resolve()
    state_path.parent.mkdir(parents=True)
    state_path.write_bytes(b"state")
    baseline = TrackSamplingAltBaseline(
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

    store.upsert_run_alt_baseline(baseline=baseline)

    recovered_store = ManagerStore(store.db_path)
    assert recovered_store.get_run_alt_baselines(run.id) == (baseline,)
    assert recovered_store.active_run_alt_baselines(run.id) == (baseline,)
    assert store.delete_run_alt_baseline(
        run_id=run.id,
        baseline_id=baseline.id,
    )
    assert store.get_run_alt_baselines(run.id) == ()
    assert not state_path.exists()


def test_manager_store_active_run_alt_baselines_require_state_file(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Missing Alt Baseline Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    baseline = TrackSamplingAltBaseline(
        id="alt-missing",
        run_id=run.id,
        course_key="mute_city",
        reset_variant_key="gp_race|novice|blue_falcon",
        source_entry_id="mute_city_gp_race_novice_blue_falcon",
        label="missing state",
        state_path=run.run_dir / "baselines" / "alt" / "alt-missing.state",
        weight=1.0,
        enabled=True,
        created_at="2026-06-13T10:00:00+00:00",
        updated_at="2026-06-13T10:00:00+00:00",
    )

    store.upsert_run_alt_baseline(baseline=baseline)

    assert store.get_run_alt_baselines(run.id) == (baseline,)
    assert store.active_run_alt_baselines(run.id) == ()


def test_manager_store_clears_run_alt_baselines_from_database_and_disk(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Alt Baseline Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    state_paths = tuple(
        run.run_dir / "baselines" / "alt" / f"alt-{index}.state" for index in range(2)
    )
    for state_path in state_paths:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(b"state")
    for index, state_path in enumerate(state_paths):
        store.upsert_run_alt_baseline(
            baseline=TrackSamplingAltBaseline(
                id=f"alt-{index}",
                run_id=run.id,
                course_key="mute_city",
                reset_variant_key="gp_race|novice|blue_falcon",
                source_entry_id="mute_city_gp_race_novice_blue_falcon",
                label=f"alt {index}",
                state_path=state_path,
                weight=1.0,
                enabled=True,
                created_at=f"2026-06-13T10:0{index}:00+00:00",
                updated_at=f"2026-06-13T10:0{index}:00+00:00",
            )
        )

    assert store.clear_run_alt_baselines(run.id) == 2

    assert store.get_run_alt_baselines(run.id) == ()
    assert all(not state_path.exists() for state_path in state_paths)


def test_manager_store_clears_run_alt_baselines_for_one_course(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Alt Baseline Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    mute_city_path = run.run_dir / "baselines" / "alt" / "alt-mute-city.state"
    silence_path = run.run_dir / "baselines" / "alt" / "alt-silence.state"
    for state_path in (mute_city_path, silence_path):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(b"state")
    store.upsert_run_alt_baseline(
        baseline=TrackSamplingAltBaseline(
            id="alt-mute-city",
            run_id=run.id,
            course_key="mute_city",
            reset_variant_key="gp_race|novice|blue_falcon",
            source_entry_id="mute_city_gp_race_novice_blue_falcon",
            label="mute city chicane",
            state_path=mute_city_path,
            weight=1.0,
            enabled=True,
            created_at="2026-06-13T10:00:00+00:00",
            updated_at="2026-06-13T10:00:00+00:00",
        )
    )
    store.upsert_run_alt_baseline(
        baseline=TrackSamplingAltBaseline(
            id="alt-silence",
            run_id=run.id,
            course_key="silence",
            reset_variant_key="gp_race|novice|blue_falcon",
            source_entry_id="silence_gp_race_novice_blue_falcon",
            label="silence jump",
            state_path=silence_path,
            weight=1.0,
            enabled=True,
            created_at="2026-06-13T10:01:00+00:00",
            updated_at="2026-06-13T10:01:00+00:00",
        )
    )

    assert (
        store.clear_run_alt_baselines_for_course(
            run_id=run.id,
            course_key="mute_city",
        )
        == 1
    )

    remaining = store.get_run_alt_baselines(run.id)
    assert len(remaining) == 1
    assert remaining[0].id == "alt-silence"
    assert not mute_city_path.exists()
    assert silence_path.exists()


def test_manager_store_replaces_track_sampling_artifact_rows(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Artifact State Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    first_path = run.run_dir / "baselines" / "x_cup_first.state"
    second_path = run.run_dir / "baselines" / "x_cup_second.state"
    first_artifact = _track_sampling_artifact(first_path, difficulty="novice")
    second_artifact = _track_sampling_artifact(second_path, difficulty="expert")

    store.replace_run_track_sampling_artifacts(
        run_id=run.id,
        artifacts=(first_artifact, second_artifact),
    )

    assert ManagerStore(store.db_path).get_run_track_sampling_artifacts(run.id) == (
        second_artifact,
        first_artifact,
    )

    store.replace_run_track_sampling_artifacts(
        run_id=run.id,
        artifacts=(first_artifact,),
    )

    assert store.get_run_track_sampling_artifacts(run.id) == (first_artifact,)

    store.clear_run_track_sampling_state(run.id)

    assert store.get_run_track_sampling_artifacts(run.id) == ()


def test_manager_store_replaces_generated_x_cup_slot_rows(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Generated Slot State Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    first_slot = GeneratedXCupSlot(
        course_key="x_cup_slot_1",
        slot=1,
        generation=3,
        course_id="x_cup_1234abcd",
        course_name="X Cup 1234abcd",
        course_hash="1234abcd",
        course_seed=1234,
        segment_count=38,
        course_length=61_743.98046875,
    )
    second_slot = GeneratedXCupSlot(
        course_key="x_cup_slot_2",
        slot=2,
        generation=1,
        course_id="x_cup_abcd1234",
        course_name="X Cup abcd1234",
        course_hash="abcd1234",
        course_seed=5678,
        segment_count=None,
        course_length=None,
    )

    store.replace_run_generated_x_cup_slots(
        run_id=run.id,
        slots=(second_slot, first_slot),
    )

    assert ManagerStore(store.db_path).get_run_generated_x_cup_slots(run.id) == (
        first_slot,
        second_slot,
    )

    store.replace_run_generated_x_cup_slots(run_id=run.id, slots=(second_slot,))

    assert store.get_run_generated_x_cup_slots(run.id) == (second_slot,)

    store.clear_run_track_sampling_state(run.id)

    assert store.get_run_generated_x_cup_slots(run.id) == ()


def test_manager_store_updates_track_sampling_rows_incrementally(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        name="Incremental Track Pool State Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    entry = TrackSamplingRuntimeEntry(
        track_id="mute",
        course_key="mute",
        label="Mute City",
        base_weight=1.0,
        current_weight=1.0,
        completed_frames=100,
        episode_count=1,
        finished_episode_count=0,
        success_sample_count=1,
        ema_episode_frames=100.0,
        ema_completion_fraction=0.4,
    )
    state = TrackSamplingRuntimeState(
        sampling_mode="step_balanced",
        action_repeat=2,
        update_episodes=5,
        ema_alpha=0.1,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.8,
        adaptive_target_completion=0.5,
        adaptive_min_confidence_episodes=12,
        adaptive_confidence_scale=3.0,
        update_count=1,
        episodes_since_update=1,
        entries=(entry,),
    )
    store.upsert_run_track_sampling_state(run_id=run.id, state=state)

    statements: list[str] = []

    def record_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    updated = replace(
        state,
        update_count=2,
        entries=(
            replace(
                entry,
                current_weight=1.5,
                completed_frames=250,
                episode_count=2,
                ema_completion_fraction=0.55,
            ),
        ),
    )
    event.listen(Engine, "before_cursor_execute", record_statement)
    try:
        store.upsert_run_track_sampling_state(run_id=run.id, state=updated)
    finally:
        event.remove(Engine, "before_cursor_execute", record_statement)

    normalized_statements = tuple(statement.strip().lower() for statement in statements)
    entry_selects = tuple(
        statement
        for statement in normalized_statements
        if statement.startswith("select") and "run_track_sampling_entries" in statement
    )
    entry_deletes = tuple(
        statement
        for statement in normalized_statements
        if statement.startswith("delete from run_track_sampling_entries")
    )
    entry_upserts = tuple(
        statement
        for statement in normalized_statements
        if statement.startswith("insert into run_track_sampling_entries")
    )
    assert not entry_selects
    assert all("not in" in statement for statement in entry_deletes)
    assert any("on conflict" in statement for statement in entry_upserts)
    assert store.get_run_track_sampling_state(run.id) == updated
