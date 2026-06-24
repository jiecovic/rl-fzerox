# tests/core/training/test_x_cup_rotation.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from pytest import MonkeyPatch

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE, generated_x_cup_slot_key
from rl_fzerox.core.runtime_spec.schema import (
    EmulatorConfig,
    EnvConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    TrainConfig,
    XCupRotationConfig,
)
from rl_fzerox.core.training.runs import RunPaths, build_run_paths, ensure_run_dirs
from rl_fzerox.core.training.runs.baseline_materializer.models import (
    BaselineArtifact,
    BaselineRequest,
)
from rl_fzerox.core.training.runs.config import load_train_run_config
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.x_cup_rotation import (
    XCupRotationManager,
)


@dataclass(frozen=True, slots=True)
class XCupRotationFailureFixture:
    run_paths: RunPaths
    env_config: EnvConfig
    train_config: TrainAppConfig
    state: TrackSamplingRuntimeState
    old_state_path: Path
    slot_key: str


def test_x_cup_rotation_replaces_solved_slot_and_prunes_past_inactive_buffer(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="x-cup-rotation")
    ensure_run_dirs(run_paths)

    old_state_path = run_paths.baselines_dir / "old_x_cup.state"
    old_state_path.write_bytes(b"old")
    old_state_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "materializer_mode": X_CUP_COURSE.materializer_mode,
                "x_cup_course_hash": "old",
                "x_cup_generation": 1,
                "x_cup_seed": 1,
                "x_cup_slot": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    protected_group_paths = _write_x_cup_state_group(
        run_paths.baselines_dir,
        "protected_extra",
        timestamp=0,
        course_hash="protected-extra",
        seed=14,
        slot=13,
        generation=1,
        count=2,
    )
    oldest_stale_paths = _write_x_cup_state_group(
        run_paths.baselines_dir,
        "stale_oldest",
        timestamp=1,
        course_hash="stale-oldest",
        seed=11,
        slot=10,
        generation=1,
        count=2,
    )
    middle_stale_paths = _write_x_cup_state_group(
        run_paths.baselines_dir,
        "stale_middle",
        timestamp=2,
        course_hash="stale-middle",
        seed=12,
        slot=11,
        generation=1,
        count=2,
    )
    newest_stale_paths = _write_x_cup_state_group(
        run_paths.baselines_dir,
        "stale_newest",
        timestamp=3,
        course_hash="stale-newest",
        seed=13,
        slot=12,
        generation=1,
        count=2,
    )
    slot_key = generated_x_cup_slot_key(0)
    entry = TrackSamplingEntryConfig(
        id="x_cup_old",
        course_id="x_cup_old",
        runtime_course_key=slot_key,
        course_name="X Cup old",
        course_index=X_CUP_COURSE.course_index,
        mode=X_CUP_COURSE.race_mode,
        gp_difficulty="novice",
        vehicle="blue_falcon",
        engine_setting_raw_value=50,
        baseline_state_path=old_state_path,
        generated_course_kind=X_CUP_COURSE.generated_kind,
        generated_course_seed=1,
        generated_course_hash="old",
        generated_course_slot=0,
        generated_course_generation=1,
        log_per_course=False,
    )
    env_config = EnvConfig(
        track_sampling=TrackSamplingConfig(
            enabled=True,
            sampling_mode="adaptive_step_balanced",
            entries=(entry,),
            x_cup_rotation=XCupRotationConfig(
                enabled=True,
                completion_threshold=0.9,
                min_episodes=1,
            ),
        ),
    )
    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=env_config,
        train=TrainConfig(output_root=tmp_path / "runs", run_name="x-cup-rotation"),
    )
    new_state_path = run_paths.baselines_dir / "new_x_cup.state"

    def fake_materialize_baseline(*args: object, **kwargs: object) -> BaselineArtifact:
        del args, kwargs
        new_state_path.write_bytes(b"new")
        new_state_path.with_suffix(".json").write_text(
            json.dumps({"materializer_mode": X_CUP_COURSE.materializer_mode}) + "\n",
            encoding="utf-8",
        )
        return BaselineArtifact(
            state_path=new_state_path,
            metadata_path=new_state_path.with_suffix(".json"),
            cache_key="new-cache",
            source="generated",
            source_course_index=X_CUP_COURSE.course_index,
            source_vehicle="blue_falcon",
            source_gp_difficulty="novice",
            source_engine_setting_raw_value=50,
            generated_course_segment_count=123,
            generated_course_length=45_678.0,
        )

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.callbacks.track_sampling.x_cup_rotation.materialize_baseline",
        fake_materialize_baseline,
    )
    manager = XCupRotationManager(config=train_config, run_paths=run_paths, cache_root=tmp_path)
    state = TrackSamplingRuntimeState(
        sampling_mode="adaptive_step_balanced",
        action_repeat=2,
        update_episodes=1,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
        adaptive_min_confidence_episodes=1,
        adaptive_confidence_scale=1.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id=slot_key,
                course_key=slot_key,
                label="X Cup old",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=100,
                episode_count=3,
                finished_episode_count=3,
                success_sample_count=3,
                ema_episode_frames=100.0,
                ema_completion_fraction=0.95,
                generation_episode_count=3,
                generation_finished_episode_count=3,
                generation_success_sample_count=3,
                generation_ema_completion_fraction=0.95,
                generated_course_slot=0,
                generated_course_generation=1,
                generated_course_id="x_cup_old",
                generated_course_name="X Cup old",
                generated_course_hash="old",
                generated_course_seed=1,
            ),
        ),
    )

    update = manager.rotate_once(env_config=env_config, state=state)

    assert update is not None
    replacement = update.env_config.track_sampling.entries[0]
    assert replacement.course_id is not None
    assert replacement.course_id != "x_cup_old"
    assert replacement.runtime_course_key == slot_key
    assert replacement.generated_course_slot == 0
    assert replacement.generated_course_generation == 2
    assert replacement.baseline_state_path == new_state_path
    assert replacement.generated_course_segment_count == 123
    assert len(update.materialized_artifacts) == 1
    assert update.materialized_artifacts[0].baseline_state_path == new_state_path.resolve()
    assert update.materialized_artifacts[0].source_gp_difficulty == "novice"
    assert update.materialized_artifacts[0].source_vehicle == "blue_falcon"
    protected_artifact = replace(
        update.materialized_artifacts[0],
        baseline_state_path=protected_group_paths[0].resolve(),
        metadata_path=protected_group_paths[0].with_suffix(".json").resolve(),
        generated_course_hash="protected-extra",
        generated_course_seed=14,
        generated_course_slot=13,
        generated_course_generation=1,
    )
    update = replace(
        update,
        materialized_artifacts=(*update.materialized_artifacts, protected_artifact),
    )

    manager.commit(update)

    saved = load_train_run_config(run_paths.run_dir)
    assert saved.env.track_sampling.entries[0].course_id == replacement.course_id
    assert saved.env.track_sampling.entries[0].runtime_course_key == slot_key
    assert all(not path.exists() for path in oldest_stale_paths)
    assert all(not path.with_suffix(".json").exists() for path in oldest_stale_paths)
    assert all(not path.exists() for path in middle_stale_paths)
    assert all(not path.with_suffix(".json").exists() for path in middle_stale_paths)
    assert all(path.exists() for path in newest_stale_paths)
    assert all(path.exists() for path in protected_group_paths)
    assert old_state_path.exists()
    assert new_state_path.exists()


def test_x_cup_rotation_materialization_failure_defers_retry(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    fixture = _x_cup_rotation_failure_fixture(tmp_path, run_name="x-cup-retry")
    materialization_attempts = 0

    def fake_materialize_baseline(*args: object, **kwargs: object) -> BaselineArtifact:
        del args, kwargs
        nonlocal materialization_attempts
        materialization_attempts += 1
        raise RuntimeError("eglMakeCurrent failed with 0x3003")

    now = 100.0

    def fake_monotonic() -> float:
        return now

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.callbacks.track_sampling.x_cup_rotation.materialize_baseline",
        fake_materialize_baseline,
    )
    monkeypatch.setattr(
        "rl_fzerox.core.training.session.callbacks.track_sampling.x_cup_rotation.monotonic",
        fake_monotonic,
    )
    manager = XCupRotationManager(
        config=fixture.train_config,
        run_paths=fixture.run_paths,
        cache_root=tmp_path,
        persist_manifest_on_commit=False,
        materialization_retry_delay_seconds=10.0,
    )

    assert manager.rotate_once(env_config=fixture.env_config, state=fixture.state) is None
    assert materialization_attempts == 1
    failure = manager.materialization_failure(fixture.slot_key)
    assert failure is not None
    assert failure.kind == "retryable"
    assert fixture.env_config.track_sampling.entries[0].course_id == "x_cup_old"
    assert fixture.old_state_path.exists()

    assert manager.rotate_once(env_config=fixture.env_config, state=fixture.state) is None
    assert materialization_attempts == 1

    now = 111.0
    assert manager.rotate_once(env_config=fixture.env_config, state=fixture.state) is None
    assert materialization_attempts == 2


def test_x_cup_rotation_permanent_materialization_failure_blocks_retry(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    fixture = _x_cup_rotation_failure_fixture(tmp_path, run_name="x-cup-blocked")
    materialization_attempts = 0

    def fake_materialize_baseline(*args: object, **kwargs: object) -> BaselineArtifact:
        del args, kwargs
        nonlocal materialization_attempts
        materialization_attempts += 1
        raise ValueError("generated X Cup replacement requires one concrete slot")

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.callbacks.track_sampling.x_cup_rotation.materialize_baseline",
        fake_materialize_baseline,
    )
    manager = XCupRotationManager(
        config=fixture.train_config,
        run_paths=fixture.run_paths,
        cache_root=tmp_path,
        persist_manifest_on_commit=False,
        materialization_retry_delay_seconds=10.0,
    )

    assert manager.rotate_once(env_config=fixture.env_config, state=fixture.state) is None
    assert materialization_attempts == 1
    failure = manager.materialization_failure(fixture.slot_key)
    assert failure is not None
    assert failure.kind == "blocked"
    assert "one concrete slot" in failure.message

    assert manager.rotate_once(env_config=fixture.env_config, state=fixture.state) is None
    assert materialization_attempts == 1


def test_x_cup_rotation_preserves_duplicate_slot_id_difficulty_entries(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="x-cup-variants")
    ensure_run_dirs(run_paths)
    slot_key = generated_x_cup_slot_key(0)
    old_state_path = run_paths.baselines_dir / "old_x_cup.state"
    old_state_path.write_bytes(b"old")
    old_entries = tuple(
        TrackSamplingEntryConfig(
            id=slot_key,
            course_id="x_cup_old",
            runtime_course_key=slot_key,
            course_name="X Cup old",
            course_index=X_CUP_COURSE.course_index,
            mode=X_CUP_COURSE.race_mode,
            gp_difficulty=difficulty,
            vehicle="blue_falcon",
            engine_setting_raw_value=50,
            baseline_state_path=old_state_path,
            generated_course_kind=X_CUP_COURSE.generated_kind,
            generated_course_seed=1,
            generated_course_hash="old",
            generated_course_slot=0,
            generated_course_generation=1,
            log_per_course=False,
        )
        for difficulty in ("expert", "novice")
    )
    env_config = EnvConfig(
        track_sampling=TrackSamplingConfig(
            enabled=True,
            sampling_mode="adaptive_step_balanced",
            entries=old_entries,
            x_cup_rotation=XCupRotationConfig(
                enabled=True,
                completion_threshold=0.9,
                min_episodes=1,
            ),
        ),
    )
    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=env_config,
        train=TrainConfig(output_root=tmp_path / "runs", run_name="x-cup-variants"),
    )
    materialized_difficulties: list[str] = []

    def fake_materialize_baseline(
        request: BaselineRequest,
        *args: object,
        **kwargs: object,
    ) -> BaselineArtifact:
        del args, kwargs
        difficulty = request.gp_difficulty
        assert difficulty is not None
        materialized_difficulties.append(difficulty)
        state_path = run_paths.baselines_dir / f"new_{difficulty}.state"
        state_path.write_bytes(f"new-{difficulty}".encode())
        state_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "materializer_mode": X_CUP_COURSE.materializer_mode,
                    "source_gp_difficulty": difficulty,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return BaselineArtifact(
            state_path=state_path,
            metadata_path=state_path.with_suffix(".json"),
            cache_key=f"new-cache-{difficulty}",
            source="generated",
            source_course_index=X_CUP_COURSE.course_index,
            source_vehicle="blue_falcon",
            source_gp_difficulty=difficulty,
            source_engine_setting_raw_value=50,
        )

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.callbacks.track_sampling.x_cup_rotation.materialize_baseline",
        fake_materialize_baseline,
    )
    state = TrackSamplingRuntimeState(
        sampling_mode="adaptive_step_balanced",
        action_repeat=2,
        update_episodes=1,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
        adaptive_min_confidence_episodes=1,
        adaptive_confidence_scale=1.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id=slot_key,
                course_key=slot_key,
                label="X Cup old",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=100,
                episode_count=3,
                finished_episode_count=3,
                success_sample_count=3,
                ema_episode_frames=100.0,
                ema_completion_fraction=0.95,
                generation_episode_count=3,
                generation_finished_episode_count=3,
                generation_success_sample_count=3,
                generation_ema_completion_fraction=0.95,
                generated_course_slot=0,
                generated_course_generation=1,
                generated_course_id="x_cup_old",
                generated_course_name="X Cup old",
                generated_course_hash="old",
                generated_course_seed=1,
            ),
        ),
    )

    manager = XCupRotationManager(
        config=train_config,
        run_paths=run_paths,
        cache_root=tmp_path,
        persist_manifest_on_commit=False,
    )
    update = manager.rotate_once(env_config=env_config, state=state)

    assert update is not None
    replacement_entries = update.env_config.track_sampling.entries
    assert len(replacement_entries) == 2
    assert len({entry.id for entry in replacement_entries}) == 2
    assert all(entry.id.startswith("x_cup_") for entry in replacement_entries)
    assert [entry.runtime_course_key for entry in replacement_entries] == [slot_key, slot_key]
    assert [entry.gp_difficulty for entry in replacement_entries] == ["expert", "novice"]
    assert [entry.source_gp_difficulty for entry in replacement_entries] == [
        "expert",
        "novice",
    ]
    assert len({entry.baseline_state_path for entry in replacement_entries}) == 2
    assert materialized_difficulties == ["expert", "novice"]


def test_x_cup_rotation_replaces_hard_slot_at_episode_cap(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="x-cup-cap")
    ensure_run_dirs(run_paths)

    old_state_path = run_paths.baselines_dir / "old_x_cup.state"
    old_state_path.write_bytes(b"old")
    old_state_path.with_suffix(".json").write_text(
        json.dumps({"materializer_mode": X_CUP_COURSE.materializer_mode}) + "\n",
        encoding="utf-8",
    )
    slot_key = generated_x_cup_slot_key(0)
    entry = TrackSamplingEntryConfig(
        id="x_cup_old",
        course_id="x_cup_old",
        runtime_course_key=slot_key,
        course_name="X Cup old",
        course_index=X_CUP_COURSE.course_index,
        mode=X_CUP_COURSE.race_mode,
        gp_difficulty="novice",
        vehicle="blue_falcon",
        engine_setting_raw_value=50,
        baseline_state_path=old_state_path,
        generated_course_kind=X_CUP_COURSE.generated_kind,
        generated_course_seed=1,
        generated_course_hash="old",
        generated_course_slot=0,
        generated_course_generation=1,
        log_per_course=False,
    )
    env_config = EnvConfig(
        track_sampling=TrackSamplingConfig(
            enabled=True,
            sampling_mode="adaptive_step_balanced",
            entries=(entry,),
            x_cup_rotation=XCupRotationConfig(
                enabled=True,
                completion_threshold=0.9,
                min_episodes=3,
                max_episodes=5,
            ),
        ),
    )
    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=env_config,
        train=TrainConfig(output_root=tmp_path / "runs", run_name="x-cup-cap"),
    )
    new_state_path = run_paths.baselines_dir / "new_x_cup.state"

    def fake_materialize_baseline(*args: object, **kwargs: object) -> BaselineArtifact:
        del args, kwargs
        new_state_path.write_bytes(b"new")
        new_state_path.with_suffix(".json").write_text(
            json.dumps({"materializer_mode": X_CUP_COURSE.materializer_mode}) + "\n",
            encoding="utf-8",
        )
        return BaselineArtifact(
            state_path=new_state_path,
            metadata_path=new_state_path.with_suffix(".json"),
            cache_key="new-cache",
            source="generated",
            source_course_index=X_CUP_COURSE.course_index,
            source_vehicle="blue_falcon",
            source_gp_difficulty="novice",
            source_engine_setting_raw_value=50,
        )

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.callbacks.track_sampling.x_cup_rotation.materialize_baseline",
        fake_materialize_baseline,
    )
    state = TrackSamplingRuntimeState(
        sampling_mode="adaptive_step_balanced",
        action_repeat=2,
        update_episodes=1,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
        adaptive_min_confidence_episodes=1,
        adaptive_confidence_scale=1.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id=slot_key,
                course_key=slot_key,
                label="X Cup old",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=100,
                episode_count=5,
                finished_episode_count=0,
                success_sample_count=5,
                ema_episode_frames=100.0,
                ema_completion_fraction=0.2,
                generation_episode_count=5,
                generation_finished_episode_count=0,
                generation_success_sample_count=5,
                generation_ema_completion_fraction=0.2,
                generated_course_slot=0,
                generated_course_generation=1,
                generated_course_id="x_cup_old",
                generated_course_name="X Cup old",
                generated_course_hash="old",
                generated_course_seed=1,
            ),
        ),
    )

    manager = XCupRotationManager(
        config=train_config,
        run_paths=run_paths,
        cache_root=tmp_path,
        persist_manifest_on_commit=False,
    )
    update = manager.rotate_once(env_config=env_config, state=state)

    assert update is not None
    replacement = update.env_config.track_sampling.entries[0]
    assert replacement.course_id is not None
    assert replacement.course_id != "x_cup_old"
    assert replacement.runtime_course_key == slot_key
    assert replacement.generated_course_generation == 2
    assert replacement.baseline_state_path == new_state_path

    manager.commit(update)

    assert not (run_paths.run_dir / "train_manifest.yaml").exists()


def _x_cup_rotation_failure_fixture(
    tmp_path: Path,
    *,
    run_name: str,
) -> XCupRotationFailureFixture:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name=run_name)
    ensure_run_dirs(run_paths)

    slot_key = generated_x_cup_slot_key(0)
    old_state_path = run_paths.baselines_dir / "old_x_cup.state"
    old_state_path.write_bytes(b"old")
    entry = TrackSamplingEntryConfig(
        id="x_cup_old",
        course_id="x_cup_old",
        runtime_course_key=slot_key,
        course_name="X Cup old",
        course_index=X_CUP_COURSE.course_index,
        mode=X_CUP_COURSE.race_mode,
        gp_difficulty="novice",
        vehicle="blue_falcon",
        engine_setting_raw_value=50,
        baseline_state_path=old_state_path,
        generated_course_kind=X_CUP_COURSE.generated_kind,
        generated_course_seed=1,
        generated_course_hash="old",
        generated_course_slot=0,
        generated_course_generation=1,
        log_per_course=False,
    )
    env_config = EnvConfig(
        track_sampling=TrackSamplingConfig(
            enabled=True,
            sampling_mode="adaptive_step_balanced",
            entries=(entry,),
            x_cup_rotation=XCupRotationConfig(
                enabled=True,
                completion_threshold=0.9,
                min_episodes=1,
            ),
        ),
    )
    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=env_config,
        train=TrainConfig(output_root=tmp_path / "runs", run_name=run_name),
    )
    state = TrackSamplingRuntimeState(
        sampling_mode="adaptive_step_balanced",
        action_repeat=2,
        update_episodes=1,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
        adaptive_min_confidence_episodes=1,
        adaptive_confidence_scale=1.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id=slot_key,
                course_key=slot_key,
                label="X Cup old",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=100,
                episode_count=3,
                finished_episode_count=3,
                success_sample_count=3,
                ema_episode_frames=100.0,
                ema_completion_fraction=0.95,
                generation_episode_count=3,
                generation_finished_episode_count=3,
                generation_success_sample_count=3,
                generation_ema_completion_fraction=0.95,
                generated_course_slot=0,
                generated_course_generation=1,
                generated_course_id="x_cup_old",
                generated_course_name="X Cup old",
                generated_course_hash="old",
                generated_course_seed=1,
            ),
        ),
    )
    return XCupRotationFailureFixture(
        run_paths=run_paths,
        env_config=env_config,
        train_config=train_config,
        state=state,
        old_state_path=old_state_path,
        slot_key=slot_key,
    )


def _write_x_cup_state_group(
    baselines_dir: Path,
    prefix: str,
    *,
    timestamp: int,
    course_hash: str,
    seed: int,
    slot: int,
    generation: int,
    count: int,
) -> tuple[Path, ...]:
    paths = tuple(baselines_dir / f"{prefix}_{index}.state" for index in range(count))
    for path in paths:
        path.write_bytes(b"stale")
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "materializer_mode": X_CUP_COURSE.materializer_mode,
                    "x_cup_course_hash": course_hash,
                    "x_cup_generation": generation,
                    "x_cup_seed": seed,
                    "x_cup_slot": slot,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        os.utime(path, (timestamp, timestamp))
        os.utime(path.with_suffix(".json"), (timestamp, timestamp))
    return paths
