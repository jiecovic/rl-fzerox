# tests/core/training/test_x_cup_rotation.py
from __future__ import annotations

import json
import os
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
from rl_fzerox.core.training.runs import build_run_paths, ensure_run_dirs
from rl_fzerox.core.training.runs.baseline_materializer.models import BaselineArtifact
from rl_fzerox.core.training.runs.config import load_train_run_config
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.x_cup_rotation import (
    XCupRotationManager,
)


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
        json.dumps({"materializer_mode": X_CUP_COURSE.materializer_mode}) + "\n",
        encoding="utf-8",
    )
    stale_paths = tuple(
        _write_x_cup_state(run_paths.baselines_dir / f"stale_{index}.state", timestamp=index)
        for index in range(3)
    )
    slot_key = generated_x_cup_slot_key(0)
    entry = TrackSamplingEntryConfig(
        id="x_cup_old_gp_race_novice_blue_falcon_balanced",
        course_id="x_cup_old",
        runtime_course_key=slot_key,
        course_name="X Cup old",
        course_index=X_CUP_COURSE.course_index,
        mode=X_CUP_COURSE.race_mode,
        gp_difficulty="novice",
        vehicle="blue_falcon",
        engine_setting="balanced",
        engine_setting_raw_value=50,
        baseline_state_path=old_state_path,
        generated_course_kind=X_CUP_COURSE.generated_kind,
        generated_course_seed=1,
        generated_course_hash="old",
        generated_course_slot=0,
        generated_course_generation=0,
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
            source_course_index=X_CUP_COURSE.course_index,
            source_vehicle="blue_falcon",
            source_gp_difficulty="novice",
            source_engine_setting="balanced",
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
                generated_course_generation=0,
                generated_entry_id="x_cup_old_gp_race_novice_blue_falcon_balanced",
                generated_course_id="x_cup_old",
                generated_course_name="X Cup old",
                generated_course_hash="old",
                generated_course_seed=1,
                generated_baseline_state_path=str(old_state_path),
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
    assert replacement.generated_course_generation == 1
    assert replacement.baseline_state_path == new_state_path
    assert replacement.generated_course_segment_count == 123

    manager.commit(update)

    saved = load_train_run_config(run_paths.run_dir)
    assert saved.env.track_sampling.entries[0].course_id == replacement.course_id
    assert saved.env.track_sampling.entries[0].runtime_course_key == slot_key
    assert not stale_paths[0].exists()
    assert not stale_paths[0].with_suffix(".json").exists()
    assert not stale_paths[1].exists()
    assert stale_paths[2].exists()
    assert old_state_path.exists()
    assert new_state_path.exists()


def _write_x_cup_state(path: Path, *, timestamp: int) -> Path:
    path.write_bytes(b"stale")
    path.with_suffix(".json").write_text(
        json.dumps({"materializer_mode": X_CUP_COURSE.materializer_mode}) + "\n",
        encoding="utf-8",
    )
    os.utime(path, (timestamp, timestamp))
    os.utime(path.with_suffix(".json"), (timestamp, timestamp))
    return path
