# tests/core/training/test_training_xcup_artifacts.py
from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE, generated_x_cup_slot_key
from rl_fzerox.core.manager.projection.x_cup_runtime import (
    restore_generated_x_cup_entries_from_slots,
)
from rl_fzerox.core.runtime_spec.schema import (
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    TrainConfig,
    XCupRotationConfig,
)
from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
from rl_fzerox.core.training.runs import (
    build_run_paths,
    continue_run_paths,
    ensure_run_dirs,
    materialize_train_run_config,
)
from rl_fzerox.core.training.runs.baseline_materializer.materialization import (
    baselines as baseline_materialization,
)
from rl_fzerox.core.training.runs.baseline_materializer.settings import (
    BASELINE_MATERIALIZER_SETTINGS,
)
from rl_fzerox.core.training.runs.race_start.x_cup import XCupMaterializedCourse
from tests.core.training.training_artifacts_support import (
    _required_baseline_path,
)


def test_materialize_train_run_config_rewrites_generated_x_cup_baselines(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    def fake_ensure_x_cup_baseline(
        *,
        label: str,
        seed: int,
        course_hash: str,
        gp_difficulty: object,
        vehicle_id: str,
        camera_setting: str | None,
        cache_root: Path,
        context: object,
        emulator_type: object,
    ) -> tuple[Path, XCupMaterializedCourse]:
        del label, gp_difficulty, vehicle_id, camera_setting, context, emulator_type
        cache_state_path = cache_root / f"x-cup-{seed}-{course_hash}.state"
        cache_state_path.parent.mkdir(parents=True, exist_ok=True)
        cache_state_path.write_bytes(b"x-cup-generated")
        cache_state_path.with_suffix(".json").write_text(
            json.dumps({"materialized_state_sha256": "fake-x-cup-sha"}, indent=2) + "\n",
            encoding="utf-8",
        )
        return cache_state_path, XCupMaterializedCourse(
            segment_count=212,
            course_length=91_234.5,
        )

    monkeypatch.setattr(
        baseline_materialization,
        "ensure_x_cup_baseline",
        fake_ensure_x_cup_baseline,
    )

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="equal",
                entries=(
                    TrackSamplingEntryConfig(
                        id="x_cup_abcd1234",
                        course_id="x_cup_abcd1234",
                        course_name="X Cup abcd1234",
                        course_index=X_CUP_COURSE.course_index,
                        mode=X_CUP_COURSE.race_mode,
                        gp_difficulty="novice",
                        vehicle="blue_falcon",
                        engine_setting_raw_value=50,
                        generated_course_kind=X_CUP_COURSE.generated_kind,
                        generated_course_seed=1234,
                        generated_course_hash="abcd1234",
                        generated_course_slot=0,
                        generated_course_generation=2,
                        log_per_course=False,
                    ),
                ),
            ),
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(run_paths)

    materialized = materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    entry = materialized.env.track_sampling.entries[0]
    baseline_path = _required_baseline_path(entry)
    metadata = json.loads(baseline_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert baseline_path.parent == run_paths.baselines_dir
    assert baseline_path.read_bytes() == b"x-cup-generated"
    assert entry.source_course_index == X_CUP_COURSE.course_index
    assert entry.generated_course_segment_count == 212
    assert entry.generated_course_length == 91_234.5
    assert entry.log_per_course is False
    assert metadata["materializer_mode"] == X_CUP_COURSE.materializer_mode
    assert metadata["x_cup_seed"] == 1234
    assert metadata["x_cup_course_hash"] == "abcd1234"
    assert metadata["x_cup_slot"] == 0
    assert metadata["x_cup_generation"] == 2


def test_generated_slot_state_restores_rotated_x_cup_entries_for_managed_continue(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    run_dir = tmp_path / "runs" / "x-cup-continue_0001"
    run_dir.mkdir(parents=True)
    run_paths = continue_run_paths(run_dir)
    ensure_run_dirs(run_paths)

    saved_state_path = run_paths.baselines_dir / "x_cup_rotated.state"
    saved_state_path.write_bytes(b"rotated")
    saved_state_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": BASELINE_MATERIALIZER_SETTINGS.schema_version,
                "cache_kind": X_CUP_COURSE.baseline_cache_kind,
                "cache_key": "rotated-cache",
                "materializer_mode": X_CUP_COURSE.materializer_mode,
                "materialized_state_sha256": "rotated-sha",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    saved_entry = TrackSamplingEntryConfig(
        id="x_cup_rotated",
        course_id="x_cup_rotated",
        runtime_course_key=generated_x_cup_slot_key(0),
        course_name="X Cup rotated",
        course_index=X_CUP_COURSE.course_index,
        mode=X_CUP_COURSE.race_mode,
        gp_difficulty="novice",
        vehicle="blue_falcon",
        engine_setting_raw_value=50,
        baseline_state_path=saved_state_path,
        generated_course_kind=X_CUP_COURSE.generated_kind,
        generated_course_seed=99,
        generated_course_hash="rotated",
        generated_course_slot=0,
        generated_course_generation=3,
        log_per_course=False,
    )

    def fake_ensure_x_cup_baseline(
        *,
        label: str,
        seed: int,
        course_hash: str,
        gp_difficulty: object,
        vehicle_id: str,
        camera_setting: str | None,
        cache_root: Path,
        context: object,
        emulator_type: object,
    ) -> tuple[Path, XCupMaterializedCourse]:
        del label, gp_difficulty, vehicle_id, camera_setting, cache_root, context, emulator_type
        assert seed == 99
        assert course_hash == "rotated"
        return saved_state_path, XCupMaterializedCourse(
            segment_count=212,
            course_length=91_234.5,
        )

    monkeypatch.setattr(
        baseline_materialization,
        "ensure_x_cup_baseline",
        fake_ensure_x_cup_baseline,
    )

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(saved_entry,),
                x_cup_rotation=XCupRotationConfig(enabled=True),
            ),
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="x-cup-continue"),
    )
    slots = (
        GeneratedXCupSlot(
            course_key=generated_x_cup_slot_key(0),
            slot=0,
            generation=3,
            course_id="x_cup_rotated",
            course_name="X Cup rotated",
            course_hash="rotated",
            course_seed=99,
            segment_count=None,
            course_length=None,
        ),
    )

    projected_entry = TrackSamplingEntryConfig(
        id="x_cup_initial",
        course_id="x_cup_initial",
        runtime_course_key=generated_x_cup_slot_key(0),
        course_name="X Cup initial",
        course_index=X_CUP_COURSE.course_index,
        mode=X_CUP_COURSE.race_mode,
        gp_difficulty="novice",
        vehicle="blue_falcon",
        engine_setting_raw_value=50,
        generated_course_kind=X_CUP_COURSE.generated_kind,
        generated_course_seed=11,
        generated_course_hash="initial",
        generated_course_slot=0,
        generated_course_generation=1,
        log_per_course=False,
    )
    projected_config = config.model_copy(
        update={
            "env": config.env.model_copy(
                update={
                    "track_sampling": config.env.track_sampling.model_copy(
                        update={"entries": (projected_entry,)}
                    )
                }
            )
        }
    )
    restored_config = restore_generated_x_cup_entries_from_slots(
        projected_config,
        slots=slots,
    )

    materialized = materialize_train_run_config(
        restored_config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    entry = materialized.env.track_sampling.entries[0]
    assert entry.id == "x_cup_rotated_gp_race_novice_blue_falcon"
    assert entry.course_id == "x_cup_rotated"
    assert entry.generated_course_generation == 3
    baseline_state_path = _required_baseline_path(entry)
    assert baseline_state_path.read_bytes() == b"rotated"
