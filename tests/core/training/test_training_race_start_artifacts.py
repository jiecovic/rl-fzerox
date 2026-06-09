# tests/core/training/test_training_race_start_artifacts.py
from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from rl_fzerox.core.runtime_spec.schema import (
    CurriculumConfig,
    CurriculumStageConfig,
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training.runs import (
    build_run_paths,
    ensure_run_dirs,
    materialize_train_run_config,
    reserve_run_paths,
)
from rl_fzerox.core.training.runs.baseline_materializer.requests import request_from_track_entry
from rl_fzerox.core.training.runs.race_start import RaceStartVariant
from rl_fzerox.core.training.session.artifacts import (
    cleanup_failed_run,
)
from tests.core.training.training_artifacts_support import (
    _patch_fake_boot_materializer,
    _required_baseline_path,
)


def test_materialize_train_run_config_reports_track_sampling_progress(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    _patch_fake_boot_materializer(monkeypatch)
    startup_messages: list[tuple[str, str]] = []

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_city",
                        course_id="mute_city",
                        course_name="Mute City",
                        course_index=0,
                        mode="time_attack",
                        vehicle="blue_falcon",
                        engine_setting_raw_value=50,
                    ),
                    TrackSamplingEntryConfig(
                        id="silence",
                        course_id="silence",
                        course_name="Silence",
                        course_index=1,
                        mode="time_attack",
                        vehicle="blue_falcon",
                        engine_setting_raw_value=50,
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

    materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
        startup_reporter=lambda kind, message: startup_messages.append((kind, message)),
    )

    materialize_messages = [
        message for kind, message in startup_messages if kind == "startup_materialize"
    ]
    assert "Materializing track sampling baselines for 2 entries" in materialize_messages
    assert (
        "Materializing track sampling baselines: 0/2 complete; next Mute City"
        in materialize_messages
    )
    assert (
        "Materializing track sampling baselines: 1/2 complete; next Silence" in materialize_messages
    )


def test_materialize_train_run_config_generates_race_start_engine_variant(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    capture = _patch_fake_boot_materializer(monkeypatch)

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            race_intro_target_timer=38,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="silence",
                        course_index=1,
                        mode="time_attack",
                        vehicle="blue_falcon",
                        engine_setting_raw_value=100,
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
    assert baseline_path.read_bytes() == b"generated"
    metadata = json.loads(baseline_path.with_suffix(".json").read_text())
    assert metadata["materializer_mode"] == "course_vehicle_seed_time_attack"
    assert capture.generic_modes == ["time_attack"]
    assert len(capture.variants) == 1
    course_vehicle_variant = capture.variants[0]
    assert course_vehicle_variant.course_index == 1
    assert course_vehicle_variant.mode == "time_attack"
    assert course_vehicle_variant.character_index == 0
    assert course_vehicle_variant.engine_setting_raw_value == 50
    assert entry.source_vehicle == "blue_falcon"
    assert entry.source_engine_setting_raw_value == 50


def test_materialize_train_run_config_reuses_target_variant_cache_without_source(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    cache_root = tmp_path / "cache"
    core_path.touch()
    rom_path.touch()
    capture = _patch_fake_boot_materializer(monkeypatch, payload=b"generated-target")

    def build_config() -> TrainAppConfig:
        return TrainAppConfig(
            seed=123,
            emulator=EmulatorConfig(
                core_path=core_path,
                rom_path=rom_path,
                renderer="angrylion",
            ),
            env=EnvConfig(
                race_intro_target_timer=38,
                track_sampling=TrackSamplingConfig(
                    enabled=True,
                    sampling_mode="balanced",
                    entries=(
                        TrackSamplingEntryConfig(
                            id="silence",
                            course_id="silence",
                            course_index=1,
                            mode="time_attack",
                            vehicle="blue_falcon",
                            engine_setting_raw_value=100,
                        ),
                    ),
                ),
            ),
            policy=PolicyConfig(),
            train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
        )

    first_run = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(first_run)
    first_materialized = materialize_train_run_config(
        build_config(),
        run_paths=first_run,
        baseline_cache_root=cache_root,
    )
    second_run = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(second_run)
    second_materialized = materialize_train_run_config(
        build_config(),
        run_paths=second_run,
        baseline_cache_root=cache_root,
    )

    first_entry = first_materialized.env.track_sampling.entries[0]
    second_entry = second_materialized.env.track_sampling.entries[0]
    first_baseline_path = _required_baseline_path(first_entry)
    second_baseline_path = _required_baseline_path(second_entry)
    assert first_baseline_path.read_bytes() == b"generated-target"
    assert second_baseline_path.read_bytes() == b"generated-target"
    assert capture.generic_modes == ["time_attack"]
    assert capture.variants == [
        RaceStartVariant(
            course_index=1,
            mode="time_attack",
            character_index=0,
            machine_select_slot=0,
            engine_setting_raw_value=50,
            race_intro_target_timer=None,
        )
    ]
    assert first_baseline_path.with_suffix(".json").is_file()
    assert second_baseline_path.with_suffix(".json").is_file()


def test_materialize_train_run_config_generates_vehicle_variant(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    capture = _patch_fake_boot_materializer(monkeypatch)

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="silence",
                        course_index=1,
                        mode="time_attack",
                        vehicle="white_cat",
                        engine_setting_raw_value=70,
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
    assert entry.baseline_state_path is not None
    assert entry.baseline_state_path.read_bytes() == b"generated"
    assert entry.source_vehicle == "white_cat"
    assert entry.source_engine_setting_raw_value == 50
    assert [variant.character_index for variant in capture.variants] == [4]
    assert [variant.machine_select_slot for variant in capture.variants] == [4]
    assert [variant.engine_setting_raw_value for variant in capture.variants] == [50]


def test_materialize_train_run_config_rewrites_curriculum_track_sampling(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    _patch_fake_boot_materializer(monkeypatch, payload=b"stage")

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        curriculum=CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(
                    name="stage",
                    track_sampling=TrackSamplingConfig(
                        enabled=True,
                        entries=(
                            TrackSamplingEntryConfig(
                                id="stage_track",
                                course_index=0,
                                mode="time_attack",
                                vehicle="blue_falcon",
                                engine_setting_raw_value=50,
                            ),
                        ),
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

    stage = materialized.curriculum.stages[0]
    assert stage.track_sampling is not None
    entry = stage.track_sampling.entries[0]
    baseline_path = _required_baseline_path(entry)
    assert baseline_path.parent == run_paths.baselines_dir
    assert baseline_path.read_bytes() == b"stage"


def test_cleanup_failed_run_preserves_explicit_run_dir_when_requested(tmp_path: Path) -> None:
    run_paths = reserve_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")

    cleanup_failed_run(run_paths, model=None, preserve_run_dir=True)

    assert run_paths.run_dir.is_dir()


def test_track_sampling_baseline_label_omits_engine_range_metadata() -> None:
    entry = TrackSamplingEntryConfig(
        id="silence",
        course_id="silence",
        course_name="Silence",
        course_index=0,
        mode="time_attack",
        vehicle="fire_stingray",
        vehicle_name="Fire Stingray",
    )

    request = request_from_track_entry(entry, camera_setting="close_behind")

    assert request.label == "silence_time_attack_fire_stingray"
