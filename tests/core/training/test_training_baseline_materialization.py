# tests/core/training/test_training_baseline_materialization.py
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from rl_fzerox.core.runtime_spec.schema import (
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    TrackConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training.runs import (
    build_run_paths,
    continue_run_paths,
    ensure_run_dirs,
    materialize_train_run_config,
)
from rl_fzerox.core.training.runs.race_start import RaceStartVariant
from tests.core.training.training_artifacts_support import (
    _patch_fake_boot_materializer,
    _required_baseline_path,
)


def test_materialize_train_run_config_reuses_baseline_materializer_cache(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    cache_root = tmp_path / "cache"
    core_path.touch()
    rom_path.touch()
    capture = _patch_fake_boot_materializer(monkeypatch)

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(camera_setting="close_behind"),
        track=TrackConfig(
            id="mute_city",
            course_index=0,
            mode="time_attack",
            vehicle="blue_falcon",
            engine_setting_raw_value=64,
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    first_run = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(first_run)
    first_materialized = materialize_train_run_config(
        config,
        run_paths=first_run,
        baseline_cache_root=cache_root,
    )
    second_run = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(second_run)
    second_materialized = materialize_train_run_config(
        config,
        run_paths=second_run,
        baseline_cache_root=cache_root,
    )

    assert first_materialized.emulator.baseline_state_path is not None
    assert second_materialized.emulator.baseline_state_path is not None
    first_metadata = json.loads(
        first_materialized.emulator.baseline_state_path.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    second_metadata = json.loads(
        second_materialized.emulator.baseline_state_path.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    assert first_metadata["cache_key"] == second_metadata["cache_key"]
    assert first_metadata["materializer_mode"] == "course_vehicle_seed_time_attack"
    assert capture.generic_modes == ["time_attack"]
    assert capture.baseline_state_paths[0] is None
    assert capture.baseline_state_paths[1] is not None
    assert len(capture.variants) == 1
    assert len(list(cache_root.rglob("*.state"))) == 2


def test_materialize_train_run_config_keys_cache_by_runtime_fingerprints(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    cache_root = tmp_path / "cache"
    core_path.write_bytes(b"core-v1")
    rom_path.write_bytes(b"rom-v1")
    capture = _patch_fake_boot_materializer(monkeypatch)

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(camera_setting="close_behind"),
        track=TrackConfig(
            id="mute_city",
            course_index=0,
            mode="time_attack",
            vehicle="blue_falcon",
            engine_setting_raw_value=64,
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    first_run = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(first_run)
    first_materialized = materialize_train_run_config(
        config,
        run_paths=first_run,
        baseline_cache_root=cache_root,
    )
    core_path.write_bytes(b"core-v2")
    rom_path.write_bytes(b"rom-v2")
    second_run = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(second_run)
    second_materialized = materialize_train_run_config(
        config,
        run_paths=second_run,
        baseline_cache_root=cache_root,
    )

    assert first_materialized.emulator.baseline_state_path is not None
    assert second_materialized.emulator.baseline_state_path is not None
    first_metadata = json.loads(
        first_materialized.emulator.baseline_state_path.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    second_metadata = json.loads(
        second_materialized.emulator.baseline_state_path.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    assert first_metadata["cache_key"] != second_metadata["cache_key"]
    assert first_metadata["core_sha256"] != second_metadata["core_sha256"]
    assert first_metadata["rom_sha256"] != second_metadata["rom_sha256"]
    assert capture.generic_modes == ["time_attack", "time_attack"]
    assert len(list(cache_root.rglob("*.state"))) == 4


def test_materialize_train_run_config_regenerates_stale_run_local_baseline_for_in_place_continue(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    capture = _patch_fake_boot_materializer(monkeypatch)

    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    run_dir.mkdir(parents=True)
    run_paths = continue_run_paths(run_dir)
    ensure_run_dirs(run_paths)

    baseline_path = run_paths.baselines_dir / "mute_city__5775c35f3d88.state"
    baseline_path.write_bytes(b"baseline")
    baseline_path.with_suffix(".json").write_text(
        json.dumps({"cache_key": "5775c35f3d88f00dbabe"}, indent=2) + "\n",
        encoding="utf-8",
    )

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            baseline_state_path=baseline_path,
        ),
        env=EnvConfig(camera_setting="close_behind"),
        track=TrackConfig(
            id="mute_city",
            baseline_state_path=baseline_path,
            course_index=0,
            mode="time_attack",
            vehicle="blue_falcon",
            engine_setting_raw_value=64,
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    materialized = materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    assert materialized.emulator.baseline_state_path != baseline_path
    assert materialized.track.baseline_state_path == materialized.emulator.baseline_state_path
    assert capture.variants == [
        RaceStartVariant(
            course_index=0,
            mode="time_attack",
            character_index=0,
            machine_select_slot=0,
            engine_setting_raw_value=64,
            race_intro_target_timer=None,
        )
    ]


def test_materialize_train_run_config_rewrites_track_sampling_baselines(
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
                        id="mute_city",
                        course_id="mute_city",
                        course_index=0,
                        mode="time_attack",
                        vehicle="blue_falcon",
                        engine_setting_raw_value=50,
                    ),
                    TrackSamplingEntryConfig(
                        id="silence",
                        course_id="silence",
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

    materialized = materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    entries = materialized.env.track_sampling.entries
    first_baseline_path = _required_baseline_path(entries[0])
    second_baseline_path = _required_baseline_path(entries[1])
    assert first_baseline_path.parent == run_paths.baselines_dir
    assert second_baseline_path.parent == run_paths.baselines_dir
    assert first_baseline_path.read_bytes() == b"generated"
    assert second_baseline_path.read_bytes() == b"generated"
    assert first_baseline_path.with_suffix(".json").is_file()
    assert second_baseline_path.with_suffix(".json").is_file()
    assert capture.generic_modes == ["time_attack"]
    assert [variant.course_index for variant in capture.variants] == [0, 1]


def test_materialize_track_sampling_expands_gp_baseline_variants(
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
                baseline_variant_count=3,
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_city_gp",
                        course_id="mute_city",
                        runtime_course_key="mute_city",
                        course_index=0,
                        mode="gp_race",
                        gp_difficulty="novice",
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

    materialized = materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    entries = materialized.env.track_sampling.entries
    assert [entry.id for entry in entries] == [
        "mute_city_gp",
        "mute_city_gp__variant_2",
        "mute_city_gp__variant_3",
    ]
    assert [entry.weight for entry in entries] == pytest.approx([1 / 3] * 3)
    assert [entry.baseline_group_id for entry in entries] == ["mute_city_gp"] * 3
    assert [entry.baseline_group_weight for entry in entries] == [1.0] * 3
    assert [entry.baseline_variant_index for entry in entries] == [0, 1, 2]
    assert [entry.baseline_variant_count for entry in entries] == [3, 3, 3]
    assert entries[0].baseline_variant_seed is None
    assert entries[1].baseline_variant_seed is not None
    assert entries[2].baseline_variant_seed is not None
    assert entries[1].baseline_variant_seed != entries[2].baseline_variant_seed

    baseline_paths = {_required_baseline_path(entry) for entry in entries}
    assert len(baseline_paths) == 3
    assert all(path.parent == run_paths.baselines_dir for path in baseline_paths)
    assert [variant.rng_seed for variant in capture.variants] == [
        None,
        entries[1].baseline_variant_seed,
        entries[2].baseline_variant_seed,
    ]


def test_materialize_track_sampling_reuses_gp_baseline_variants_across_run_seeds(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    _patch_fake_boot_materializer(monkeypatch)

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="balanced",
                baseline_variant_count=3,
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_city_gp",
                        course_id="mute_city",
                        runtime_course_key="mute_city",
                        course_index=0,
                        mode="gp_race",
                        gp_difficulty="novice",
                        vehicle="blue_falcon",
                        engine_setting_raw_value=50,
                    ),
                ),
            ),
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    cache_root = tmp_path / "cache"
    first_run = build_run_paths(output_root=config.train.output_root, run_name="first")
    second_run = build_run_paths(output_root=config.train.output_root, run_name="second")
    ensure_run_dirs(first_run)
    ensure_run_dirs(second_run)

    first_materialized = materialize_train_run_config(
        config,
        run_paths=first_run,
        baseline_cache_root=cache_root,
    )
    second_materialized = materialize_train_run_config(
        config.model_copy(update={"seed": 999}),
        run_paths=second_run,
        baseline_cache_root=cache_root,
    )

    first_entries = first_materialized.env.track_sampling.entries
    second_entries = second_materialized.env.track_sampling.entries
    assert [entry.baseline_variant_seed for entry in first_entries] == [
        entry.baseline_variant_seed for entry in second_entries
    ]
    assert [_required_baseline_path(entry).name for entry in first_entries] == [
        _required_baseline_path(entry).name for entry in second_entries
    ]


def test_materialize_track_sampling_does_not_expand_time_attack_baselines(
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
                baseline_variant_count=4,
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_city_time_attack",
                        course_id="mute_city",
                        runtime_course_key="mute_city",
                        course_index=0,
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

    materialized = materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    (entry,) = materialized.env.track_sampling.entries
    assert entry.id == "mute_city_time_attack"
    assert entry.baseline_variant_index is None
    assert [variant.rng_seed for variant in capture.variants] == [None]
