# tests/core/training/test_training_run_config.py
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from pytest import MonkeyPatch

from rl_fzerox.core.runtime_spec.schema import (
    ActionConfig,
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    RewardConfig,
    TrackConfig,
    TrainAppConfig,
    TrainConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.core.training.runs import (
    apply_train_run_to_watch_config,
    build_run_paths,
    ensure_run_dirs,
    load_train_run_config,
    materialize_train_run_config,
    save_train_run_config,
)
from tests.core.training.training_artifacts_support import (
    _patch_fake_boot_materializer,
)


def test_train_run_config_round_trip_and_watch_inheritance(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    train_runtime_dir = tmp_path / "runtime"
    watch_runtime_dir = tmp_path / "watch-runtime"
    core_path.touch()
    rom_path.touch()
    train_runtime_dir.mkdir()
    watch_runtime_dir.mkdir()

    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            runtime_dir=train_runtime_dir,
        ),
        env=EnvConfig(action_repeat=3),
        reward=RewardConfig(progress_bucket_reward=9.0),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=train_config.train.output_root,
        run_name=train_config.train.run_name,
    )
    ensure_run_dirs(run_paths)

    save_train_run_config(config=train_config, run_dir=run_paths.run_dir)
    loaded_train_config = load_train_run_config(run_paths.run_dir)

    watch_config = WatchAppConfig(
        seed=999,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            runtime_dir=watch_runtime_dir,
        ),
        env=EnvConfig(action_repeat=1),
        reward=RewardConfig(progress_bucket_reward=1.0),
        watch=WatchConfig(control_fps=30.0, render_fps=30.0),
    )
    merged_watch_config = apply_train_run_to_watch_config(
        watch_config,
        run_dir=run_paths.run_dir,
        train_config=loaded_train_config,
    )

    assert loaded_train_config.train.output_root == train_config.train.output_root.resolve()
    assert merged_watch_config.seed == 123
    assert loaded_train_config.emulator.runtime_dir == train_runtime_dir.resolve()
    assert merged_watch_config.emulator.runtime_dir == watch_runtime_dir.resolve()
    assert merged_watch_config.env.action_repeat == 3
    assert merged_watch_config.reward.progress_bucket_reward == 9.0
    assert merged_watch_config.policy == loaded_train_config.policy
    assert merged_watch_config.train == loaded_train_config.train
    assert merged_watch_config.watch.policy_run_dir == run_paths.run_dir
    assert merged_watch_config.watch.control_fps == 30.0
    assert merged_watch_config.watch.render_fps == 30.0


def test_load_train_run_config_ignores_removed_adaptive_step_balance_fields(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="legacy_tracks"),
    )
    run_paths = build_run_paths(
        output_root=train_config.train.output_root,
        run_name=train_config.train.run_name,
    )
    ensure_run_dirs(run_paths)
    config_path = save_train_run_config(config=train_config, run_dir=run_paths.run_dir)
    saved = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(saved, dict)
    env_data = saved.setdefault("env", {})
    assert isinstance(env_data, dict)
    track_sampling = env_data.setdefault("track_sampling", {})
    assert isinstance(track_sampling, dict)
    track_sampling.update(
        {
            "adaptive_step_balance_completion_weight": 1.0,
            "adaptive_step_balance_confidence_scale": 4.0,
            "adaptive_step_balance_min_confidence_episodes": 24,
            "adaptive_step_balance_target_completion": 0.8,
            "sampling_mode": "adaptive_step_balanced",
        }
    )
    OmegaConf.save(config=OmegaConf.create(saved), f=str(config_path))

    loaded_config = load_train_run_config(run_paths.run_dir)
    serialized_track_sampling = loaded_config.env.track_sampling.model_dump(mode="json")

    assert loaded_config.env.track_sampling.sampling_mode == "step_balanced"
    assert "adaptive_step_balance_completion_weight" not in serialized_track_sampling
    assert "adaptive_step_balance_confidence_scale" not in serialized_track_sampling
    assert "adaptive_step_balance_min_confidence_episodes" not in serialized_track_sampling
    assert "adaptive_step_balance_target_completion" not in serialized_track_sampling


def test_load_train_run_config_maps_removed_track_sampling_mode_names(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    for old_mode, expected_mode in {
        "adaptive_step_balanced": "step_balanced",
        "balanced": "equal",
        "random": "equal",
    }.items():
        train_config = TrainAppConfig(
            seed=123,
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            train=TrainConfig(output_root=tmp_path / "runs", run_name=f"legacy_{old_mode}"),
        )
        run_paths = build_run_paths(
            output_root=train_config.train.output_root,
            run_name=train_config.train.run_name,
        )
        ensure_run_dirs(run_paths)
        config_path = save_train_run_config(config=train_config, run_dir=run_paths.run_dir)
        saved = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        assert isinstance(saved, dict)
        env_data = saved.setdefault("env", {})
        assert isinstance(env_data, dict)
        track_sampling = env_data.setdefault("track_sampling", {})
        assert isinstance(track_sampling, dict)
        track_sampling["sampling_mode"] = old_mode
        OmegaConf.save(config=OmegaConf.create(saved), f=str(config_path))

        loaded_config = load_train_run_config(run_paths.run_dir)

        assert loaded_config.env.track_sampling.sampling_mode == expected_mode


def test_save_train_run_config_persists_configured_action_layout(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(
            action=ActionConfig.model_validate(
                {
                    "layout_continuous_axes": ["steer"],
                    "layout_discrete_axes": ["gas", "boost", "lean"],
                    "mask": {
                        "gas": [0, 1],
                        "boost": [0],
                        "lean": [0, 1, 2],
                    },
                    "lean_mode": "release_cooldown",
                }
            ),
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="exp_v3_cnn"),
    )
    run_paths = build_run_paths(
        output_root=train_config.train.output_root,
        run_name=train_config.train.run_name,
    )
    ensure_run_dirs(run_paths)

    config_path = save_train_run_config(config=train_config, run_dir=run_paths.run_dir)

    saved = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(saved, dict)
    env_data = saved["env"]
    assert isinstance(env_data, dict)
    action_data = env_data["action"]
    assert isinstance(action_data, dict)
    assert "name" not in action_data
    assert action_data["layout_continuous_axes"] == ["steer"]
    assert action_data["layout_discrete_axes"] == ["gas", "boost", "lean"]
    assert action_data["mask"] == {
        "gas": [0, 1],
        "boost": [0],
        "lean": [0, 1, 2],
    }
    assert "configured_mask_overrides" not in action_data
    assert action_data["boost_decision_interval_frames"] == 1
    assert action_data["mask_boost_when_active"] is True
    assert action_data["mask_boost_when_airborne"] is True
    assert action_data["boost_request_lockout_frames"] == 5
    assert action_data["spin_cooldown_frames"] == 120

    loaded_config = load_train_run_config(run_paths.run_dir)
    action_config = loaded_config.env.action.runtime()

    assert action_config.name == "configured_hybrid"
    assert action_config.layout_continuous_axes == ("steer",)
    assert action_config.layout_discrete_axes == ("gas", "boost", "lean")
    assert action_config.boost_decision_interval_frames == 1
    assert action_config.boost_request_lockout_frames == 5
    assert action_config.spin_cooldown_frames == 120
    assert action_config.mask_boost_when_active is True
    assert action_config.mask_boost_when_airborne is True


def test_watch_inheritance_preserves_local_baseline_when_run_snapshot_lacks_it(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    baseline_state_path = tmp_path / "first-race.state"
    core_path.touch()
    rom_path.touch()
    baseline_state_path.write_bytes(b"baseline")

    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(action_repeat=3),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=train_config.train.output_root,
        run_name=train_config.train.run_name,
    )
    ensure_run_dirs(run_paths)

    watch_config = WatchAppConfig(
        seed=999,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            baseline_state_path=baseline_state_path,
        ),
        env=EnvConfig(action_repeat=1),
        watch=WatchConfig(control_fps=30.0, render_fps=30.0),
    )

    merged_watch_config = apply_train_run_to_watch_config(
        watch_config,
        run_dir=run_paths.run_dir,
        train_config=train_config,
    )

    assert merged_watch_config.emulator.baseline_state_path == baseline_state_path.resolve()


def test_watch_inheritance_uses_train_camera_setting(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(action_repeat=3, camera_setting="regular"),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    watch_config = WatchAppConfig(
        seed=999,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(action_repeat=1, camera_setting="close_behind"),
        watch=WatchConfig(control_fps=30.0, render_fps=30.0),
    )

    merged_watch_config = apply_train_run_to_watch_config(
        watch_config,
        run_dir=tmp_path / "runs" / "ppo_cnn_0001",
        train_config=train_config,
    )

    assert merged_watch_config.env.action_repeat == 3
    assert merged_watch_config.env.camera_setting == "regular"


def test_materialize_train_run_config_preserves_resume_artifact_source(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    init_run_dir = tmp_path / "runs" / "ppo_cnn_0042"
    core_path.touch()
    rom_path.touch()
    init_run_dir.mkdir(parents=True)

    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(
            output_root=tmp_path / "runs",
            run_name="ppo_cnn",
            resume_run_dir=init_run_dir,
            resume_artifact="latest",
        ),
    )
    run_paths = build_run_paths(
        output_root=train_config.train.output_root,
        run_name=train_config.train.run_name,
    )

    materialized = materialize_train_run_config(train_config, run_paths=run_paths)

    assert materialized.train.resume_run_dir == init_run_dir.resolve()
    assert materialized.train.resume_artifact == "latest"


def test_materialize_train_run_config_does_not_copy_init_run_baseline(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    init_baseline_path = tmp_path / "runs" / "source_0001" / "baselines" / "baseline.state"
    core_path.touch()
    rom_path.touch()
    init_baseline_path.parent.mkdir(parents=True)
    init_baseline_path.write_bytes(b"init-run-baseline")
    (init_baseline_path.parent.parent / "train_manifest.yaml").write_text(
        "\n".join(
            [
                "seed: 7",
                "emulator:",
                f"  core_path: {core_path}",
                f"  rom_path: {rom_path}",
                f"  baseline_state_path: {init_baseline_path}",
                "env: {}",
                "reward: {}",
                "policy: {}",
                "train:",
                "  algorithm: maskable_hybrid_action_ppo",
            ]
        ),
        encoding="utf-8",
    )

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(),
        track=TrackConfig(
            id="mute_city",
            course_index=0,
            mode="time_attack",
            vehicle="blue_falcon",
            engine_setting_raw_value=50,
        ),
        policy=PolicyConfig(),
        train=TrainConfig(
            output_root=tmp_path / "runs",
            run_name="target",
            resume_run_dir=init_baseline_path.parent.parent,
            resume_artifact="latest",
        ),
    )
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(run_paths)
    _patch_fake_boot_materializer(monkeypatch, payload=b"current-baseline")

    materialized = materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    assert materialized.emulator.baseline_state_path is not None
    assert materialized.emulator.baseline_state_path.parent == run_paths.baselines_dir
    assert materialized.emulator.baseline_state_path.read_bytes() == b"current-baseline"
    assert materialized.train.resume_run_dir == init_baseline_path.parent.parent.resolve()
