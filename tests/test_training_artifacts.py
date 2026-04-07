# tests/test_training_artifacts.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.config.schema import (
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
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
    resolve_latest_model_path,
    resolve_latest_policy_path,
    save_train_run_config,
)


def test_train_run_config_round_trip_and_watch_inheritance(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    runtime_dir = tmp_path / "runtime"
    core_path.touch()
    rom_path.touch()
    runtime_dir.mkdir()

    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            runtime_dir=runtime_dir,
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

    save_train_run_config(config=train_config, run_dir=run_paths.run_dir)
    loaded_train_config = load_train_run_config(run_paths.run_dir)

    watch_config = WatchAppConfig(
        seed=999,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action_repeat=1),
        watch=WatchConfig(fps=30.0),
    )
    merged_watch_config = apply_train_run_to_watch_config(
        watch_config,
        run_dir=run_paths.run_dir,
        train_config=loaded_train_config,
    )

    assert loaded_train_config.train.output_root == train_config.train.output_root.resolve()
    assert merged_watch_config.seed == 123
    assert merged_watch_config.emulator.runtime_dir == runtime_dir.resolve()
    assert merged_watch_config.env.action_repeat == 3
    assert merged_watch_config.watch.policy_run_dir == run_paths.run_dir
    assert merged_watch_config.watch.fps == 30.0


def test_resolve_latest_model_path_prefers_checkpoint_over_final_model(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    final_model_path = run_paths.final_model_path
    final_model_path.write_bytes(b"final")
    older_checkpoint = run_paths.checkpoints_dir / "ppo_000100.zip"
    newer_checkpoint = run_paths.checkpoints_dir / "ppo_000200.zip"
    older_checkpoint.write_bytes(b"older")
    newer_checkpoint.write_bytes(b"newer")

    resolved_model_path = resolve_latest_model_path(run_paths.run_dir)

    assert resolved_model_path == newer_checkpoint


def test_resolve_latest_policy_path_prefers_checkpoint_over_final_policy(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    run_paths.final_policy_path.write_bytes(b"final-policy")
    older_policy = run_paths.policy_checkpoints_dir / "ppo_policy_000000000100.zip"
    newer_policy = run_paths.policy_checkpoints_dir / "ppo_policy_000000000200.zip"
    older_policy.write_bytes(b"older-policy")
    newer_policy.write_bytes(b"newer-policy")

    resolved_policy_path = resolve_latest_policy_path(run_paths.run_dir)

    assert resolved_policy_path == newer_policy


def test_build_run_paths_allocates_numbered_run_directories(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    first = build_run_paths(output_root=output_root, run_name="ppo_cnn")
    ensure_run_dirs(first)
    second = build_run_paths(output_root=output_root, run_name="ppo_cnn")

    assert first.run_dir.name == "ppo_cnn_0001"
    assert second.run_dir.name == "ppo_cnn_0002"
