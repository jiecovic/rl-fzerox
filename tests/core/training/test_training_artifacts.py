# tests/core/training/test_training_artifacts.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.config.schema import (
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    RewardConfig,
    TrainAppConfig,
    TrainConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.core.training.runs import (
    apply_train_run_to_watch_config,
    build_run_paths,
    build_watch_session_paths,
    ensure_run_dirs,
    ensure_watch_session_dirs,
    load_train_run_config,
    materialize_train_run_config,
    materialize_watch_session_config,
    resolve_latest_model_path,
    resolve_latest_policy_path,
    resolve_model_artifact_path,
    resolve_policy_artifact_path,
    save_train_run_config,
)
from rl_fzerox.core.training.session.artifacts import (
    PolicyArtifactMetadata,
    load_policy_artifact_metadata,
    save_artifacts_atomically,
)


class _FakeSaveable:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def save(self, path: str) -> None:
        Path(path).write_bytes(self._payload)


class _FakeModel:
    def __init__(self) -> None:
        self.policy = _FakeSaveable(b"policy")

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"model")


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
        reward=RewardConfig(milestone_bonus=9.0),
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
        reward=RewardConfig(milestone_bonus=1.0),
        watch=WatchConfig(fps=30.0),
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
    assert merged_watch_config.reward.milestone_bonus == 9.0
    assert merged_watch_config.watch.policy_run_dir == run_paths.run_dir
    assert merged_watch_config.watch.fps == 30.0


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
        watch=WatchConfig(fps=30.0),
    )

    merged_watch_config = apply_train_run_to_watch_config(
        watch_config,
        run_dir=run_paths.run_dir,
        train_config=train_config,
    )

    assert merged_watch_config.emulator.baseline_state_path == baseline_state_path.resolve()


def test_materialize_train_run_config_normalizes_auto_to_maskable_ppo(tmp_path: Path) -> None:
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
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(
            algorithm="auto",
            output_root=tmp_path / "runs",
            run_name="ppo_cnn",
        ),
    )
    run_paths = build_run_paths(
        output_root=train_config.train.output_root,
        run_name=train_config.train.run_name,
    )

    materialized = materialize_train_run_config(train_config, run_paths=run_paths)

    assert materialized.train.algorithm == "maskable_ppo"


def test_materialize_train_run_config_preserves_init_artifact_source(tmp_path: Path) -> None:
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
            init_run_dir=init_run_dir,
            init_artifact="latest",
        ),
    )
    run_paths = build_run_paths(
        output_root=train_config.train.output_root,
        run_name=train_config.train.run_name,
    )

    materialized = materialize_train_run_config(train_config, run_paths=run_paths)

    assert materialized.train.init_run_dir == init_run_dir.resolve()
    assert materialized.train.init_artifact == "latest"


def test_resolve_latest_model_path_prefers_latest_over_best_and_final(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    run_paths.final_model_path.write_bytes(b"final")
    run_paths.best_model_path.write_bytes(b"best")
    run_paths.latest_model_path.write_bytes(b"latest")

    resolved_model_path = resolve_latest_model_path(run_paths.run_dir)

    assert resolved_model_path == run_paths.latest_model_path


def test_resolve_latest_policy_path_prefers_latest_over_best_and_final(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    run_paths.final_policy_path.write_bytes(b"final-policy")
    run_paths.best_policy_path.write_bytes(b"best-policy")
    run_paths.latest_policy_path.write_bytes(b"latest-policy")

    resolved_policy_path = resolve_latest_policy_path(run_paths.run_dir)

    assert resolved_policy_path == run_paths.latest_policy_path


def test_save_artifacts_atomically_persists_policy_stage_metadata(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)

    save_artifacts_atomically(
        model=_FakeModel(),
        model_path=run_paths.latest_model_path,
        policy_path=run_paths.latest_policy_path,
        policy_metadata=PolicyArtifactMetadata(
            curriculum_stage_index=1,
            curriculum_stage_name="drift_enabled",
        ),
    )

    metadata = load_policy_artifact_metadata(run_paths.latest_policy_path)

    assert metadata == PolicyArtifactMetadata(
        curriculum_stage_index=1,
        curriculum_stage_name="drift_enabled",
    )


def test_resolve_best_policy_path_requires_best_artifact(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    run_paths.final_policy_path.write_bytes(b"final-policy")
    run_paths.best_policy_path.write_bytes(b"best-policy")
    run_paths.latest_policy_path.write_bytes(b"latest-policy")

    resolved_policy_path = resolve_policy_artifact_path(
        run_paths.run_dir,
        artifact="best",
    )

    assert resolved_policy_path == run_paths.best_policy_path


def test_resolve_final_model_path_requires_final_artifact(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    run_paths.final_model_path.write_bytes(b"final")
    run_paths.best_model_path.write_bytes(b"best")
    run_paths.latest_model_path.write_bytes(b"latest")

    resolved_model_path = resolve_model_artifact_path(
        run_paths.run_dir,
        artifact="final",
    )

    assert resolved_model_path == run_paths.final_model_path


def test_build_run_paths_allocates_numbered_run_directories(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    first = build_run_paths(output_root=output_root, run_name="ppo_cnn")
    ensure_run_dirs(first)
    second = build_run_paths(output_root=output_root, run_name="ppo_cnn")

    assert first.run_dir.name == "ppo_cnn_0001"
    assert second.run_dir.name == "ppo_cnn_0002"
    assert first.runtime_root == first.run_dir / "runtime"
    assert first.env_runtime_dir(0) == first.run_dir / "runtime" / "env_000"
    assert first.baseline_state_path == first.run_dir / "baseline.state"


def test_materialize_train_run_config_copies_baseline_into_run_dir(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    source_baseline_path = tmp_path / "shared.state"
    core_path.touch()
    rom_path.touch()
    source_baseline_path.write_bytes(b"baseline")

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            baseline_state_path=source_baseline_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )

    ensure_run_dirs(run_paths)
    materialized = materialize_train_run_config(config, run_paths=run_paths)

    assert materialized.emulator.runtime_dir == run_paths.runtime_root
    assert materialized.emulator.baseline_state_path == run_paths.baseline_state_path
    assert run_paths.baseline_state_path.read_bytes() == b"baseline"


def test_build_watch_session_paths_uses_run_local_watch_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    baseline_state_path = tmp_path / "baseline.state"

    session_paths = build_watch_session_paths(
        run_dir=run_dir,
        runtime_dir=tmp_path / "runtime",
        baseline_state_path=baseline_state_path,
        session_name="session-001",
    )

    assert session_paths.session_dir == run_dir / "watch" / "session-001"
    assert session_paths.runtime_dir == run_dir / "watch" / "session-001" / "runtime"
    assert session_paths.baseline_state_path == (
        run_dir / "watch" / "session-001" / "baseline.state"
    )


def test_materialize_watch_session_config_isolates_runtime_and_baseline(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    baseline_state_path = tmp_path / "shared.state"
    core_path.touch()
    rom_path.touch()
    baseline_state_path.write_bytes(b"baseline")

    watch_config = WatchAppConfig(
        seed=7,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            runtime_dir=tmp_path / "runtime",
            baseline_state_path=baseline_state_path,
        ),
        watch=WatchConfig(),
    )

    materialized = materialize_watch_session_config(
        watch_config,
        run_dir=tmp_path / "runs" / "ppo_cnn_0001",
        session_name="session-001",
    )

    assert materialized.emulator.runtime_dir == (
        tmp_path / "runs" / "ppo_cnn_0001" / "watch" / "session-001" / "runtime"
    )
    assert materialized.emulator.baseline_state_path == (
        tmp_path / "runs" / "ppo_cnn_0001" / "watch" / "session-001" / "baseline.state"
    )
    assert materialized.emulator.baseline_state_path is not None
    assert materialized.emulator.baseline_state_path.read_bytes() == b"baseline"


def test_ensure_watch_session_dirs_creates_runtime_root(tmp_path: Path) -> None:
    paths = build_watch_session_paths(
        run_dir=None,
        runtime_dir=tmp_path / "runtime",
        baseline_state_path=None,
        session_name="session-001",
    )

    ensure_watch_session_dirs(paths)

    assert paths.session_dir.is_dir()
    assert paths.runtime_dir.is_dir()
