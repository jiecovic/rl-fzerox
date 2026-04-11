# tests/core/training/test_training_runner.py
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rl_fzerox.core.config.schema import (
    ActionConfig,
    ActionMaskConfig,
    CurriculumConfig,
    CurriculumStageConfig,
    CurriculumTriggerConfig,
    EmulatorConfig,
    EnvConfig,
    ObservationConfig,
    PolicyConfig,
    PolicyRecurrentConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.info import MONITOR_INFO_KEYS
from rl_fzerox.core.training.runs import build_run_paths
from rl_fzerox.core.training.session.artifacts import (
    atomic_save_artifact,
    resolve_train_run_config,
    validate_training_baseline_state,
)
from rl_fzerox.core.training.session.callbacks import RolloutInfoAccumulator, info_sequence
from rl_fzerox.core.training.session.curriculum import ActionMaskCurriculumController
from rl_fzerox.core.training.session.model import (
    build_ppo_model,
    build_training_model,
    maybe_preload_training_parameters,
    resolve_effective_training_algorithm,
    resolve_policy_activation_fn,
    training_requires_action_masks,
    validate_training_algorithm_config,
)
from tests.support.fakes import SyntheticBackend


def test_validate_training_baseline_state_requires_existing_file(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    baseline_state_path = tmp_path / "first-race.state"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            baseline_state_path=baseline_state_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    with pytest.raises(RuntimeError, match="Configured training baseline state"):
        validate_training_baseline_state(config)


def test_validate_training_baseline_state_accepts_existing_file(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    baseline_state_path = tmp_path / "first-race.state"
    core_path.touch()
    rom_path.touch()
    baseline_state_path.write_bytes(b"baseline")

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            baseline_state_path=baseline_state_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    validate_training_baseline_state(config)


def test_resolve_train_run_config_sets_run_local_runtime_root(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )

    resolved_config = resolve_train_run_config(config=config, run_paths=run_paths)

    assert resolved_config.emulator.runtime_dir == run_paths.runtime_root


def test_rollout_info_accumulator_summarizes_state_and_episode_metrics() -> None:
    accumulator = RolloutInfoAccumulator()
    infos = [
        {
            "race_distance": 10.0,
            "speed_kph": 100.0,
            "position": 5,
            "lap": 1,
            "race_laps_completed": 0,
            "episode": {
                "position": 2,
                "race_laps_completed": 3,
                "termination_reason": "finished",
                "truncation_reason": None,
            },
        },
        {
            "race_distance": 14.0,
            "speed_kph": 120.0,
            "position": 7,
            "lap": 1,
            "race_laps_completed": 0,
            "episode": {
                "position": 8,
                "race_laps_completed": 1,
                "termination_reason": None,
                "truncation_reason": "wrong_way",
            },
        },
    ]

    accumulator.add_infos(infos)

    assert accumulator.state_metrics["race_distance"].mean() == 12.0
    assert accumulator.state_metrics["speed_kph"].mean() == 110.0
    assert accumulator.state_metrics["race_laps_completed"].mean() == 0.0
    assert accumulator.episode_metrics["position"].mean() == 5.0
    assert accumulator.episode_metrics["race_laps_completed"].mean() == 2.0
    assert accumulator.episode_count == 2
    assert accumulator.termination_counts["finished"] == 1
    assert accumulator.truncation_counts["wrong_way"] == 1


def test_info_sequence_accepts_tuple_infos() -> None:
    infos = ({"race_distance": 10.0}, {"race_distance": 12.0})

    assert info_sequence(infos) == infos
    assert info_sequence([{"race_distance": 10.0}]) == [{"race_distance": 10.0}]
    assert info_sequence(None) is None


def test_resolve_policy_activation_fn_supports_known_names() -> None:
    from torch import nn

    assert resolve_policy_activation_fn("tanh") is nn.Tanh
    assert resolve_policy_activation_fn("relu") is nn.ReLU


def test_resolve_policy_activation_fn_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported policy activation"):
        resolve_policy_activation_fn("gelu")


def test_atomic_save_artifact_replaces_target_without_leaving_tmp(tmp_path: Path) -> None:
    target_path = tmp_path / "latest_policy.zip"

    def _fake_save(path: str) -> None:
        Path(path).write_bytes(b"new-policy")

    atomic_save_artifact(_fake_save, target_path)

    assert target_path.read_bytes() == b"new-policy"
    assert list(tmp_path.glob("*.tmp.zip")) == []


def test_validate_training_algorithm_config_rejects_plain_ppo(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(shoulder=(0,)))),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="ppo"),
    )

    with pytest.raises(RuntimeError, match="Plain PPO training is no longer supported"):
        validate_training_algorithm_config(config)


def test_train_app_config_rejects_recurrent_policy_without_recurrent_algorithm(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    with pytest.raises(
        ValidationError,
        match="policy.recurrent.enabled=true requires "
        "train.algorithm=maskable_recurrent_ppo",
    ):
        TrainAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(),
            policy=PolicyConfig(
                recurrent=PolicyRecurrentConfig(enabled=True),
            ),
            curriculum=CurriculumConfig(),
            train=TrainConfig(algorithm="maskable_ppo"),
        )


def test_train_app_config_rejects_recurrent_algorithm_without_recurrent_policy(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    with pytest.raises(
        ValidationError,
        match="train.algorithm=maskable_recurrent_ppo requires "
        "policy.recurrent.enabled=true",
    ):
        TrainAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(),
            policy=PolicyConfig(),
            curriculum=CurriculumConfig(),
            train=TrainConfig(algorithm="maskable_recurrent_ppo"),
        )


def test_curriculum_controller_promotes_after_smoothed_finish_threshold() -> None:
    controller = ActionMaskCurriculumController(
        CurriculumConfig(
            enabled=True,
            smoothing_episodes=1,
            min_stage_episodes=1,
            stages=(
                CurriculumStageConfig(
                    name="basic_drive",
                    until=CurriculumTriggerConfig(race_laps_completed_mean_gte=3.0),
                    action_mask=ActionMaskConfig(shoulder=(0,)),
                ),
                CurriculumStageConfig(name="drift_enabled"),
            ),
        )
    )

    promoted_stage = controller.record_episodes(
        [{"race_laps_completed": 3, "milestones_completed": 10}]
    )

    assert promoted_stage == 1
    assert controller.stage_index == 1
    assert controller.stage_name == "drift_enabled"


def test_resolve_effective_training_algorithm_uses_maskable_auto_mode(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="auto"),
    )

    assert training_requires_action_masks(config) is True
    assert (
        resolve_effective_training_algorithm(
            train_config=config.train,
            masking_required=training_requires_action_masks(config),
        )
        == "maskable_ppo"
    )


def test_training_requires_no_action_masks_for_sac(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action=ActionConfig(name="continuous_steer_drive")),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="sac", ent_coef="auto"),
    )

    assert training_requires_action_masks(config) is False
    assert (
        resolve_effective_training_algorithm(
            train_config=config.train,
            masking_required=training_requires_action_masks(config),
        )
        == "sac"
    )


def test_validate_training_algorithm_config_rejects_sac_without_continuous_action(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action=ActionConfig(name="steer_drive")),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="sac", ent_coef="auto"),
    )

    with pytest.raises(RuntimeError, match="continuous_steer_drive"):
        validate_training_algorithm_config(config)


def test_build_ppo_model_can_construct_maskable_ppo() -> None:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(shoulder=(0,)))),
            )
        ]
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(algorithm="auto"),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
            masking_required=True,
        )
    finally:
        env.close()

    assert isinstance(model, MaskablePPO)


def test_build_training_model_can_construct_sac() -> None:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=ActionConfig(name="continuous_steer_drive"),
                    observation=ObservationConfig(mode="image_state"),
                ),
            )
        ]
    )

    try:
        model = build_training_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="sac",
                buffer_size=4,
                learning_starts=0,
                ent_coef="auto",
                device="cpu",
            ),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
            masking_required=False,
        )
    finally:
        env.close()

    assert isinstance(model, SAC)


def test_build_ppo_model_rejects_recurrent_policy_with_feedforward_algorithm() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(shoulder=(0,)))),
            )
        ]
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="Recurrent policy config requires train.algorithm=maskable_recurrent_ppo",
        ):
            build_ppo_model(
                train_env=env,
                train_config=TrainConfig(algorithm="maskable_ppo"),
                policy_config=PolicyConfig(
                    recurrent=PolicyRecurrentConfig(enabled=True),
                ),
                tensorboard_log=None,
                masking_required=True,
            )
    finally:
        env.close()


def test_build_ppo_model_can_construct_maskable_recurrent_ppo() -> None:
    from sb3x import MaskableRecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=ActionConfig(mask=ActionMaskConfig(shoulder=(0,))),
                    observation=ObservationConfig(mode="image_state"),
                ),
            )
        ]
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(algorithm="maskable_recurrent_ppo"),
            policy_config=PolicyConfig(
                recurrent=PolicyRecurrentConfig(
                    enabled=True,
                    hidden_size=512,
                    n_lstm_layers=1,
                )
            ),
            tensorboard_log=None,
            masking_required=True,
        )
    finally:
        env.close()

    assert isinstance(model, MaskableRecurrentPPO)
    assert model.policy.state_dict()["lstm_actor.weight_ih_l0"].shape[0] == 4 * 512


def test_maybe_preload_training_parameters_loads_requested_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ppo_cnn_0042"
    run_dir.mkdir(parents=True)
    model_path = run_dir / "latest_model.zip"
    model_path.write_bytes(b"checkpoint")
    train_config_path = run_dir / "train_config.yaml"
    train_config_path.write_text(
        "\n".join(
            [
                "seed: 7",
                "emulator:",
                f"  core_path: {tmp_path / 'core.so'}",
                f"  rom_path: {tmp_path / 'rom.n64'}",
                "env: {}",
                "reward: {}",
                "policy: {}",
                "curriculum: {}",
                "train:",
                "  algorithm: maskable_ppo",
                "  total_timesteps: 1000",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "core.so").touch()
    (tmp_path / "rom.n64").touch()

    class _FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"
            self.calls: list[tuple[str, bool, str]] = []

        def set_parameters(
            self,
            load_path_or_dict: str,
            *,
            exact_match: bool,
            device: str,
        ) -> None:
            self.calls.append((load_path_or_dict, exact_match, device))

    model = _FakeModel()

    maybe_preload_training_parameters(
        model=model,
        train_config=TrainConfig(
            init_run_dir=run_dir,
            init_artifact="latest",
        ),
    )

    assert model.calls == [(str(model_path.resolve()), True, "cpu")]


def test_maybe_preload_training_parameters_rejects_algorithm_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ppo_cnn_0042"
    run_dir.mkdir(parents=True)
    (run_dir / "latest_model.zip").write_bytes(b"checkpoint")
    (tmp_path / "core.so").touch()
    (tmp_path / "rom.n64").touch()
    (run_dir / "train_config.yaml").write_text(
        "\n".join(
            [
                "seed: 7",
                "emulator:",
                f"  core_path: {tmp_path / 'core.so'}",
                f"  rom_path: {tmp_path / 'rom.n64'}",
                "env: {}",
                "reward: {}",
                "policy: {}",
                "curriculum: {}",
                "train:",
                "  algorithm: maskable_ppo",
                "  total_timesteps: 1000",
            ]
        ),
        encoding="utf-8",
    )

    class _FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"

        def set_parameters(
            self,
            load_path_or_dict: str,
            *,
            exact_match: bool,
            device: str,
        ) -> None:
            raise AssertionError("set_parameters should not be reached on mismatch")

    with pytest.raises(
        RuntimeError,
        match="Warm-start checkpoint algorithm mismatch",
    ):
        maybe_preload_training_parameters(
            model=_FakeModel(),
            train_config=TrainConfig(
                algorithm="maskable_recurrent_ppo",
                init_run_dir=run_dir,
                init_artifact="latest",
            ),
        )


def test_monitor_info_keys_include_milestones_completed_for_curriculum() -> None:
    assert "milestones_completed" in MONITOR_INFO_KEYS
