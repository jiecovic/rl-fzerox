# tests/core/training/test_training_algorithms.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.runtime_spec.schema import (
    ActionMaskConfig,
    CurriculumConfig,
    EmulatorConfig,
    EnvConfig,
    ObservationConfig,
    ObservationStateComponentConfig,
    PolicyActionBiasConfig,
    PolicyAuxiliaryStateConfig,
    PolicyAuxiliaryStateLossConfig,
    PolicyConfig,
    PolicyRecurrentConfig,
    TrainActorRegularizationConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training.session.auxiliary_state import (
    maybe_wrap_training_auxiliary_state_observation,
)
from rl_fzerox.core.training.session.model import (
    build_ppo_model,
    resolve_effective_training_algorithm,
    training_requires_action_masks,
    validate_training_algorithm_config,
)
from rl_fzerox.core.training.session.model.action_bias import (
    MODEL_ACTION_BIAS_OFFSETS_ATTR,
    apply_resume_action_bias_delta,
)
from tests.support.action_configs import (
    configured_discrete_action,
    configured_hybrid_action,
)
from tests.support.fakes import SyntheticBackend, vec_env_fns

_VEHICLE_STATE_COMPONENT = (ObservationStateComponentConfig(name="vehicle_state"),)


def _emulator_config(tmp_path: Path) -> EmulatorConfig:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    return EmulatorConfig(core_path=core_path, rom_path=rom_path)


def test_training_requires_action_masks_for_current_supported_algorithms(
    tmp_path: Path,
) -> None:
    emulator = _emulator_config(tmp_path)
    hybrid_env = EnvConfig(
        action=configured_hybrid_action(
            continuous_axes=("steer", "drive"),
            discrete_axes=("boost", "lean"),
        )
    )
    discrete_env = EnvConfig(action=configured_discrete_action("steer", "gas", "boost", "lean"))

    assert (
        training_requires_action_masks(
            TrainAppConfig(
                emulator=emulator,
                env=hybrid_env,
                policy=PolicyConfig(),
                curriculum=CurriculumConfig(),
                train=TrainConfig(algorithm="maskable_hybrid_action_ppo"),
            )
        )
        is True
    )
    assert (
        training_requires_action_masks(
            TrainAppConfig(
                emulator=emulator,
                env=hybrid_env,
                policy=PolicyConfig(),
                curriculum=CurriculumConfig(),
                train=TrainConfig(algorithm="maskable_hybrid_action_ppo"),
            )
        )
        is True
    )
    assert (
        training_requires_action_masks(
            TrainAppConfig(
                emulator=emulator,
                env=hybrid_env,
                policy=PolicyConfig(recurrent=PolicyRecurrentConfig(enabled=True)),
                curriculum=CurriculumConfig(),
                train=TrainConfig(algorithm="maskable_hybrid_recurrent_ppo"),
            )
        )
        is True
    )
    assert (
        training_requires_action_masks(
            TrainAppConfig(
                emulator=emulator,
                env=discrete_env,
                policy=PolicyConfig(),
                curriculum=CurriculumConfig(),
                train=TrainConfig(algorithm="maskable_ppo"),
            )
        )
        is True
    )


def test_resolve_effective_training_algorithm_returns_configured_algorithm(
    tmp_path: Path,
) -> None:
    config = TrainAppConfig(
        emulator=_emulator_config(tmp_path),
        env=EnvConfig(
            action=configured_hybrid_action(
                continuous_axes=("steer", "drive"),
                discrete_axes=("boost", "lean"),
            )
        ),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="maskable_hybrid_action_ppo"),
    )

    assert resolve_effective_training_algorithm(train_config=config.train) == (
        "maskable_hybrid_action_ppo"
    )


def test_validate_training_algorithm_config_rejects_hybrid_ppo_without_hybrid_action(
    tmp_path: Path,
) -> None:
    config = TrainAppConfig(
        emulator=_emulator_config(tmp_path),
        env=EnvConfig(action=configured_discrete_action("steer", "gas", "boost", "lean")),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(algorithm="maskable_hybrid_action_ppo"),
    )

    with pytest.raises(RuntimeError, match="configured_hybrid"):
        validate_training_algorithm_config(config)


def test_build_ppo_model_can_construct_maskable_hybrid_action_ppo() -> None:
    from sb3x import MaskableHybridActionPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=configured_hybrid_action(
                        continuous_axes=("steer", "drive"),
                        discrete_axes=("boost", "lean"),
                    ),
                    observation=ObservationConfig(
                        mode="image_state",
                        state_components=_VEHICLE_STATE_COMPONENT,
                    ),
                ),
            )
        )
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_action_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
        )
    finally:
        env.close()

    assert isinstance(model, MaskableHybridActionPPO)


def test_hybrid_ppo_accepts_group_entropy_and_pitch_actor_loss() -> None:
    from sb3x import MaskableHybridActionPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_config = EnvConfig(
        action=configured_hybrid_action(
            continuous_axes=("steer", "pitch"),
            discrete_axes=("boost", "lean"),
        ),
        observation=ObservationConfig(
            mode="image_state",
            state_components=_VEHICLE_STATE_COMPONENT,
        ),
    )
    policy_config = PolicyConfig()
    train_config = TrainConfig(
        algorithm="maskable_hybrid_action_ppo",
        n_steps=4,
        batch_size=4,
        device="cpu",
        entropy_group_weights={"pitch": 1.0, "steer": 0.0},
        actor_regularization=TrainActorRegularizationConfig(
            grounded_pitch_neutral_loss_weight=0.01,
            pitch_std_cap_loss_weight=0.05,
            grounded_pitch_std_cap=0.5,
        ),
    )
    env = DummyVecEnv(
        vec_env_fns(
            lambda: maybe_wrap_training_auxiliary_state_observation(
                FZeroXEnv(
                    backend=SyntheticBackend(),
                    config=env_config,
                ),
                policy_config=policy_config,
                train_config=train_config,
            )
        )
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=train_config,
            policy_config=policy_config,
            env_config=env_config,
            tensorboard_log=None,
        )
        model.learn(total_timesteps=4)
    finally:
        env.close()

    assert isinstance(model, MaskableHybridActionPPO)
    assert model.entropy_group_weights == {"pitch": 1.0, "steer": 0.0}
    assert not any("auxiliary_state_heads" in key for key in model.policy.state_dict())


def test_hybrid_ppo_accepts_state_scoped_pitch_std_actor_loss() -> None:
    from sb3x import MaskableHybridActionPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_config = EnvConfig(
        action=configured_hybrid_action(
            continuous_axes=("steer", "pitch"),
            discrete_axes=("boost", "lean"),
        ),
        observation=ObservationConfig(
            mode="image_state",
            state_components=_VEHICLE_STATE_COMPONENT,
        ),
    )
    policy_config = PolicyConfig()
    train_config = TrainConfig(
        algorithm="maskable_hybrid_action_ppo",
        n_steps=4,
        batch_size=4,
        device="cpu",
        actor_regularization=TrainActorRegularizationConfig(
            grounded_pitch_neutral_loss_weight=0.01,
            pitch_std_cap_loss_weight=0.05,
            grounded_pitch_std_cap=0.35,
            airborne_pitch_std_cap=0.8,
        ),
    )
    env = DummyVecEnv(
        vec_env_fns(
            lambda: maybe_wrap_training_auxiliary_state_observation(
                FZeroXEnv(
                    backend=SyntheticBackend(),
                    config=env_config,
                ),
                policy_config=policy_config,
                train_config=train_config,
            )
        )
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=train_config,
            policy_config=policy_config,
            env_config=env_config,
            tensorboard_log=None,
        )
        model.learn(total_timesteps=4)
    finally:
        env.close()

    assert isinstance(model, MaskableHybridActionPPO)
    assert model.policy.continuous_log_std_mode == "parameter"


def test_hybrid_ppo_accepts_discrete_pitch_actor_loss() -> None:
    from sb3x import MaskableHybridActionPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_config = EnvConfig(
        action=configured_hybrid_action(
            continuous_axes=("steer",),
            discrete_axes=("boost", "lean", "pitch"),
            pitch_buckets=5,
        ),
        observation=ObservationConfig(
            mode="image_state",
            state_components=_VEHICLE_STATE_COMPONENT,
        ),
    )
    policy_config = PolicyConfig()
    train_config = TrainConfig(
        algorithm="maskable_hybrid_action_ppo",
        n_steps=4,
        batch_size=4,
        device="cpu",
        actor_regularization=TrainActorRegularizationConfig(
            grounded_pitch_neutral_loss_weight=0.01,
            pitch_std_cap_loss_weight=0.05,
            grounded_pitch_std_cap=0.35,
            airborne_pitch_std_cap=0.8,
        ),
    )
    env = DummyVecEnv(
        vec_env_fns(
            lambda: maybe_wrap_training_auxiliary_state_observation(
                FZeroXEnv(
                    backend=SyntheticBackend(),
                    config=env_config,
                ),
                policy_config=policy_config,
                train_config=train_config,
            )
        )
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=train_config,
            policy_config=policy_config,
            env_config=env_config,
            tensorboard_log=None,
        )
        model.learn(total_timesteps=4)
    finally:
        env.close()

    assert isinstance(model, MaskableHybridActionPPO)
    assert model.policy.continuous_log_std_mode == "parameter"


def test_build_ppo_model_rejects_unavailable_explicit_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.builders.th.cuda.is_available",
        lambda: False,
    )
    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=configured_hybrid_action(
                        continuous_axes=("steer", "drive"),
                        discrete_axes=("boost", "lean"),
                    ),
                    observation=ObservationConfig(
                        mode="image_state",
                        state_components=_VEHICLE_STATE_COMPONENT,
                    ),
                ),
            )
        )
    )

    try:
        with pytest.raises(RuntimeError, match="PyTorch cannot access CUDA"):
            build_ppo_model(
                train_env=env,
                train_config=TrainConfig(
                    algorithm="maskable_hybrid_action_ppo",
                    n_steps=4,
                    batch_size=4,
                    device="cuda",
                ),
                policy_config=PolicyConfig(),
                tensorboard_log=None,
            )
    finally:
        env.close()


def test_build_ppo_model_can_construct_maskable_hybrid_recurrent_ppo() -> None:
    from sb3x import MaskableHybridRecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=configured_hybrid_action(
                        continuous_axes=("steer", "drive"),
                        discrete_axes=("boost", "lean"),
                    ),
                    observation=ObservationConfig(
                        mode="image_state",
                        state_components=_VEHICLE_STATE_COMPONENT,
                    ),
                ),
            )
        )
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_recurrent_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(
                recurrent=PolicyRecurrentConfig(
                    enabled=True,
                    hidden_size=512,
                    n_lstm_layers=1,
                )
            ),
            tensorboard_log=None,
        )
    finally:
        env.close()

    assert isinstance(model, MaskableHybridRecurrentPPO)
    assert model.policy.state_dict()["lstm_actor.weight_ih_l0"].shape[0] == 4 * 512


def test_build_ppo_model_rejects_auxiliary_state_for_maskable_ppo() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        vec_env_fns(
            lambda: maybe_wrap_training_auxiliary_state_observation(
                FZeroXEnv(
                    backend=SyntheticBackend(),
                    config=EnvConfig(
                        action=configured_discrete_action("steer", "gas", "boost", "lean"),
                        observation=ObservationConfig(
                            mode="image_state",
                            state_components=_VEHICLE_STATE_COMPONENT,
                        ),
                    ),
                ),
                policy_config=PolicyConfig(
                    auxiliary_state=PolicyAuxiliaryStateConfig(enabled=True)
                ),
            )
        )
    )

    try:
        with pytest.raises(RuntimeError, match="maskable_ppo"):
            build_ppo_model(
                train_env=env,
                train_config=TrainConfig(
                    algorithm="maskable_ppo",
                    n_steps=4,
                    batch_size=4,
                    device="cpu",
                ),
                policy_config=PolicyConfig(
                    auxiliary_state=PolicyAuxiliaryStateConfig(enabled=True)
                ),
                tensorboard_log=None,
            )
    finally:
        env.close()


def test_maskable_hybrid_action_ppo_learns_with_auxiliary_state_loss() -> None:
    from sb3x import MaskableHybridActionPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    policy_config = PolicyConfig(
        auxiliary_state=PolicyAuxiliaryStateConfig(
            enabled=True,
            losses=(
                PolicyAuxiliaryStateLossConfig(
                    name="track_position.edge_ratio",
                    weight=0.25,
                    grounded_only=True,
                ),
            ),
        )
    )
    env = DummyVecEnv(
        vec_env_fns(
            lambda: maybe_wrap_training_auxiliary_state_observation(
                FZeroXEnv(
                    backend=SyntheticBackend(),
                    config=EnvConfig(
                        action=configured_hybrid_action(
                            continuous_axes=("steer", "drive"),
                            discrete_axes=("boost", "lean"),
                        ),
                        observation=ObservationConfig(
                            mode="image_state",
                            state_components=_VEHICLE_STATE_COMPONENT,
                        ),
                    ),
                ),
                policy_config=policy_config,
            )
        )
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_action_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=policy_config,
            tensorboard_log=None,
        )
        # On-policy training metrics are recorded during train() and dumped on the
        # next log cycle, so run two short rollouts here.
        model.learn(total_timesteps=8, log_interval=1)
    finally:
        env.close()

    assert isinstance(model, MaskableHybridActionPPO)
    assert hasattr(model.policy, "evaluate_actions_with_aux")


def test_maskable_hybrid_recurrent_ppo_learns_with_auxiliary_state_loss() -> None:
    from sb3x import MaskableHybridRecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    policy_config = PolicyConfig(
        recurrent=PolicyRecurrentConfig(
            enabled=True,
            hidden_size=64,
            n_lstm_layers=1,
        ),
        auxiliary_state=PolicyAuxiliaryStateConfig(
            enabled=True,
            losses=(
                PolicyAuxiliaryStateLossConfig(
                    name="track_position.edge_ratio",
                    weight=0.25,
                    grounded_only=True,
                ),
            ),
        ),
    )
    env = DummyVecEnv(
        vec_env_fns(
            lambda: maybe_wrap_training_auxiliary_state_observation(
                FZeroXEnv(
                    backend=SyntheticBackend(),
                    config=EnvConfig(
                        action=configured_hybrid_action(
                            continuous_axes=("steer", "drive"),
                            discrete_axes=("boost", "lean"),
                        ),
                        observation=ObservationConfig(
                            mode="image_state",
                            state_components=_VEHICLE_STATE_COMPONENT,
                        ),
                    ),
                ),
                policy_config=policy_config,
            )
        )
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_recurrent_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=policy_config,
            tensorboard_log=None,
        )
        model.learn(total_timesteps=4)
    finally:
        env.close()

    assert isinstance(model, MaskableHybridRecurrentPPO)
    assert hasattr(model.policy, "evaluate_actions_with_aux")


def test_auxiliary_state_losses_are_logged_to_tensorboard(tmp_path: Path) -> None:
    from stable_baselines3.common import logger as sb3_logger
    from stable_baselines3.common.vec_env import DummyVecEnv

    event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator"
    )

    policy_config = PolicyConfig(
        auxiliary_state=PolicyAuxiliaryStateConfig(
            enabled=True,
            losses=(
                PolicyAuxiliaryStateLossConfig(
                    name="track_position.edge_ratio",
                    weight=0.25,
                    grounded_only=True,
                ),
                PolicyAuxiliaryStateLossConfig(
                    name="vehicle_state.airborne",
                    weight=0.1,
                ),
            ),
        )
    )
    env = DummyVecEnv(
        vec_env_fns(
            lambda: maybe_wrap_training_auxiliary_state_observation(
                FZeroXEnv(
                    backend=SyntheticBackend(),
                    config=EnvConfig(
                        action=configured_hybrid_action(
                            continuous_axes=("steer", "drive"),
                            discrete_axes=("boost", "lean"),
                        ),
                        observation=ObservationConfig(
                            mode="image_state",
                            state_components=_VEHICLE_STATE_COMPONENT,
                        ),
                    ),
                ),
                policy_config=policy_config,
            )
        )
    )

    tensorboard_dir = tmp_path / "tensorboard"
    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_action_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=policy_config,
            tensorboard_log=None,
        )
        model.set_logger(sb3_logger.configure(str(tensorboard_dir), ["tensorboard"]))
        # On-policy train metrics are dumped on the next log cycle, so run
        # two short rollouts here.
        model.learn(total_timesteps=8, log_interval=1)
    finally:
        env.close()

    event_dirs = {path.parent for path in tensorboard_dir.rglob("events.out.tfevents.*")}
    assert event_dirs
    scalar_tags: set[str] = set()
    for event_dir in event_dirs:
        accumulator = event_accumulator.EventAccumulator(
            str(event_dir),
            size_guidance={"scalars": 0},
        )
        accumulator.Reload()
        scalar_tags.update(accumulator.Tags().get("scalars", ()))

    assert "train/aux_loss" in scalar_tags
    assert "train_aux/track_position.edge_ratio" in scalar_tags
    assert "train_aux/vehicle_state.airborne" in scalar_tags


def test_build_ppo_model_applies_hybrid_gas_on_logit_bias() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=configured_hybrid_action(
                        continuous_axes=("steer",),
                        discrete_axes=("gas", "air_brake", "boost", "lean", "pitch"),
                    ),
                    observation=ObservationConfig(
                        mode="image_state",
                        state_components=_VEHICLE_STATE_COMPONENT,
                    ),
                ),
            )
        )
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_recurrent_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(
                recurrent=PolicyRecurrentConfig(
                    enabled=True,
                    hidden_size=512,
                    n_lstm_layers=1,
                ),
                action_bias=PolicyActionBiasConfig(gas_on_logit=0.5),
            ),
            tensorboard_log=None,
        )
    finally:
        env.close()

    bias = model.policy.state_dict()["action_net.discrete_net.bias"].detach().cpu()
    assert float(bias[1] - bias[0]) == pytest.approx(0.5)


def test_build_ppo_model_applies_hybrid_spin_idle_logit_bias() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=configured_hybrid_action(
                        continuous_axes=("steer",),
                        discrete_axes=("gas", "air_brake", "boost", "lean", "spin", "pitch"),
                    ),
                    observation=ObservationConfig(
                        mode="image_state",
                        state_components=_VEHICLE_STATE_COMPONENT,
                    ),
                ),
            )
        )
    )

    try:
        dimensions = env.get_attr("action_dimensions")[0]
        spin_offset = 0
        for dimension in dimensions:
            if dimension.label == "spin":
                break
            spin_offset += dimension.size
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_recurrent_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(
                recurrent=PolicyRecurrentConfig(
                    enabled=True,
                    hidden_size=512,
                    n_lstm_layers=1,
                ),
                action_bias=PolicyActionBiasConfig(spin_idle_logit=1.0),
            ),
            tensorboard_log=None,
        )
    finally:
        env.close()

    bias = model.policy.state_dict()["action_net.discrete_net.bias"].detach().cpu()
    assert float(bias[spin_offset]) == pytest.approx(1.0)
    assert float(bias[spin_offset + 1]) == pytest.approx(0.0)
    assert float(bias[spin_offset + 2]) == pytest.approx(0.0)
    assert getattr(model, MODEL_ACTION_BIAS_OFFSETS_ATTR)["spin_idle_logit"] == pytest.approx(1.0)


def test_resume_action_bias_delta_does_not_stack_spin_idle_bias() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=configured_hybrid_action(
                        continuous_axes=("steer",),
                        discrete_axes=("gas", "air_brake", "boost", "lean", "spin", "pitch"),
                    ),
                    observation=ObservationConfig(
                        mode="image_state",
                        state_components=_VEHICLE_STATE_COMPONENT,
                    ),
                ),
            )
        )
    )

    try:
        dimensions = env.get_attr("action_dimensions")[0]
        spin_offset = 0
        for dimension in dimensions:
            if dimension.label == "spin":
                break
            spin_offset += dimension.size
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_recurrent_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(
                recurrent=PolicyRecurrentConfig(
                    enabled=True,
                    hidden_size=512,
                    n_lstm_layers=1,
                ),
            ),
            tensorboard_log=None,
        )
        policy_config = PolicyConfig(
            recurrent=PolicyRecurrentConfig(
                enabled=True,
                hidden_size=512,
                n_lstm_layers=1,
            ),
            action_bias=PolicyActionBiasConfig(spin_idle_logit=1.0),
        )
        apply_resume_action_bias_delta(model, train_env=env, policy_config=policy_config)
        apply_resume_action_bias_delta(model, train_env=env, policy_config=policy_config)
    finally:
        env.close()

    bias = model.policy.state_dict()["action_net.discrete_net.bias"].detach().cpu()
    assert float(bias[spin_offset]) == pytest.approx(1.0)
    assert getattr(model, MODEL_ACTION_BIAS_OFFSETS_ATTR)["spin_idle_logit"] == pytest.approx(1.0)


def test_markerless_resume_action_bias_delta_skips_legacy_gas_bias() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=configured_hybrid_action(
                        continuous_axes=("steer",),
                        discrete_axes=("gas", "air_brake", "boost", "lean", "spin", "pitch"),
                    ),
                    observation=ObservationConfig(
                        mode="image_state",
                        state_components=_VEHICLE_STATE_COMPONENT,
                    ),
                ),
            )
        )
    )

    try:
        dimensions = env.get_attr("action_dimensions")[0]
        spin_offset = 0
        for dimension in dimensions:
            if dimension.label == "spin":
                break
            spin_offset += dimension.size
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_recurrent_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(
                recurrent=PolicyRecurrentConfig(
                    enabled=True,
                    hidden_size=512,
                    n_lstm_layers=1,
                ),
            ),
            tensorboard_log=None,
        )
        delattr(model, MODEL_ACTION_BIAS_OFFSETS_ATTR)
        apply_resume_action_bias_delta(
            model,
            train_env=env,
            policy_config=PolicyConfig(
                recurrent=PolicyRecurrentConfig(
                    enabled=True,
                    hidden_size=512,
                    n_lstm_layers=1,
                ),
                action_bias=PolicyActionBiasConfig(
                    gas_on_logit=1.0,
                    spin_idle_logit=0.5,
                ),
            ),
        )
    finally:
        env.close()

    bias = model.policy.state_dict()["action_net.discrete_net.bias"].detach().cpu()
    assert float(bias[1]) == pytest.approx(0.0)
    assert float(bias[spin_offset]) == pytest.approx(0.5)


def test_build_ppo_model_can_construct_maskable_recurrent_ppo() -> None:
    from sb3x import MaskableRecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(
                    action=configured_discrete_action(
                        "steer",
                        "gas",
                        "boost",
                        "lean",
                        mask=ActionMaskConfig(lean=(0,)),
                    ),
                    observation=ObservationConfig(
                        mode="image_state",
                        state_components=_VEHICLE_STATE_COMPONENT,
                    ),
                ),
            )
        )
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_recurrent_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(recurrent=PolicyRecurrentConfig(enabled=True)),
            tensorboard_log=None,
        )
    finally:
        env.close()

    assert isinstance(model, MaskableRecurrentPPO)
