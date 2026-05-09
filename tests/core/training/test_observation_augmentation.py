# tests/core/training/test_observation_augmentation.py
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_fzerox.core.policy.auxiliary_state.observations import (
    auxiliary_state_targets_field,
)
from rl_fzerox.core.runtime_spec.schema import (
    EnvConfig,
    ObservationConfig,
    StateFeatureDropoutGroupConfig,
    TrainConfig,
)
from rl_fzerox.core.training.session.observation_augmentation import (
    EpisodeStateFeatureDropoutWrapper,
    _StateFeatureDropoutGroup,
    maybe_wrap_training_observation_augmentation,
)


def test_state_feature_dropout_wrapper_zeros_selected_state_features() -> None:
    env = EpisodeStateFeatureDropoutWrapper(
        _DictStateEnv(),
        dropout_groups=(
            _StateFeatureDropoutGroup(feature_indices=(1, 2), dropout_prob=1.0),
        ),
    )

    observation, _info = env.reset(seed=123)

    assert isinstance(observation, dict)
    np.testing.assert_array_equal(
        observation["state"],
        np.array([1.0, 0.0, 0.0, 4.0], dtype=np.float32),
    )


def test_state_feature_dropout_wrapper_preserves_auxiliary_targets() -> None:
    env = EpisodeStateFeatureDropoutWrapper(
        _DictImageStateAuxEnv(),
        dropout_groups=(
            _StateFeatureDropoutGroup(feature_indices=(1, 2), dropout_prob=1.0),
        ),
    )

    observation, _info = env.reset(seed=123)

    assert isinstance(observation, dict)
    field_name = auxiliary_state_targets_field()
    assert field_name in observation
    np.testing.assert_array_equal(
        observation["state"],
        np.array([1.0, 0.0, 0.0, 4.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        observation[field_name],
        np.array([0.25, 0.5], dtype=np.float32),
    )


def test_training_observation_augmentation_wraps_course_context_state() -> None:
    env = maybe_wrap_training_observation_augmentation(
        _DictStateEnv(),
        env_config=EnvConfig(
            observation=ObservationConfig.model_validate(
                {
                    "mode": "image_state",
                    "state_components": [
                        "vehicle_state",
                        {"course_context": {"encoding": "one_hot_builtin"}},
                    ],
                }
            )
        ),
        train_config=TrainConfig(
            state_feature_dropout_groups=(
                StateFeatureDropoutGroupConfig(
                    feature_names=tuple(
                        f"course_context.course_builtin_{index:02d}" for index in range(24)
                    ),
                    dropout_prob=0.25,
                ),
            )
        ),
    )

    assert isinstance(env, EpisodeStateFeatureDropoutWrapper)


def test_training_observation_augmentation_supports_feature_level_dropout_groups() -> None:
    env = maybe_wrap_training_observation_augmentation(
        _DictStateEnv(),
        env_config=EnvConfig(
            observation=ObservationConfig.model_validate(
                {
                    "mode": "image_state",
                    "state_components": ["vehicle_state"],
                }
            )
        ),
        train_config=TrainConfig(
            state_feature_dropout_groups=(
                StateFeatureDropoutGroupConfig(
                    feature_names=("vehicle_state.speed_norm",),
                    dropout_prob=1.0,
                ),
            )
        ),
    )

    observation, _info = env.reset(seed=123)

    assert isinstance(env, EpisodeStateFeatureDropoutWrapper)
    assert isinstance(observation, dict)
    np.testing.assert_array_equal(
        observation["state"][:4],
        np.array([0.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )


def test_training_observation_augmentation_skips_when_disabled() -> None:
    base_env = _DictStateEnv()

    env = maybe_wrap_training_observation_augmentation(
        base_env,
        env_config=EnvConfig(
            observation=ObservationConfig.model_validate(
                {
                    "mode": "image_state",
                    "state_components": ["vehicle_state"],
                }
            )
        ),
        train_config=TrainConfig(),
    )

    assert env is base_env


class _DictStateEnv(gym.Env):
    action_space = spaces.Discrete(1)
    observation_space = spaces.Dict(
        {
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }
    )

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        del options
        super().reset(seed=seed)
        return {"state": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}, {}

    def step(self, action: object):
        del action
        return self.reset()[0], 0.0, False, False, {}


class _DictImageStateAuxEnv(gym.Env):
    action_space = spaces.Discrete(1)
    observation_space = spaces.Dict(
        {
            "image": spaces.Box(low=0, high=255, shape=(4, 4, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            auxiliary_state_targets_field(): spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
        }
    )

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        del options
        super().reset(seed=seed)
        return {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "state": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            auxiliary_state_targets_field(): np.array([0.25, 0.5], dtype=np.float32),
        }, {}

    def step(self, action: object):
        del action
        return self.reset()[0], 0.0, False, False, {}
