# tests/core/training/test_observation_augmentation.py
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_fzerox.core.config.schema import EnvConfig, ObservationConfig, TrainConfig
from rl_fzerox.core.training.session.observation_augmentation import (
    CourseContextDropoutWrapper,
    maybe_wrap_training_observation_augmentation,
)


def test_course_context_dropout_wrapper_zeros_selected_state_features() -> None:
    env = CourseContextDropoutWrapper(
        _DictStateEnv(),
        dropout_prob=1.0,
        feature_indices=(1, 2),
    )

    observation, _info = env.reset(seed=123)

    assert isinstance(observation, dict)
    np.testing.assert_array_equal(
        observation["state"],
        np.array([1.0, 0.0, 0.0, 4.0], dtype=np.float32),
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
        train_config=TrainConfig(course_context_dropout_prob=0.25),
    )

    assert isinstance(env, CourseContextDropoutWrapper)


def test_training_observation_augmentation_skips_when_disabled() -> None:
    base_env = _DictStateEnv()

    env = maybe_wrap_training_observation_augmentation(
        base_env,
        env_config=EnvConfig(observation=ObservationConfig(mode="image_state")),
        train_config=TrainConfig(course_context_dropout_prob=0.0),
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
