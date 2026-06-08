# tests/core/policy/test_auxiliary_state.py
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_fzerox.core.policy.auxiliary_state.observations import (
    auxiliary_state_targets_field,
)
from rl_fzerox.core.policy.auxiliary_state.policies import _std_cap_loss
from rl_fzerox.core.policy.auxiliary_state.targets import (
    auxiliary_state_target_name_for_feature,
    auxiliary_state_target_spec,
    auxiliary_state_target_vector,
    auxiliary_state_target_vector_or_zeros,
    resolve_auxiliary_state_target,
)
from rl_fzerox.core.training.session.auxiliary_state import (
    AuxiliaryStateObservationWrapper,
)
from tests.support.native_objects import make_telemetry


def test_auxiliary_state_target_vector_matches_expected_slots() -> None:
    telemetry = make_telemetry(
        course_index=7,
        speed_kph=750.0,
        energy=89.0,
        max_energy=178.0,
        reverse_timer=3,
        lap_distance=40_000.0,
        course_length=80_000.0,
        local_lateral_velocity=-16.0,
        height_above_ground=500.0,
    )

    vector = auxiliary_state_target_vector(telemetry)
    spec = auxiliary_state_target_spec()

    assert vector.shape == (spec.count,)
    assert float(vector[0]) == 0.5
    assert float(vector[1]) == 0.5
    assert float(vector[2]) == 1.0
    assert float(vector[8]) == 0.5
    assert float(vector[10]) == 0.5
    assert int(np.argmax(vector[15:39])) == 7


def test_pitch_std_cap_loss_caps_existing_pitch_std_per_sample() -> None:
    pitch_std = torch.tensor([0.6, 0.2])

    loss = _std_cap_loss(pitch_std, cap=0.5, sample_mask=None)

    assert loss is not None
    assert float(loss) == pytest.approx(0.005)


def test_pitch_std_cap_loss_ignores_inactive_samples() -> None:
    values = torch.tensor([0.6, 0.2, 100.0])
    mask = torch.tensor([True, True, False])

    loss = _std_cap_loss(values, cap=0.5, sample_mask=mask)

    assert loss is not None
    assert float(loss) == pytest.approx(0.005)


def test_auxiliary_state_target_vector_returns_zeros_without_telemetry() -> None:
    vector = auxiliary_state_target_vector_or_zeros(None)

    assert vector.shape == (auxiliary_state_target_spec().count,)
    assert np.count_nonzero(vector) == 0


def test_auxiliary_state_refill_surface_target_ignores_energy_fullness() -> None:
    telemetry = make_telemetry(
        state_flags=1,
        energy=178.0,
        max_energy=178.0,
    )

    vector = auxiliary_state_target_vector(telemetry)
    refill_surface = resolve_auxiliary_state_target("surface_state.on_refill_surface")

    assert telemetry.player.on_energy_refill is False
    assert float(vector[refill_surface.vector_start]) == 1.0


def test_course_one_hot_features_map_to_course_auxiliary_target() -> None:
    assert (
        auxiliary_state_target_name_for_feature("course_context.course_builtin_00")
        == "course_context.builtin_course_id"
    )
    assert (
        auxiliary_state_target_name_for_feature("course_context.course_builtin_23")
        == "course_context.builtin_course_id"
    )


class _DummyDictEnv(gym.Env[dict[str, np.ndarray], np.int64]):
    observation_space = spaces.Dict(
        {
            "image": spaces.Box(low=0, high=255, shape=(4, 4, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        }
    )
    action_space = spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        del seed, options
        observation = {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "state": np.array([0.25, -0.5], dtype=np.float32),
        }
        info = {auxiliary_state_targets_field(): np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        return observation, info

    def step(self, action: np.int64):
        del action
        observation = {
            "image": np.ones((4, 4, 3), dtype=np.uint8),
            "state": np.array([0.5, 0.75], dtype=np.float32),
        }
        info = {auxiliary_state_targets_field(): np.array([4.0, 5.0, 6.0], dtype=np.float32)}
        return observation, 0.0, False, False, info


def test_auxiliary_state_observation_wrapper_attaches_hidden_targets() -> None:
    wrapped = AuxiliaryStateObservationWrapper(_DummyDictEnv())
    assert isinstance(wrapped.observation_space, spaces.Dict)

    observation, info = wrapped.reset()
    assert isinstance(observation, dict)
    field_name = auxiliary_state_targets_field()
    assert field_name in wrapped.observation_space.spaces
    assert np.array_equal(
        observation[field_name],
        info[field_name],
    )

    next_observation, _, _, _, next_info = wrapped.step(0)
    assert isinstance(next_observation, dict)
    assert np.array_equal(
        next_observation[field_name],
        next_info[field_name],
    )
