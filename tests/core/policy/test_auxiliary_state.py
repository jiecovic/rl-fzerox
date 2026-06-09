# tests/core/policy/test_auxiliary_state.py
from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from sb3x.common.auxiliary_losses import PolicyAuxiliaryLoss

from rl_fzerox.core.policy.auxiliary_state.actor_regularization import (
    _AxisDistributionStats,
    _categorical_lean_expected_signed_values,
    _signed_balance_loss,
    _split_lean_expected_signed_values,
    _std_cap_loss,
)
from rl_fzerox.core.policy.auxiliary_state.mixin import _AuxiliaryStatePolicyMixin
from rl_fzerox.core.policy.auxiliary_state.observations import (
    auxiliary_state_targets_field,
)
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


class _PitchStdCapHarness(_AuxiliaryStatePolicyMixin):
    def __init__(self) -> None:
        self._pitch_std_cap_loss_weight = 1.0
        self._grounded_pitch_std_cap = 0.3
        self._airborne_pitch_std_cap = 0.3

    def pitch_std_cap_loss(
        self,
        stats: _AxisDistributionStats,
        *,
        obs: dict[str, torch.Tensor],
    ) -> PolicyAuxiliaryLoss | None:
        return self._pitch_std_cap_loss(stats, obs=obs, sample_mask=None)


def _hybrid_discrete_distribution(*probabilities: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        discrete_dist=SimpleNamespace(
            distributions=tuple(
                torch.distributions.Categorical(probs=probs) for probs in probabilities
            )
        )
    )


def _auxiliary_target_observation(
    *,
    airborne_flags: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    targets = torch.zeros((len(airborne_flags), auxiliary_state_target_spec().count))
    airborne_index = resolve_auxiliary_state_target("vehicle_state.airborne").vector_start
    targets[:, airborne_index] = torch.tensor(airborne_flags)
    return {auxiliary_state_targets_field(): targets}


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


def test_discrete_pitch_std_cap_loss_skips_airborne_scope() -> None:
    stats = _AxisDistributionStats(
        mean=torch.zeros(2),
        std=torch.tensor([0.8, 0.8]),
        entropy=torch.zeros(2),
        source="discrete",
    )

    loss = _PitchStdCapHarness().pitch_std_cap_loss(
        stats,
        obs=_auxiliary_target_observation(airborne_flags=(0.0, 1.0)),
    )

    assert loss is not None
    assert float(loss.total_loss) == pytest.approx(0.25)
    assert "pitch/grounded_std_cap_loss" in loss.metrics
    assert "pitch/airborne_std_cap_loss" not in loss.metrics


def test_continuous_pitch_std_cap_loss_keeps_airborne_scope() -> None:
    stats = _AxisDistributionStats(
        mean=torch.zeros(2),
        std=torch.tensor([0.8, 0.8]),
        entropy=torch.zeros(2),
        source="continuous",
        log_std=torch.zeros(2),
    )

    loss = _PitchStdCapHarness().pitch_std_cap_loss(
        stats,
        obs=_auxiliary_target_observation(airborne_flags=(0.0, 1.0)),
    )

    assert loss is not None
    assert float(loss.total_loss) == pytest.approx(0.5)
    assert "pitch/grounded_std_cap_loss" in loss.metrics
    assert "pitch/airborne_std_cap_loss" in loss.metrics


def test_categorical_lean_signed_balance_treats_both_as_neutral() -> None:
    distribution = _hybrid_discrete_distribution(torch.tensor([[0.0, 0.7, 0.2, 0.1]]))

    expected = _categorical_lean_expected_signed_values(distribution, branch_index=0)

    assert expected.tolist() == pytest.approx([-0.5])


def test_split_lean_signed_balance_uses_right_minus_left_probability() -> None:
    distribution = _hybrid_discrete_distribution(
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[0.9, 0.1]]),
    )

    expected = _split_lean_expected_signed_values(
        distribution,
        left_branch_index=0,
        right_branch_index=1,
    )

    assert expected.tolist() == pytest.approx([-0.7])


def test_signed_balance_loss_penalizes_batch_bias_outside_deadzone() -> None:
    loss = _signed_balance_loss(
        torch.tensor([0.25, 0.35]),
        deadzone=0.1,
        loss_weight=2.0,
        sample_mask=None,
    )

    assert loss is not None
    assert float(loss.bias) == pytest.approx(0.3)
    assert float(loss.loss_value) == pytest.approx(0.04)
    assert float(loss.total_loss) == pytest.approx(0.08)


def test_signed_balance_loss_respects_sample_mask() -> None:
    loss = _signed_balance_loss(
        torch.tensor([0.25, 0.9]),
        deadzone=0.1,
        loss_weight=2.0,
        sample_mask=torch.tensor([True, False]),
    )

    assert loss is not None
    assert float(loss.bias) == pytest.approx(0.25)
    assert float(loss.loss_value) == pytest.approx(0.0225)


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
