# tests/core/envs/test_env_controls.py

import numpy as np
import pytest

from fzerox_emulator import RaceControlState
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.runtime_spec.schema import (
    ActionMaskConfig,
    EnvConfig,
    ObservationConfig,
    RewardConfig,
)
from tests.core.envs.env_support import (
    _state_components,
)
from tests.core.envs.helpers import (
    ScriptedStepBackend,
)
from tests.core.envs.helpers import (
    backend_step_result as _backend_step_result,
)
from tests.core.envs.helpers import (
    step_summary as _step_summary,
)
from tests.core.envs.helpers import (
    telemetry as _telemetry,
)
from tests.support.action_configs import (
    configured_discrete_action,
    configured_hybrid_action,
)
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import make_step_status


def test_step_control_applies_manual_controller_state() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=2,
            action=configured_discrete_action("steer", "gas"),
        ),
    )

    env.reset(seed=21)
    control_state = RaceControlState.from_mask(5, stick_x=-1.0)
    env.step_control(control_state)

    assert backend.last_race_control_state == control_state


def test_step_control_spin_bypasses_policy_spin_mask_for_manual_macro() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=5.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(
                    max_race_distance=5.0,
                    final_frame_index=1,
                    spin_macro_started=True,
                    spin_macro_active_frames=1,
                ),
                status=make_step_status(step_count=1, spin_macro_active=True),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active", "can_boost")),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=configured_discrete_action(
                "steer",
                "gas",
                "boost",
                "lean",
                "spin",
                mask=ActionMaskConfig(spin=(0,)),
            ),
        ),
    )

    env.reset(seed=21)
    _, _, _, _, info = env.step_control(RaceControlState(), spin_request="left")

    assert backend.last_spin_request == "left"
    assert info["spin_requested"] is True
    assert info["spin_request"] == "left"
    assert info["spin_started"] is True


def test_step_control_suppresses_air_brake_until_airborne_when_configured() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=5.0,
                    state_labels=("active", "airborne"),
                ),
                summary=_step_summary(max_race_distance=5.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "airborne"),
                ),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=2),
                status=make_step_status(step_count=2),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active",)),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=configured_hybrid_action(
                continuous_axes=("steer", "drive", "air_brake"),
                discrete_axes=("boost", "lean"),
                continuous_drive_deadzone=0.0,
                continuous_air_brake_mode="disable_on_ground",
            ),
        ),
        reward_config=RewardConfig(time_penalty_per_frame=0.0),
    )
    air_brake_state = RaceControlState(air_brake=True)

    env.reset(seed=21)
    assert (
        env.action_to_control_state(
            {
                "continuous": np.array([0.0, -1.0, 1.0], dtype=np.float32),
                "discrete": np.array([0, 0], dtype=np.int64),
            }
        )
        == air_brake_state
    )
    _, reward, _, _, info = env.step_control(air_brake_state)
    assert backend.last_race_control_state == RaceControlState()
    assert reward == 0.0
    assert "reward_breakdown" not in info

    _, reward, _, _, info = env.step_control(air_brake_state)
    assert backend.last_race_control_state == air_brake_state
    assert reward == 0.0
    assert "reward_breakdown" not in info


def test_step_control_keeps_ground_air_brake_when_configured() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=5.0, state_labels=("active",)),
                summary=_step_summary(max_race_distance=5.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active",)),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=configured_hybrid_action(
                continuous_axes=("steer", "pitch"),
                discrete_axes=("gas", "air_brake", "boost", "lean"),
                mask_air_brake_on_ground=False,
            ),
        ),
        reward_config=RewardConfig(
            name="reward_main",
            time_penalty_per_frame=0.0,
            air_brake_request_penalty=-0.2,
            progress_bucket_reward=0.0,
            impact_frame_penalty=0.0,
        ),
    )
    air_brake_state = RaceControlState(air_brake=True)

    env.reset(seed=21)
    _, reward, _, _, info = env.step_control(air_brake_state)

    assert backend.last_race_control_state == air_brake_state
    assert reward == pytest.approx(-0.2)
    reward_breakdown = info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert reward_breakdown["air_brake"] == pytest.approx(-0.2)


def test_step_control_holds_discrete_air_brake_pulse() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=5.0, state_labels=("active",)),
                summary=_step_summary(max_race_distance=5.0, frames_run=2, final_frame_index=2),
                status=make_step_status(step_count=2),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active",)),
                summary=_step_summary(max_race_distance=10.0, frames_run=2, final_frame_index=4),
                status=make_step_status(step_count=4),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=15.0, state_labels=("active",)),
                summary=_step_summary(max_race_distance=15.0, frames_run=2, final_frame_index=6),
                status=make_step_status(step_count=6),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=20.0, state_labels=("active",)),
                summary=_step_summary(max_race_distance=20.0, frames_run=2, final_frame_index=8),
                status=make_step_status(step_count=8),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active",)),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=2,
            action=configured_hybrid_action(
                continuous_axes=("steer",),
                discrete_axes=("air_brake", "boost", "lean"),
                air_brake_pulse_frames=5,
                mask_air_brake_on_ground=False,
            ),
        ),
        reward_config=RewardConfig(time_penalty_per_frame=0.0),
    )

    env.reset(seed=21)
    _, _, _, _, info = env.step_control(RaceControlState(air_brake=True))

    assert backend.last_race_control_state == RaceControlState(air_brake=True)
    assert info["air_brake_requested"] is True
    assert info["air_brake_used"] is True
    assert info["air_brake_pulse_active"] is True
    assert info["air_brake_pulse_remaining_frames"] == 3
    assert env.action_mask_branches()["air_brake"] == (True, False)

    _, _, _, _, info = env.step_control(RaceControlState())

    assert backend.last_race_control_state == RaceControlState(air_brake=True)
    assert info["air_brake_requested"] is False
    assert info["air_brake_used"] is True
    assert info["air_brake_pulse_active"] is True
    assert info["air_brake_pulse_remaining_frames"] == 1
    assert env.action_mask_branches()["air_brake"] == (True, False)

    _, _, _, _, info = env.step_control(RaceControlState())

    assert backend.last_race_control_state == RaceControlState(air_brake=True)
    assert info["air_brake_requested"] is False
    assert info["air_brake_used"] is True
    assert info["air_brake_pulse_active"] is False
    assert info["air_brake_pulse_remaining_frames"] == 0
    assert env.action_mask_branches()["air_brake"] == (True, True)

    env.step_control(RaceControlState())

    assert backend.last_race_control_state == RaceControlState()


def test_step_suppresses_air_only_controls_until_airborne() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=5.0,
                    state_labels=("active", "airborne"),
                ),
                summary=_step_summary(max_race_distance=5.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "airborne"),
                ),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=2),
                status=make_step_status(step_count=2),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active",)),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=configured_hybrid_action(
                continuous_axes=("steer", "drive"),
                discrete_axes=("air_brake", "boost", "lean", "pitch"),
            ),
        ),
        reward_config=RewardConfig(time_penalty_per_frame=0.0),
    )
    airborne_action = {
        "continuous": np.array([0.0, -1.0], dtype=np.float32),
        "discrete": np.array([1, 0, 0, 4], dtype=np.int64),
    }

    env.reset(seed=21)
    env.step(airborne_action)

    assert backend.last_race_control_state == RaceControlState()

    env.step(airborne_action)

    assert backend.last_race_control_state == RaceControlState(air_brake=True, pitch=1.0)


def test_step_forces_accelerate_when_min_thrust_is_full() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=5.0),
                summary=_step_summary(max_race_distance=5.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=configured_hybrid_action(
                continuous_axes=("steer", "drive", "air_brake"),
                discrete_axes=("boost", "lean"),
                continuous_drive_min_thrust=1.0,
                continuous_air_brake_mode="off",
            ),
        ),
        reward_config=RewardConfig(time_penalty_per_frame=0.0),
    )

    env.reset(seed=21)
    _, reward, _, _, info = env.step(
        {
            "continuous": np.array([0.0, -0.5, 0.0], dtype=np.float32),
            "discrete": np.array([0, 0], dtype=np.int64),
        }
    )

    assert backend.last_race_control_state.gas
    assert reward == 0.0
    assert "reward_breakdown" not in info


def test_step_tracks_raw_continuous_gas_level_before_pwm_button_output() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=5.0),
                summary=_step_summary(max_race_distance=5.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=configured_hybrid_action(
                continuous_axes=("steer", "drive"),
                continuous_drive_deadzone=0.0,
                continuous_drive_full_threshold=1.0,
                continuous_drive_min_thrust=0.0,
            ),
            observation=ObservationConfig(
                mode="image_state",
                state_components=_state_components(
                    "vehicle_state",
                    {"control_history": {"length": 1, "controls": ("thrust",)}},
                ),
            ),
        ),
        reward_config=RewardConfig(time_penalty_per_frame=0.0),
    )

    env.reset(seed=21)
    obs, reward, _, _, info = env.step(
        {
            "continuous": np.array([0.0, -0.5], dtype=np.float32),
            "discrete": np.array([], dtype=np.int64),
        }
    )

    assert env.last_gas_level == pytest.approx(0.25)
    assert env.last_requested_control_state == RaceControlState()
    assert backend.last_race_control_state == RaceControlState()
    assert isinstance(obs, dict)
    raw_feature_names = info["observation_state_features"]
    assert isinstance(raw_feature_names, tuple)
    values = {
        name: float(value) for name, value in zip(raw_feature_names, obs["state"], strict=True)
    }
    assert values["control_history.prev_thrust_1"] == pytest.approx(0.25)
    assert reward == 0.0
    assert "reward_breakdown" not in info
