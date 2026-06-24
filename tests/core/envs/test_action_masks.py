# tests/core/envs/test_action_masks.py
import numpy as np
import pytest

from fzerox_emulator.arrays import Float32Array, Int64Array
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.runtime_spec.schema import (
    ActionMaskConfig,
    EnvConfig,
    RewardConfig,
)
from tests.core.envs.helpers import ScriptedStepBackend
from tests.core.envs.helpers import backend_step_result as _backend_step_result
from tests.core.envs.helpers import step_summary as _step_summary
from tests.core.envs.helpers import telemetry as _telemetry
from tests.support.action_configs import (
    configured_discrete_action,
    configured_hybrid_action,
)
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import make_step_status


def _discrete_gas_boost_lean_action(*, lean_index: int = 0, boost_index: int = 0) -> Int64Array:
    return np.array([3, 0, boost_index, lean_index], dtype=np.int64)


def _discrete_gas_boost_lean_spin_action(
    *,
    lean_index: int = 0,
    boost_index: int = 0,
    spin_index: int = 0,
) -> Int64Array:
    return np.array([3, 0, boost_index, lean_index, spin_index], dtype=np.int64)


def _discrete_gas_boost_action(*, boost_index: int = 0) -> Int64Array:
    return np.array([3, 0, boost_index], dtype=np.int64)


def _discrete_gas_air_brake_boost_action(
    *,
    air_brake_index: int = 0,
    boost_index: int = 0,
) -> Int64Array:
    return np.array([3, 0, air_brake_index, boost_index], dtype=np.int64)


def _hybrid_boost_lean_action(
    *,
    boost_index: int = 0,
    lean_index: int = 0,
) -> dict[str, Float32Array | Int64Array]:
    return {
        "continuous": np.array([0.0, 1.0], dtype=np.float32),
        "discrete": np.array([boost_index, lean_index], dtype=np.int64),
    }


def test_env_action_masks_reflect_base_action_mask_config() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "boost",
                "lean",
                mask=ActionMaskConfig(lean=(0,)),
            )
        ),
    )

    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 2) + ([True] * 2) + [True, False, False]
    )


def test_env_action_masks_reject_base_mask_branch_missing_from_adapter() -> None:
    with pytest.raises(ValueError, match="env\\.action\\.mask.*'gas'"):
        FZeroXEnv(
            backend=SyntheticBackend(),
            config=EnvConfig(
                action=configured_hybrid_action(
                    continuous_axes=("steer", "drive"),
                    discrete_axes=("boost", "lean"),
                    mask=ActionMaskConfig(gas=(0,)),
                )
            ),
        )


def test_hybrid_env_action_masks_disable_lean_for_initial_episode_frames() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(max_race_distance=10.0, frames_run=2),
                status=make_step_status(step_count=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=20.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(max_race_distance=20.0, frames_run=2),
                status=make_step_status(step_count=2),
            ),
        ],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_hybrid_action(
                continuous_axes=("steer", "drive"),
                discrete_axes=("boost", "lean"),
                lean_initial_lockout_frames=4,
            )
        ),
    )

    env.reset(seed=1)
    assert env.action_masks().tolist() == [True, True, True, False, False]

    env.step(_hybrid_boost_lean_action())
    assert env.action_masks().tolist() == [True, True, True, False, False]

    env.step(_hybrid_boost_lean_action())
    assert env.action_masks().tolist() == [True, True, True, True, True]


def test_env_action_masks_disable_boost_until_telemetry_unlocks_it() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, frames_run=2, final_frame_index=2),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active",)),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action=configured_discrete_action("steer", "gas", "boost")),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 2) + [True, False])

    env.step(_discrete_gas_boost_action())

    assert env.action_masks().tolist() == ([True] * (7 + 2 + 2))


def test_hybrid_env_action_masks_disable_boost_until_telemetry_unlocks_it() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, frames_run=2, final_frame_index=2),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active",)),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_hybrid_action(
                continuous_axes=("steer", "drive"),
                discrete_axes=("boost", "lean"),
            )
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == [True, False, True, True, True]

    env.step(_hybrid_boost_lean_action())

    assert env.action_masks().tolist() == [True, True, True, True, True]


def test_env_action_masks_disable_lean_below_speed_threshold() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "can_boost"),
                    speed_kph=650.0,
                ),
                summary=_step_summary(max_race_distance=10.0, frames_run=2, final_frame_index=2),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
            speed_kph=300.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "boost",
                "lean",
                lean_unmask_min_speed_kph=500.0,
            ),
        ),
    )

    env.reset(seed=1)
    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 2) + ([True] * 2) + [True, False, False]
    )

    env.step(_discrete_gas_boost_lean_action())
    assert env.action_masks().tolist() == ([True] * (7 + 2 + 2 + 3))


def test_env_action_masks_can_mask_lean_for_full_episode() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active", "can_boost")),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "boost",
                "lean",
                "spin",
                lean_episode_mask_probability=1.0,
            ),
        ),
    )

    _, reset_info = env.reset(seed=1)

    assert reset_info["lean_episode_masked"] is True
    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 2) + ([True] * 2) + [True, False, False] + [True, False, False]
    )

    _, _, _, _, info = env.step(_discrete_gas_boost_lean_spin_action(lean_index=1, spin_index=1))

    assert info["lean_episode_masked"] is True
    assert info["spin_requested"] is False
    assert not backend.last_race_control_state.lean_left
    assert not backend.last_race_control_state.lean_right


def test_env_action_masks_can_mask_discrete_air_brake_for_full_episode() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active", "can_boost")),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "air_brake",
                "boost",
                air_brake_episode_mask_probability=1.0,
                mask_air_brake_on_ground=False,
            ),
        ),
    )

    _, reset_info = env.reset(seed=1)

    assert reset_info["air_brake_episode_masked"] is True
    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 2) + [True, False] + ([True] * 2)
    )

    _, _, _, _, info = env.step(_discrete_gas_air_brake_boost_action(air_brake_index=1))

    assert info["air_brake_episode_masked"] is True
    assert info["air_brake_used"] is False
    assert not backend.last_race_control_state.air_brake


def test_env_action_masks_ignore_air_brake_episode_mask_without_discrete_branch() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active", "can_boost")),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_hybrid_action(
                continuous_axes=("steer", "air_brake"),
                discrete_axes=("gas",),
                air_brake_episode_mask_probability=1.0,
                mask_air_brake_on_ground=False,
            ),
        ),
    )

    _, reset_info = env.reset(seed=1)

    assert reset_info["air_brake_episode_masked"] is False
    assert env.action_masks().tolist() == [True, True]

    _, _, _, _, info = env.step(
        {
            "continuous": np.array([0.0, 1.0], dtype=np.float32),
            "discrete": np.array([1], dtype=np.int64),
        }
    )

    assert info["air_brake_episode_masked"] is False
    assert info["air_brake_used"] is True
    assert backend.last_race_control_state.air_brake


def test_env_action_masks_can_mask_spin_for_full_episode() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active", "can_boost")),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "boost",
                "lean",
                "spin",
                spin_episode_mask_probability=1.0,
            ),
        ),
    )

    _, reset_info = env.reset(seed=1)

    assert reset_info["spin_episode_masked"] is True
    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 2) + ([True] * 2) + ([True] * 3) + [True, False, False]
    )

    _, _, _, _, info = env.step(_discrete_gas_boost_lean_spin_action(spin_index=1))

    assert info["spin_episode_masked"] is True
    assert info["spin_requested"] is False
    assert backend.last_spin_request == "none"


def test_env_action_masks_disable_spin_and_lean_during_native_spin_macro() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(
                    max_race_distance=10.0,
                    final_frame_index=1,
                    spin_macro_started=True,
                    spin_macro_active_frames=1,
                    lean_macro_owned_frames=1,
                ),
                status=make_step_status(
                    step_count=1,
                    spin_macro_active=True,
                    spin_macro_frames_remaining=5,
                ),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active", "can_boost")),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_discrete_action("steer", "gas", "boost", "lean", "spin"),
        ),
    )

    env.reset(seed=1)
    assert env.action_masks().tolist() == ([True] * (7 + 2 + 2 + 3 + 3))

    _, _, _, _, info = env.step(_discrete_gas_boost_lean_spin_action(spin_index=1))

    assert info["spin_requested"] is True
    assert info["spin_started"] is True
    assert info["spin_macro_active_frames"] == 1
    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 2) + ([True] * 2) + [True, False, False] + [True, False, False]
    )


def test_env_action_masks_disable_spin_and_lean_during_native_spin_cooldown() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(
                    max_race_distance=10.0,
                    final_frame_index=1,
                    spin_macro_started=False,
                    spin_macro_active_frames=0,
                    lean_macro_owned_frames=0,
                ),
                status=make_step_status(
                    step_count=1,
                    spin_macro_active=False,
                    spin_macro_cooldown_frames=42,
                ),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active", "can_boost")),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_discrete_action("steer", "gas", "boost", "lean", "spin"),
        ),
    )

    env.reset(seed=1)
    assert env.action_masks().tolist() == ([True] * (7 + 2 + 2 + 3 + 3))

    env.step(_discrete_gas_boost_lean_spin_action())

    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 2) + ([True] * 2) + [True, False, False] + [True, False, False]
    )


def test_env_action_masks_disable_boost_below_energy_threshold() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "can_boost"),
                    energy=20.0,
                    max_energy=100.0,
                ),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
            energy=5.0,
            max_energy=100.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            boost_min_energy_fraction=0.1,
            action=configured_discrete_action("steer", "gas", "boost"),
        ),
    )

    env.reset(seed=1)
    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 2) + [True, False])

    env.step(_discrete_gas_boost_action())
    assert env.action_masks().tolist() == ([True] * (7 + 2 + 2))


def test_env_action_masks_disable_zero_energy_boost_and_reward() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=0.0,
                    state_labels=("active", "can_boost"),
                    energy=0.0,
                    max_energy=100.0,
                ),
                summary=_step_summary(max_race_distance=0.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
            energy=0.0,
            max_energy=100.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            boost_min_energy_fraction=0.0,
            action=configured_discrete_action("steer", "gas", "boost"),
        ),
        reward_config=RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            manual_boost_reward=1.0,
            impact_frame_penalty=0.0,
        ),
    )

    env.reset(seed=1)
    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 2) + [True, False])

    _, reward, _, _, info = env.step(_discrete_gas_boost_action(boost_index=1))

    assert reward == 0.0
    reward_breakdown = info.get("reward_breakdown", {})
    assert isinstance(reward_breakdown, dict)
    assert "manual_boost" not in reward_breakdown


def test_env_action_masks_disable_boost_while_dash_pad_boost_is_active_by_default() -> None:
    env = FZeroXEnv(
        backend=ScriptedStepBackend(
            [],
            reset_telemetry=_telemetry(
                race_distance=0.0,
                state_labels=("active", "can_boost", "dash_pad_boost"),
            ),
        ),
        config=EnvConfig(action=configured_discrete_action("steer", "gas", "boost")),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 2) + [True, False])


def test_env_action_masks_can_allow_boost_while_dash_pad_boost_is_active() -> None:
    env = FZeroXEnv(
        backend=ScriptedStepBackend(
            [],
            reset_telemetry=_telemetry(
                race_distance=0.0,
                state_labels=("active", "can_boost", "dash_pad_boost"),
            ),
        ),
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "boost",
                mask_boost_when_active=False,
                boost_request_lockout_frames=0,
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == ([True] * (7 + 2 + 2))


def test_env_action_masks_disable_boost_while_airborne_by_default() -> None:
    env = FZeroXEnv(
        backend=ScriptedStepBackend(
            [],
            reset_telemetry=_telemetry(
                race_distance=0.0,
                state_labels=("active", "can_boost", "airborne"),
            ),
        ),
        config=EnvConfig(action=configured_discrete_action("steer", "gas", "boost")),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 2) + [True, False])


def test_env_action_masks_can_allow_boost_while_airborne() -> None:
    env = FZeroXEnv(
        backend=ScriptedStepBackend(
            [],
            reset_telemetry=_telemetry(
                race_distance=0.0,
                state_labels=("active", "can_boost", "airborne"),
            ),
        ),
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "boost",
                mask_boost_when_airborne=False,
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == ([True] * (7 + 2 + 2))


def test_env_action_masks_apply_boost_request_cooldown_without_active_boost_mask() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "boost",
                mask_boost_when_active=False,
                boost_request_lockout_frames=5,
            ),
        ),
    )

    env.reset(seed=1)
    env.step(_discrete_gas_boost_action(boost_index=1))

    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 2) + [True, False])


def test_env_action_masks_apply_boost_decision_interval() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(max_race_distance=10.0, frames_run=2, final_frame_index=2),
                status=make_step_status(step_count=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=20.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(max_race_distance=20.0, frames_run=2, final_frame_index=4),
                status=make_step_status(step_count=2),
            ),
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=30.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(max_race_distance=30.0, frames_run=2, final_frame_index=6),
                status=make_step_status(step_count=3),
            ),
        ],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "boost",
                boost_decision_interval_frames=4,
                boost_request_lockout_frames=0,
                mask_boost_when_active=False,
            ),
        ),
    )

    env.reset(seed=1)
    env.step(_discrete_gas_boost_action(boost_index=0))

    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 2) + [True, False])

    env.step(_discrete_gas_boost_action(boost_index=0))

    assert env.action_masks().tolist() == ([True] * (7 + 2 + 2))


def test_env_action_masks_disable_boost_above_speed_threshold() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "can_boost"),
                    speed_kph=650.0,
                ),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
            speed_kph=750.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "boost",
                boost_unmask_max_speed_kph=700.0,
            ),
        ),
    )

    env.reset(seed=1)
    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 2) + [True, False])

    env.step(_discrete_gas_boost_action())
    assert env.action_masks().tolist() == ([True] * (7 + 2 + 2))


def test_env_control_gates_suppress_masked_lean_request() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
            speed_kph=500.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "boost",
                "lean",
                lean_unmask_min_speed_kph=900.0,
            ),
        ),
    )

    env.reset(seed=1)
    env.step(_discrete_gas_boost_lean_action(lean_index=1))

    assert not backend.last_race_control_state.lean_left


def test_env_control_gates_suppress_masked_independent_lean_request() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
            speed_kph=500.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_discrete_action(
                "steer",
                "gas",
                "boost",
                "lean_left",
                "lean_right",
                lean_output_mode="independent_buttons",
                lean_unmask_min_speed_kph=900.0,
            ),
        ),
    )

    env.reset(seed=1)
    env.step(np.array([3, 0, 0, 1, 0], dtype=np.int64))

    assert not backend.last_race_control_state.lean_left


def test_env_action_masks_keep_air_brake_and_pitch_airborne_only() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "airborne"),
                ),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active",)),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_hybrid_action(
                continuous_axes=("steer",),
                discrete_axes=("gas", "air_brake", "lean", "pitch"),
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        True,
        False,
        False,
    ]

    env.step(
        {
            "continuous": np.array([0.0], dtype=np.float32),
            "discrete": np.array([1, 1, 0, 4], dtype=np.int64),
        }
    )

    assert not backend.last_race_control_state.air_brake
    assert env.action_masks().tolist() == [True] * 12


def test_env_action_masks_can_leave_discrete_pitch_available_on_ground() -> None:
    env = FZeroXEnv(
        backend=ScriptedStepBackend(
            [],
            reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active",)),
        ),
        config=EnvConfig(
            action=configured_hybrid_action(
                continuous_axes=("steer",),
                discrete_axes=("gas", "air_brake", "lean", "pitch"),
                mask_pitch_on_ground=False,
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_env_action_masks_lock_boost_after_manual_request() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=20.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=20.0, final_frame_index=2),
                status=make_step_status(step_count=2),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=30.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=30.0, final_frame_index=3),
                status=make_step_status(step_count=3),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=40.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=40.0, final_frame_index=4),
                status=make_step_status(step_count=4),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=50.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=50.0, final_frame_index=5),
                status=make_step_status(step_count=5),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active", "can_boost")),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=configured_discrete_action("steer", "gas", "boost"),
        ),
    )

    env.reset(seed=1)
    assert env.action_masks().tolist() == ([True] * (7 + 2 + 2))

    env.step(_discrete_gas_boost_action(boost_index=1))
    assert backend.last_race_control_state.boost
    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 2) + [True, False])

    env.step(_discrete_gas_boost_action())
    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 2) + [True, False])

    env.step(_discrete_gas_boost_action())
    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 2) + [True, False])

    env.step(_discrete_gas_boost_action())
    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 2) + [True, False])

    env.step(_discrete_gas_boost_action())
    assert env.action_masks().tolist() == ([True] * (7 + 2 + 2))
