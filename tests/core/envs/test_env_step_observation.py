# tests/core/envs/test_env_step_observation.py

import numpy as np
import pytest

from fzerox_emulator import RaceControlState
from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.actions import RACE_CONTROL_MASKS
from rl_fzerox.core.runtime_spec.schema import (
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
    image_obs as _image_obs,
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


def test_step_advances_backend_by_action_repeat():
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=3,
            action=configured_discrete_action("steer", "gas"),
        ),
    )

    env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(np.array([3, 1], dtype=np.int64))
    obs = _image_obs(obs)

    assert obs.shape == (84, 84, 12)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert backend.frame_index == 3
    assert backend.capture_video_flags == [False, False, True]
    assert info["repeat_index"] == 2
    assert backend.last_race_control_state == RaceControlState(gas=True, stick_x=0.0)


def test_watch_step_captures_each_repeated_display_frame():
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=3,
            action=configured_discrete_action("steer", "gas"),
        ),
    )

    env.reset(seed=7)
    watch_step = env.step_watch(np.array([3, 1], dtype=np.int64))

    assert backend.frame_index == 3
    assert backend.capture_video_flags == [True, True, True]
    assert not isinstance(watch_step.display_frames, tuple)
    assert watch_step.display_frames.shape[0] == 3
    assert watch_step.display_frames[0].shape == (444, 592, 3)
    assert _image_obs(watch_step.observation).shape == (84, 84, 12)
    assert watch_step.info["repeat_index"] == 2


def test_step_clips_reward_and_exposes_raw_reward_diagnostics() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=configured_discrete_action("steer", "gas"),
        ),
        reward_config=RewardConfig(
            progress_bucket_distance=1.0,
            progress_bucket_reward=1.0,
            step_reward_clip_max=5.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        ),
    )

    env.reset(seed=7)
    _, reward, _, _, info = env.step(np.array([2, 0], dtype=np.int64))

    assert reward == 5.0
    assert info["step_reward"] == 5.0
    assert info["step_reward_raw"] == 10.0
    assert info["step_reward_clipped"] is True
    assert info["step_reward_clip_delta"] == -5.0
    assert info["step_reward_clip_abs_excess"] == 5.0
    assert info["step_reward_clip_positive"] is True
    assert info["step_reward_clip_negative"] is False
    assert info["episode_return"] == 5.0


def test_grounded_pitch_penalty_does_not_suppress_continuous_pitch() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=0.0),
                summary=_step_summary(max_race_distance=0.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=configured_hybrid_action(
                continuous_axes=("steer", "pitch"),
                discrete_axes=("gas",),
                pitch_deadzone=0.05,
            ),
        ),
        reward_config=RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            grounded_pitch_penalty=-0.5,
            impact_frame_penalty=0.0,
        ),
    )

    env.reset(seed=7)
    _, reward, _, _, info = env.step(
        {
            "continuous": np.array([0.0, 0.82], dtype=np.float32),
            "discrete": np.array([0], dtype=np.int64),
        }
    )

    assert backend.last_race_control_state.pitch == pytest.approx(0.82)
    expected = -0.5 * ((0.82 - 0.05) / (1.0 - 0.05))
    assert reward == pytest.approx(expected)
    reward_breakdown = info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert reward_breakdown["grounded_pitch"] == pytest.approx(expected)


def test_reset_resets_continuous_drive_pwm_phase() -> None:
    backend = SyntheticBackend()
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
        ),
    )

    env.reset(seed=7)
    env.step(
        {
            "continuous": np.array([0.0, -0.5], dtype=np.float32),
            "discrete": np.array([], dtype=np.int64),
        }
    )
    assert backend.last_race_control_state.control_mask == 0
    env.step(
        {
            "continuous": np.array([0.0, -0.5], dtype=np.float32),
            "discrete": np.array([], dtype=np.int64),
        }
    )
    assert backend.last_race_control_state.control_mask == 0
    env.step(
        {
            "continuous": np.array([0.0, -0.5], dtype=np.float32),
            "discrete": np.array([], dtype=np.int64),
        }
    )
    assert backend.last_race_control_state.control_mask == 0
    env.step(
        {
            "continuous": np.array([0.0, -0.5], dtype=np.float32),
            "discrete": np.array([], dtype=np.int64),
        }
    )
    assert backend.last_race_control_state.control_mask == RACE_CONTROL_MASKS.accelerate

    env.reset(seed=8)
    env.step(
        {
            "continuous": np.array([0.0, -0.5], dtype=np.float32),
            "discrete": np.array([], dtype=np.int64),
        }
    )

    assert backend.last_race_control_state.control_mask == 0


def test_step_updates_image_state_observation_from_step_telemetry() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    speed_kph=1_500.0,
                    energy=178.0,
                    max_energy=178.0,
                    reverse_timer=0,
                ),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=configured_discrete_action("steer", "gas"),
            observation=ObservationConfig(
                mode="image_state",
                frame_stack=4,
                state_components=_state_components("vehicle_state"),
            ),
        ),
    )

    env.reset(seed=7)
    obs, _, _, _, info = env.step(np.array([2, 0], dtype=np.int64))

    assert isinstance(obs, dict)
    assert set(obs) == {"image", "state"}
    assert obs["image"].shape == (84, 84, 12)
    raw_feature_names = info["observation_state_features"]
    assert isinstance(raw_feature_names, tuple)
    values = {
        name: float(value) for name, value in zip(raw_feature_names, obs["state"], strict=True)
    }
    assert values["vehicle_state.speed_norm"] == 1.0
    assert values["vehicle_state.energy_frac"] == 1.0
    assert values["vehicle_state.reverse_active"] == 0.0
    assert values["vehicle_state.airborne"] == 0.0
    assert values["vehicle_state.boost_active"] == 0.0
    assert info["observation_mode"] == "image_state"


def test_step_exposes_raw_step_signals_in_info() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, energy=120.0, max_energy=178.0),
                summary=_step_summary(
                    max_race_distance=10.0,
                    energy_loss_total=3.5,
                    damage_taken_frames=1,
                    final_frame_index=1,
                ),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, energy=123.5, max_energy=178.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=configured_discrete_action("steer", "gas"),
        ),
    )

    env.reset(seed=7)
    _, _, _, _, info = env.step(np.array([2, 0], dtype=np.int64))

    assert info["energy_loss_total"] == 3.5
    assert info["damage_taken_frames"] == 1


def test_step_updates_boost_control_history_in_image_state_observation() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=120,
            action=configured_discrete_action("steer", "gas", "boost"),
            observation=ObservationConfig(
                mode="image_state",
                frame_stack=4,
                state_components=_state_components(
                    "vehicle_state",
                    {"control_history": {"length": 1, "controls": ("boost",)}},
                ),
            ),
        ),
    )

    env.reset(seed=7)
    obs, _, _, _, _ = env.step(np.array([3, 0, 1], dtype=np.int64))

    assert isinstance(obs, dict)
    assert obs["state"][-1] == pytest.approx(1.0)


def test_step_updates_control_history_lean_branch() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=30,
            action=configured_discrete_action(
                "steer",
                "gas",
                "air_brake",
                "boost",
                "lean",
                steer_buckets=3,
            ),
            observation=ObservationConfig(
                mode="image_state",
                frame_stack=4,
                state_components=_state_components(
                    "vehicle_state",
                    {"control_history": {"length": 1, "controls": ("steer", "lean")}},
                ),
            ),
        ),
    )

    env.reset(seed=7)
    obs, _, _, _, info = env.step(np.array([0, 1, 1, 0, 0], dtype=np.int64))

    assert isinstance(obs, dict)
    assert info["observation_state_shape"] == (10,)
    raw_feature_names = info["observation_state_features"]
    assert isinstance(raw_feature_names, tuple)
    feature_names = tuple(str(name) for name in raw_feature_names)
    values = {name: float(value) for name, value in zip(feature_names, obs["state"], strict=True)}
    assert values["control_history.prev_steer_1"] == pytest.approx(-1.0)
    assert values["control_history.prev_lean_1"] == pytest.approx(0.0)


def test_step_updates_component_state_with_action_history() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=configured_hybrid_action(
                continuous_axes=("steer",),
                discrete_axes=("gas", "boost", "lean"),
            ),
            observation=ObservationConfig(
                mode="image_state",
                frame_stack=4,
                state_components=_state_components(
                    "vehicle_state",
                    {
                        "control_history": {
                            "length": 3,
                            "controls": ("steer", "gas", "boost", "lean"),
                        }
                    },
                ),
            ),
        ),
    )

    env.reset(seed=7)
    obs, _, _, _, info = env.step(
        {
            "continuous": np.array([-0.5], dtype=np.float32),
            "discrete": np.array([1, 1, 1], dtype=np.int64),
        }
    )

    assert isinstance(obs, dict)
    assert info["observation_action_history_len"] == 3
    assert info["observation_action_history_controls"] == ("steer", "gas", "boost", "lean")
    raw_feature_names = info["observation_state_features"]
    assert isinstance(raw_feature_names, tuple)
    feature_names = tuple(str(name) for name in raw_feature_names)
    values = {name: float(value) for name, value in zip(feature_names, obs["state"], strict=True)}
    assert "control_history.prev_air_brake_1" not in values
    assert values["control_history.prev_steer_1"] == pytest.approx(-0.5)
    assert values["control_history.prev_thrust_1"] == 1.0
    assert values["control_history.prev_boost_1"] == 1.0
    assert values["control_history.prev_lean_1"] == -1.0
    assert values["control_history.prev_steer_2"] == 0.0
    assert values["control_history.prev_steer_3"] == 0.0


def test_action_history_records_requested_continuous_pitch() -> None:
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
                continuous_axes=("steer", "pitch"),
                discrete_axes=("gas",),
            ),
            observation=ObservationConfig(
                mode="image_state",
                state_components=_state_components(
                    "vehicle_state",
                    {"control_history": {"length": 1, "controls": ("pitch",)}},
                ),
            ),
        ),
        reward_config=RewardConfig(time_penalty_per_frame=0.0),
    )
    env.reset(seed=7)

    obs, _, _, _, info = env.step(
        {
            "continuous": np.array([0.0, 0.5], dtype=np.float32),
            "discrete": np.array([1], dtype=np.int64),
        }
    )

    assert backend.last_race_control_state.pitch == pytest.approx(0.5)
    assert isinstance(obs, dict)
    raw_feature_names = info["observation_state_features"]
    assert isinstance(raw_feature_names, tuple)
    values = {
        name: float(value) for name, value in zip(raw_feature_names, obs["state"], strict=True)
    }
    assert values["control_history.prev_pitch_1"] == pytest.approx(0.5)


def test_step_updates_right_lean_control_history_in_image_state_observation() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=configured_discrete_action("steer", "gas", "boost", "lean"),
            observation=ObservationConfig(
                mode="image_state",
                frame_stack=4,
                state_components=_state_components(
                    "vehicle_state",
                    {"control_history": {"length": 1, "controls": ("lean",)}},
                ),
            ),
        ),
    )

    env.reset(seed=7)
    obs, _, _, _, _ = env.step(np.array([4, 1, 0, 2], dtype=np.int64))

    assert isinstance(obs, dict)
    assert obs["state"][-1] == pytest.approx(1.0)


def test_step_shifts_the_frame_stack_forward():
    class DistinctFrameBackend(SyntheticBackend):
        def _build_frame(self) -> RgbFrame:
            value = np.uint8((self.frame_index * 40) % 255)
            return np.full((240, 640, 3), value, dtype=np.uint8)

    env = FZeroXEnv(
        backend=DistinctFrameBackend(),
        config=EnvConfig(
            action_repeat=1,
            action=configured_discrete_action("steer", "gas"),
            observation=ObservationConfig(frame_stack=4),
        ),
    )

    obs_before, _ = env.reset(seed=9)
    obs_after, _, _, _, _ = env.step(np.array([2, 0], dtype=np.int64))
    obs_later, _, _, _, _ = env.step(np.array([2, 0], dtype=np.int64))
    obs_before = _image_obs(obs_before)
    obs_after = _image_obs(obs_after)
    obs_later = _image_obs(obs_later)

    assert not np.array_equal(obs_before, obs_after)
    assert np.array_equal(obs_later[:, :, 0:9], obs_after[:, :, 3:12])
