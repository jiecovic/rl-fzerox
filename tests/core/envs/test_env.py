# tests/core/envs/test_env.py
import pickle

import numpy as np
import pytest
from gymnasium.spaces import Box

from fzerox_emulator import (
    ControllerState,
)
from fzerox_emulator.arrays import ObservationFrame, RgbFrame
from rl_fzerox.core.config.schema import (
    ActionConfig,
    EnvConfig,
    ObservationConfig,
    RewardConfig,
)
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.actions import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
)
from rl_fzerox.core.envs.observations import (
    LEAN_DOUBLE_TAP_WINDOW_FRAMES,
    ObservationStackMode,
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
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import make_step_status


def test_step_advances_backend_by_action_repeat():
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=3, action=ActionConfig(name="steer_drive")),
    )

    env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(np.array([3, 1], dtype=np.int64))
    obs = _image_obs(obs)

    assert obs.shape == (116, 164, 12)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert backend.frame_index == 3
    assert backend.capture_video_flags == [False, False, True]
    assert info["repeat_index"] == 2
    assert backend.last_controller_state == ControllerState(
        joypad_mask=ACCELERATE_MASK,
        left_stick_x=0.0,
    )


def test_watch_step_captures_each_repeated_display_frame():
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=3, action=ActionConfig(name="steer_drive")),
    )

    env.reset(seed=7)
    watch_step = env.step_watch(np.array([3, 1], dtype=np.int64))

    assert backend.frame_index == 3
    assert backend.capture_video_flags == [True, True, True]
    assert len(watch_step.display_frames) == 3
    assert watch_step.display_frames[0].shape == (444, 592, 3)
    assert _image_obs(watch_step.observation).shape == (116, 164, 12)
    assert watch_step.info["repeat_index"] == 2


def test_reset_resets_continuous_drive_pwm_phase() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=ActionConfig(
                name="continuous_steer_drive",
                continuous_drive_mode="pwm",
                continuous_drive_deadzone=0.0,
            ),
        ),
    )

    env.reset(seed=7)
    env.step(np.array([0.0, -0.5], dtype=np.float32))
    assert backend.last_controller_state.joypad_mask == 0
    env.step(np.array([0.0, -0.5], dtype=np.float32))
    assert backend.last_controller_state.joypad_mask == ACCELERATE_MASK

    env.reset(seed=8)
    env.step(np.array([0.0, -0.5], dtype=np.float32))

    assert backend.last_controller_state.joypad_mask == 0


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
            action=ActionConfig(name="steer_drive"),
            observation=ObservationConfig(mode="image_state", frame_stack=4),
        ),
    )

    env.reset(seed=7)
    obs, _, _, _, info = env.step(np.array([2, 0], dtype=np.int64))

    assert isinstance(obs, dict)
    assert set(obs) == {"image", "state"}
    assert obs["image"].shape == (116, 164, 12)
    assert obs["state"].tolist() == pytest.approx(
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    )
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
            action=ActionConfig(name="steer_drive"),
        ),
    )

    env.reset(seed=7)
    _, _, _, _, info = env.step(np.array([2, 0], dtype=np.int64))

    assert info["energy_loss_total"] == 3.5
    assert info["damage_taken_frames"] == 1


def test_step_updates_recent_boost_pressure_in_image_state_observation() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=120,
            action=ActionConfig(name="steer_drive_boost"),
            observation=ObservationConfig(mode="image_state", frame_stack=4),
        ),
    )

    env.reset(seed=7)
    obs, _, _, _, _ = env.step(np.array([3, 0, 1], dtype=np.int64))

    assert isinstance(obs, dict)
    assert obs["state"].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    )


def test_step_updates_steer_history_state_profile() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=30,
            action=ActionConfig(name="steer_gas_air_brake_boost_lean", steer_buckets=3),
            observation=ObservationConfig(
                mode="image_state",
                state_profile="steer_history",
                frame_stack=4,
            ),
        ),
    )

    env.reset(seed=7)
    obs, _, _, _, info = env.step(np.array([0, 1, 1, 0, 0], dtype=np.int64))

    assert isinstance(obs, dict)
    assert obs["state"].shape == (14,)
    assert info["observation_state_profile"] == "steer_history"
    assert info["observation_state_shape"] == (14,)
    raw_feature_names = info["observation_state_features"]
    assert isinstance(raw_feature_names, tuple)
    feature_names = tuple(str(name) for name in raw_feature_names)
    values = {name: float(value) for name, value in zip(feature_names, obs["state"], strict=True)}
    assert values["steer_left_held"] == 1.0
    assert values["steer_right_held"] == 0.0
    assert values["recent_steer_pressure"] == pytest.approx(-1.0)


def test_step_updates_race_core_with_action_history() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=ActionConfig(name="hybrid_steer_gas_boost_lean"),
            observation=ObservationConfig(
                mode="image_state",
                state_profile="race_core",
                action_history_len=3,
                frame_stack=4,
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
    assert obs["state"].shape == (18,)
    assert info["observation_state_profile"] == "race_core"
    assert info["observation_action_history_len"] == 3
    assert info["observation_action_history_controls"] == ("steer", "gas", "boost", "lean")
    assert info["observation_state_shape"] == (18,)
    raw_feature_names = info["observation_state_features"]
    assert isinstance(raw_feature_names, tuple)
    feature_names = tuple(str(name) for name in raw_feature_names)
    values = {name: float(value) for name, value in zip(feature_names, obs["state"], strict=True)}
    assert "recent_boost_pressure" not in values
    assert "recent_steer_pressure" not in values
    assert "prev_air_brake_1" not in values
    assert values["prev_steer_1"] == pytest.approx(-0.5)
    assert values["prev_gas_1"] == 1.0
    assert values["prev_boost_1"] == 1.0
    assert values["prev_lean_1"] == -1.0
    assert values["prev_steer_2"] == 0.0
    assert values["prev_steer_3"] == 0.0


def test_step_updates_right_lean_hold_and_press_age_in_image_state_observation() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=ActionConfig(name="steer_drive_boost_lean"),
            observation=ObservationConfig(mode="image_state", frame_stack=4),
        ),
    )

    env.reset(seed=7)
    obs, _, _, _, _ = env.step(np.array([4, 1, 0, 2], dtype=np.int64))

    assert isinstance(obs, dict)
    assert obs["state"].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0 / LEAN_DOUBLE_TAP_WINDOW_FRAMES,
            0.0,
        ]
    )


def test_step_shifts_the_frame_stack_forward():
    class DistinctFrameBackend(SyntheticBackend):
        def _build_frame(self) -> RgbFrame:
            value = np.uint8((self.frame_index * 40) % 255)
            return np.full((240, 640, 3), value, dtype=np.uint8)

    env = FZeroXEnv(
        backend=DistinctFrameBackend(),
        config=EnvConfig(
            action_repeat=1,
            action=ActionConfig(name="steer_drive"),
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


def test_env_reset_passes_preset_to_render_observation() -> None:
    class ObservationPresetBackend(SyntheticBackend):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.render_observation_calls: list[tuple[str, int, str]] = []

        def render_observation(
            self,
            *,
            preset: str,
            frame_stack: int,
            stack_mode: ObservationStackMode = "rgb",
        ) -> ObservationFrame:
            self.render_observation_calls.append((preset, frame_stack, stack_mode))
            return super().render_observation(
                preset=preset,
                frame_stack=frame_stack,
                stack_mode=stack_mode,
            )

    backend = ObservationPresetBackend()

    env = FZeroXEnv(backend=backend, config=EnvConfig(action_repeat=1))

    obs, info = env.reset(seed=13)
    obs = _image_obs(obs)

    assert obs.shape == (116, 164, 12)
    assert info["observation_frame_shape"] == (116, 164, 3)
    assert backend.render_observation_calls == [("crop_116x164", 4, "rgb")]


def test_env_reset_uses_rgb_gray_stack_shape() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            observation=ObservationConfig(
                preset="crop_98x130",
                frame_stack=4,
                stack_mode="rgb_gray",
            ),
        ),
    )

    obs, info = env.reset(seed=13)
    obs = _image_obs(obs)

    assert obs.shape == (98, 130, 6)
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (98, 130, 6)
    assert info["observation_stack"] == 4
    assert info["observation_stack_mode"] == "rgb_gray"


def test_env_render_uses_cropped_aspect_corrected_display_size() -> None:
    backend = SyntheticBackend(width=640, height=240)
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(observation=ObservationConfig(frame_stack=4)),
    )

    env.reset(seed=1)
    frame = env.render()

    assert frame.shape == (444, 592, 3)


def test_step_control_applies_manual_controller_state() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=2, action=ActionConfig(name="steer_drive")),
    )

    env.reset(seed=21)
    control_state = ControllerState(joypad_mask=5, left_stick_x=-1.0)
    env.step_control(control_state)

    assert backend.last_controller_state == control_state


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
            action=ActionConfig(
                name="hybrid_steer_drive_boost_lean_primitive",
                continuous_drive_mode="pwm",
                continuous_drive_deadzone=0.0,
                continuous_air_brake_mode="disable_on_ground",
            ),
        ),
        reward_config=RewardConfig(time_penalty_per_frame=0.0),
    )
    air_brake_state = ControllerState(joypad_mask=AIR_BRAKE_MASK)

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
    assert backend.last_controller_state == ControllerState()
    assert reward == 0.0
    assert "reward_breakdown" not in info

    _, reward, _, _, info = env.step_control(air_brake_state)
    assert backend.last_controller_state == air_brake_state
    assert reward == 0.0
    assert "reward_breakdown" not in info


def test_step_forces_accelerate_when_drive_mode_always_accelerate() -> None:
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
            action=ActionConfig(
                name="hybrid_steer_drive_boost_lean_primitive",
                continuous_drive_mode="always_accelerate",
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

    assert backend.last_controller_state.joypad_mask & ACCELERATE_MASK
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
            action=ActionConfig(
                name="continuous_steer_drive",
                continuous_drive_mode="pwm",
                continuous_drive_deadzone=0.0,
            ),
            observation=ObservationConfig(
                mode="image_state",
                state_profile="race_core",
                action_history_len=1,
            ),
        ),
        reward_config=RewardConfig(time_penalty_per_frame=0.0),
    )

    env.reset(seed=21)
    obs, reward, _, _, info = env.step(np.array([0.0, -0.5], dtype=np.float32))

    assert env.last_gas_level == pytest.approx(0.5)
    assert env.last_requested_control_state == ControllerState()
    assert backend.last_controller_state == ControllerState()
    assert isinstance(obs, dict)
    raw_feature_names = info["observation_state_features"]
    assert isinstance(raw_feature_names, tuple)
    values = {
        name: float(value) for name, value in zip(raw_feature_names, obs["state"], strict=True)
    }
    assert values["prev_gas_1"] == pytest.approx(0.5)
    assert reward == 0.0
    assert "reward_breakdown" not in info


def test_terminal_step_exposes_monitor_info_keys() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=42.0, state_labels=("finished",)),
                summary=_step_summary(
                    max_race_distance=42.0,
                    entered_state_labels=("finished",),
                    final_frame_index=1,
                ),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=1, action=ActionConfig(name="steer_drive")),
    )

    env.reset(seed=5)
    _, _, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert terminated
    assert not truncated
    assert info["termination_reason"] == "finished"
    assert "truncation_reason" in info
    assert info["truncation_reason"] is None
    assert isinstance(info["episode_return"], float)


def test_terminal_step_returns_an_observation_at_step_boundary() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=42.0, state_labels=("finished",)),
                summary=_step_summary(
                    frames_run=1,
                    max_race_distance=42.0,
                    entered_state_labels=("finished",),
                    final_frame_index=1,
                ),
                status=make_step_status(
                    step_count=1,
                    termination_reason="finished",
                ),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=3, action=ActionConfig(name="steer_drive")),
    )

    env.reset(seed=6)
    obs, _, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))
    obs = _image_obs(obs)

    assert obs.shape == (116, 164, 12)
    assert terminated
    assert not truncated
    assert info["repeat_index"] == 0
    assert info["termination_reason"] == "finished"


def test_step_info_is_pickle_safe_with_native_telemetry() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=42.0, lap=1, laps_completed=1),
                summary=_step_summary(
                    max_race_distance=42.0,
                    final_frame_index=1,
                ),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=1, action=ActionConfig(name="steer_drive")),
    )

    env.reset(seed=8)
    _, _, _, _, info = env.step(np.array([2, 0], dtype=np.int64))

    assert "telemetry" not in info
    assert info["race_distance"] == pytest.approx(42.0)
    assert info["lap"] == 1
    assert info["laps_completed"] == 0
    assert info["race_laps_completed"] == 0
    assert info["raw_laps_completed"] == 1
    pickle.dumps(info)
