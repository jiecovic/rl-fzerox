# tests/core/envs/test_env.py
import pickle

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, MultiDiscrete

from fzerox_emulator import (
    BackendStepResult,
    ControllerState,
    FZeroXTelemetry,
    ResetState,
    StepStatus,
    StepSummary,
)
from rl_fzerox.core.config.schema import (
    ActionConfig,
    ActionMaskConfig,
    CurriculumConfig,
    CurriculumStageConfig,
    CurriculumTriggerConfig,
    EnvConfig,
    ObservationConfig,
)
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.actions import ACCELERATE_MASK, DRIFT_LEFT_MASK
from rl_fzerox.core.envs.observations import ObservationValue
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import (
    make_step_status,
    make_step_summary,
    make_telemetry,
)


class ScriptedStepBackend(SyntheticBackend):
    def __init__(
        self,
        results: list[BackendStepResult],
        *,
        reset_telemetry: FZeroXTelemetry | None = None,
    ) -> None:
        super().__init__()
        self._results = list(results)
        self._reset_telemetry = reset_telemetry

    def step_repeat_raw(
        self,
        controller_state: ControllerState,
        *,
        action_repeat: int,
        preset: str,
        frame_stack: int,
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        stuck_step_limit: int,
        wrong_way_timer_limit: int,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
    ) -> BackendStepResult:
        _ = (
            stuck_min_speed_kph,
            energy_loss_epsilon,
            max_episode_steps,
            stuck_step_limit,
            wrong_way_timer_limit,
            progress_frontier_stall_limit_frames,
            progress_frontier_epsilon,
            terminate_on_energy_depleted,
        )
        self.set_controller_state(controller_state)
        result = self._results.pop(0)
        frames_run = result.summary.frames_run
        self._capture_video_flags.extend([False] * max(frames_run - 1, 0))
        self._capture_video_flags.append(True)
        self._state.frame_index = result.summary.final_frame_index
        self._state.progress = result.summary.max_race_distance
        self._last_frame = self._build_frame()
        if result.observation.shape[2] != frame_stack * 3:
            raise AssertionError("Scripted observation stack does not match frame_stack")
        if preset != "native_crop_v3":
            raise AssertionError(f"Unexpected preset {preset!r}")
        return result

    def try_read_telemetry(self) -> FZeroXTelemetry | None:
        return self._reset_telemetry


class CameraSyncBackend(SyntheticBackend):
    def __init__(self, *, camera_setting_raw: int = 2) -> None:
        super().__init__()
        self.camera_setting_raw = camera_setting_raw

    def step_frame(self):
        if self.last_controller_state.right_stick_x > 0.5:
            self.camera_setting_raw = (self.camera_setting_raw + 1) % 4
        return super().step_frame()

    def try_read_telemetry(self) -> FZeroXTelemetry | None:
        return make_telemetry(
            game_mode_raw=1,
            game_mode_name="gp_race",
            in_race_mode=True,
            race_distance=0.0,
            camera_setting_raw=self.camera_setting_raw,
            camera_setting_name=_camera_setting_name(self.camera_setting_raw),
        )


def test_reset_returns_stacked_observation():
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=2,
            observation=ObservationConfig(frame_stack=4),
        ),
    )

    obs, info = env.reset(seed=123)
    obs = _image_obs(obs)

    assert obs.shape == (116, 164, 12)
    assert obs.dtype == np.uint8
    assert info["backend"] == "synthetic"
    assert info["seed"] == 123
    assert info["observation_shape"] == (116, 164, 12)
    assert info["observation_frame_shape"] == (116, 164, 3)
    assert info["observation_stack"] == 4
    assert np.array_equal(obs[:, :, 0:3], obs[:, :, 3:6])
    assert np.array_equal(obs[:, :, 3:6], obs[:, :, 6:9])
    assert np.array_equal(obs[:, :, 6:9], obs[:, :, 9:12])
    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [7, 3, 2, 3]


def test_reset_can_return_image_state_observation() -> None:
    backend = ScriptedStepBackend(
        [],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne", "can_boost"),
            speed_kph=750.0,
            energy=89.0,
            max_energy=178.0,
            reverse_timer=1,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            observation=ObservationConfig(mode="image_state", frame_stack=4),
        ),
    )

    obs, info = env.reset(seed=123)

    assert isinstance(obs, dict)
    assert set(obs) == {"image", "state"}
    assert obs["image"].shape == (116, 164, 12)
    assert obs["image"].dtype == np.uint8
    assert obs["state"].shape == (11,)
    assert obs["state"].dtype == np.float32
    assert obs["state"].tolist() == pytest.approx(
        [0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    )
    assert info["observation_mode"] == "image_state"
    assert info["observation_shape"] == (116, 164, 12)
    assert info["observation_state_shape"] == (11,)
    assert info["observation_state_features"] == (
        "speed_norm",
        "energy_frac",
        "reverse_active",
        "airborne",
        "can_boost",
        "boost_active",
        "left_drift_held",
        "right_drift_held",
        "left_press_age_norm",
        "right_press_age_norm",
        "recent_boost_pressure",
    )


def test_reset_info_is_pickle_safe_with_live_telemetry() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=1),
    )

    _, info = env.reset(seed=123)

    assert "telemetry" not in info
    pickle.dumps(info)


def test_reset_randomizes_game_rng_when_enabled_and_in_race() -> None:
    backend = ScriptedStepBackend([], reset_telemetry=_telemetry(race_distance=0.0))
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=1, randomize_game_rng_on_reset=True),
    )

    _, first_info = env.reset(seed=123)
    _, second_info = env.reset()

    assert first_info["rng_randomized"] is True
    assert second_info["rng_randomized"] is True
    assert len(backend.randomized_rng_seeds) == 2
    assert backend.randomized_rng_seeds[0] != backend.randomized_rng_seeds[1]
    assert first_info["rng_state"] != second_info["rng_state"]


def test_reset_can_randomize_game_rng_without_race_mode_requirement() -> None:
    backend = ScriptedStepBackend([], reset_telemetry=None)
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            randomize_game_rng_on_reset=True,
            randomize_game_rng_requires_race_mode=False,
        ),
    )

    _, info = env.reset(seed=123)

    assert info["rng_randomized"] is True
    assert len(backend.randomized_rng_seeds) == 1


def test_reset_skips_rng_randomization_outside_race() -> None:
    backend = ScriptedStepBackend(
        [],
        reset_telemetry=make_telemetry(
            game_mode_raw=0,
            game_mode_name="title",
            in_race_mode=False,
            race_distance=0.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=1, randomize_game_rng_on_reset=True),
    )

    _, info = env.reset(seed=123)

    assert info["rng_randomized"] is False
    assert info["rng_randomization_skip_reason"] == "not_in_race"
    assert backend.randomized_rng_seeds == []


def test_reset_applies_configured_camera_setting_with_button_loop() -> None:
    backend = CameraSyncBackend(camera_setting_raw=2)
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=1, camera_setting="close_behind"),
    )

    _, info = env.reset(seed=123)

    assert info["camera_setting"] == "close_behind"
    assert info["camera_setting_raw"] == 1
    assert info["camera_setting_sync"] == "changed"
    assert info["camera_setting_taps"] == 3
    assert info["frame_index"] == 6
    assert backend.last_controller_state == ControllerState()


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
    env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert backend.last_controller_state.joypad_mask == 0
    env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert backend.last_controller_state.joypad_mask == ACCELERATE_MASK

    env.reset(seed=8)
    env.step(np.array([0.0, 0.5], dtype=np.float32))

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


def test_step_updates_right_drift_hold_and_press_age_in_image_state_observation() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            action=ActionConfig(name="steer_drive_boost_drift"),
            observation=ObservationConfig(mode="image_state", frame_stack=4),
        ),
    )

    env.reset(seed=7)
    obs, _, _, _, _ = env.step(np.array([4, 1, 0, 2], dtype=np.int64))

    assert isinstance(obs, dict)
    assert obs["state"].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 / 15.0, 0.0]
    )


def test_step_shifts_the_frame_stack_forward():
    class DistinctFrameBackend(SyntheticBackend):
        def _build_frame(self) -> np.ndarray:
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
            self.render_observation_calls: list[tuple[str, int]] = []

        def render_observation(self, *, preset: str, frame_stack: int) -> np.ndarray:
            self.render_observation_calls.append((preset, frame_stack))
            return super().render_observation(preset=preset, frame_stack=frame_stack)

    backend = ObservationPresetBackend()

    env = FZeroXEnv(backend=backend, config=EnvConfig(action_repeat=1))

    obs, info = env.reset(seed=13)
    obs = _image_obs(obs)

    assert obs.shape == (116, 164, 12)
    assert info["observation_frame_shape"] == (116, 164, 3)
    assert backend.render_observation_calls == [("native_crop_v3", 4)]


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


def test_extended_action_env_exposes_four_head_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="steer_drive_boost_drift")),
    )

    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [7, 3, 2, 3]


def test_boost_action_env_exposes_three_head_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="steer_drive_boost")),
    )

    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [7, 3, 2]


def test_continuous_action_env_exposes_box_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="continuous_steer_drive")),
    )

    assert isinstance(env.action_space, Box)
    assert env.action_space.shape == (2,)
    assert env.action_masks().tolist() == []


def test_continuous_drift_action_env_exposes_box_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="continuous_steer_drive_drift")),
    )

    assert isinstance(env.action_space, Box)
    assert env.action_space.shape == (3,)
    assert env.action_masks().tolist() == []


def test_hybrid_drift_action_env_exposes_dict_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_drift")),
    )

    assert isinstance(env.action_space, Dict)
    assert isinstance(env.action_space.spaces["continuous"], Box)
    assert isinstance(env.action_space.spaces["discrete"], MultiDiscrete)
    assert env.action_space.spaces["continuous"].shape == (2,)
    assert env.action_space.spaces["discrete"].nvec.tolist() == [3]
    assert env.action_masks().tolist() == [True, True, True]


def test_hybrid_boost_drift_action_env_exposes_boost_mask_branch() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_boost_drift")),
    )

    assert isinstance(env.action_space, Dict)
    assert isinstance(env.action_space.spaces["continuous"], Box)
    assert isinstance(env.action_space.spaces["discrete"], MultiDiscrete)
    assert env.action_space.spaces["continuous"].shape == (2,)
    assert env.action_space.spaces["discrete"].nvec.tolist() == [3, 2]
    assert env.action_masks().tolist() == [True, True, True, True, True]


def test_hybrid_boost_shoulder_primitive_env_masks_future_primitives_by_default() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=ActionConfig(name="hybrid_steer_drive_boost_shoulder_primitive")
        ),
    )

    assert isinstance(env.action_space, Dict)
    assert isinstance(env.action_space.spaces["continuous"], Box)
    assert isinstance(env.action_space.spaces["discrete"], MultiDiscrete)
    assert env.action_space.spaces["continuous"].shape == (3,)
    assert env.action_space.spaces["discrete"].nvec.tolist() == [7, 2]
    assert env.action_masks().tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_env_action_masks_reflect_base_action_mask_config() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=ActionConfig(
                name="steer_drive_boost_drift",
                mask=ActionMaskConfig(shoulder=(0,)),
            )
        ),
    )

    mask = env.action_masks()

    assert mask.tolist() == (([True] * 7) + ([True] * 3) + ([True] * 2) + [True, False, False])


def test_env_action_masks_reject_base_mask_branch_missing_from_adapter() -> None:
    with pytest.raises(ValueError, match="env\\.action\\.mask.*'drive'"):
        FZeroXEnv(
            backend=SyntheticBackend(),
            config=EnvConfig(
                action=ActionConfig(
                    name="hybrid_steer_drive_boost_drift",
                    mask=ActionMaskConfig(drive=(0,)),
                )
            ),
        )


def test_env_action_masks_reject_curriculum_mask_branch_missing_from_adapter() -> None:
    with pytest.raises(
        ValueError,
        match=r"curriculum\.stages\[0\]\.action_mask.*'boost'",
    ):
        FZeroXEnv(
            backend=SyntheticBackend(),
            config=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_drift")),
            curriculum_config=CurriculumConfig(
                enabled=True,
                stages=(
                    CurriculumStageConfig(
                        name="no_boost_branch",
                        action_mask=ActionMaskConfig(boost=(0,)),
                    ),
                ),
            ),
        )


def test_env_action_masks_update_with_curriculum_stage_changes() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=ActionConfig(
                name="steer_drive_boost_drift",
                mask=ActionMaskConfig(shoulder=(0,)),
            )
        ),
        curriculum_config=CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(
                    name="basic_drive",
                    until=CurriculumTriggerConfig(race_laps_completed_mean_gte=3.0),
                    action_mask=ActionMaskConfig(shoulder=(0,)),
                ),
                CurriculumStageConfig(
                    name="full_controls",
                    action_mask=ActionMaskConfig(shoulder=(0, 1, 2)),
                ),
            ),
        ),
    )

    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 3) + ([True] * 2) + [True, False, False]
    )

    env.set_curriculum_stage(1)

    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2 + 3))


def test_env_sync_checkpoint_curriculum_stage_resets_to_default_stage() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=ActionConfig(
                name="steer_drive_boost_drift",
                mask=ActionMaskConfig(shoulder=(0,)),
            )
        ),
        curriculum_config=CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(
                    name="basic_drive",
                    until=CurriculumTriggerConfig(race_laps_completed_mean_gte=3.0),
                    action_mask=ActionMaskConfig(shoulder=(0,)),
                ),
                CurriculumStageConfig(
                    name="full_controls",
                    action_mask=ActionMaskConfig(shoulder=(0, 1, 2)),
                ),
            ),
        ),
    )

    env.set_curriculum_stage(1)
    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2 + 3))

    env.sync_checkpoint_curriculum_stage(None)

    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 3) + ([True] * 2) + [True, False, False]
    )


def test_env_action_masks_disable_boost_until_telemetry_unlocks_it() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active",)),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action=ActionConfig(name="steer_drive_boost")),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 3) + [True, False])

    env.step(np.array([3, 1, 0], dtype=np.int64))

    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2))


def test_hybrid_env_action_masks_disable_boost_until_telemetry_unlocks_it() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active",)),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_boost_drift")),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == [True, True, True, True, False]

    env.step(
        {
            "continuous": np.array([0.0, 1.0], dtype=np.float32),
            "discrete": np.array([0, 0], dtype=np.int64),
        }
    )

    assert env.action_masks().tolist() == [True, True, True, True, True]


def test_hybrid_shoulder_primitive_masks_boost_until_telemetry_unlocks_it() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active",)),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=ActionConfig(name="hybrid_steer_drive_boost_shoulder_primitive")
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
    ]

    env.step(
        {
            "continuous": np.array([0.0, 1.0, -1.0], dtype=np.float32),
            "discrete": np.array([0, 0], dtype=np.int64),
        }
    )

    assert env.action_masks().tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_env_action_masks_disable_drift_below_speed_threshold() -> None:
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
            speed_kph=300.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=ActionConfig(
                name="steer_drive_boost_drift",
                drift_unmask_min_speed_kph=500.0,
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 3) + ([True] * 2) + [True, False, False]
    )

    env.step(np.array([3, 1, 0, 0], dtype=np.int64))

    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2 + 3))


def test_env_action_masks_static_shoulder_mask_wins_over_speed_unmask() -> None:
    backend = ScriptedStepBackend(
        [],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
            speed_kph=650.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=ActionConfig(
                name="steer_drive_boost_drift",
                drift_unmask_min_speed_kph=500.0,
                mask=ActionMaskConfig(shoulder=(0,)),
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 3) + ([True] * 2) + [True, False, False]
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
            action=ActionConfig(name="steer_drive_boost"),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 3) + [True, False])

    env.step(np.array([3, 1, 0], dtype=np.int64))

    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2))


def test_env_action_masks_disable_boost_while_boost_is_active() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "can_boost"),
                    boost_timer=0,
                ),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
            boost_timer=12,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action=ActionConfig(name="steer_drive_boost")),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 3) + [True, False])

    env.step(np.array([3, 1, 0], dtype=np.int64))

    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2))


def test_env_action_masks_intersect_curriculum_and_boost_unlock_rules() -> None:
    env = FZeroXEnv(
        backend=ScriptedStepBackend(
            [],
            reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active",)),
        ),
        config=EnvConfig(
            action=ActionConfig(
                name="steer_drive_boost_drift",
                mask=ActionMaskConfig(shoulder=(0,)),
            )
        ),
        curriculum_config=CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(
                    name="basic_drive",
                    until=CurriculumTriggerConfig(race_laps_completed_mean_gte=3.0),
                    action_mask=ActionMaskConfig(shoulder=(0,)),
                ),
            ),
        ),
    )

    env.reset(seed=2)

    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 3) + [True, False] + [True, False, False]
    )


def test_env_keeps_shoulder_input_latched_for_minimum_internal_frames() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=2, frames_run=2),
                status=make_step_status(step_count=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=20.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(max_race_distance=20.0, final_frame_index=4, frames_run=2),
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
            action_repeat=2,
            action=ActionConfig(name="steer_drive_boost_drift"),
        ),
    )

    env.reset(seed=3)
    env.step(np.array([3, 0, 0, 1], dtype=np.int64))

    assert env.action_masks().tolist()[-3:] == [False, True, False]

    env.step(np.array([3, 0, 0, 0], dtype=np.int64))

    assert backend.last_controller_state.joypad_mask == DRIFT_LEFT_MASK
    assert env.action_masks().tolist()[-3:] == [False, True, False]


def test_env_unlocks_shoulder_branch_after_minimum_hold_window() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=5, frames_run=5),
                status=make_step_status(step_count=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=20.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(max_race_distance=20.0, final_frame_index=10, frames_run=5),
                status=make_step_status(step_count=2),
            ),
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=30.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(max_race_distance=30.0, final_frame_index=15, frames_run=5),
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
            action_repeat=5,
            action=ActionConfig(name="steer_drive_boost_drift"),
        ),
    )

    env.reset(seed=4)
    assert env.action_masks().tolist()[-3:] == [True, True, True]

    env.step(np.array([3, 0, 0, 1], dtype=np.int64))
    assert env.action_masks().tolist()[-3:] == [False, True, False]

    env.step(np.array([3, 0, 0, 1], dtype=np.int64))
    assert env.action_masks().tolist()[-3:] == [False, True, False]

    env.step(np.array([3, 0, 0, 1], dtype=np.int64))
    assert env.action_masks().tolist()[-3:] == [True, True, True]


def test_reset_can_boot_into_the_first_race_path():
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=2, reset_to_race=True),
    )

    obs, info = env.reset(seed=5)
    obs = _image_obs(obs)

    assert obs.shape == (116, 164, 12)
    assert info["seed"] == 5
    assert info["reset_mode"] == "boot_to_race"
    assert info["boot_state"] == "gp_race"
    assert backend.frame_index == 1_592


def test_reset_skips_bootstrap_when_a_custom_baseline_is_active():
    class BaselineBackend(SyntheticBackend):
        def reset(self) -> ResetState:
            reset_state = super().reset()
            info = dict(reset_state.info)
            info["baseline_kind"] = "custom"
            return ResetState(frame=reset_state.frame, info=info)

    backend = BaselineBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=2, reset_to_race=True),
    )

    _, info = env.reset(seed=11)

    assert info["seed"] == 11
    assert "boot_state" not in info
    assert backend.frame_index == 0


def test_reset_can_continue_to_next_race_after_terminal_episode(monkeypatch) -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=2, reset_to_race=True),
    )

    env.reset(seed=1)
    env._engine._episode_done = True

    def fake_continue_to_next_race(_backend):
        return backend.render(), {
            "reset_mode": "continue_to_next_race",
            "boot_state": "gp_race",
            "frame_index": 4242,
        }

    monkeypatch.setattr(
        "rl_fzerox.core.envs.engine.runtime.continue_to_next_race",
        fake_continue_to_next_race,
    )

    _, info = env.reset(seed=2)

    assert info["seed"] == 2
    assert info["reset_mode"] == "continue_to_next_race"
    assert info["boot_state"] == "gp_race"


def test_reset_surfaces_continue_to_next_race_fallback(monkeypatch) -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=2, reset_to_race=True),
    )

    env.reset(seed=1)
    env._engine._episode_done = True

    monkeypatch.setattr(
        "rl_fzerox.core.envs.engine.runtime.continue_to_next_race",
        lambda _backend: (_ for _ in ()).throw(RuntimeError("continue failed")),
    )

    _, info = env.reset(seed=2)

    assert info["seed"] == 2
    assert info["reset_mode"] == "boot_to_race"
    assert info["reset_fallback"] == "continue_to_next_race_failed"
    assert info["continue_to_next_race_error"] == "continue failed"


def test_step_truncates_on_timeout() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            max_episode_steps=2,
            stuck_step_limit=10,
            action=ActionConfig(name="steer_drive"),
        ),
    )

    env.reset(seed=3)
    _, _, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))
    assert not terminated
    assert not truncated

    _, _, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert not terminated
    assert truncated
    assert info["truncation_reason"] == "timeout"
    assert info["episode_step"] == 2


def test_step_truncates_when_speed_is_stuck() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=0.0, speed_kph=40.0),
                summary=_step_summary(
                    max_race_distance=0.0,
                    low_speed_frames=1,
                    consecutive_low_speed_frames=1,
                    final_frame_index=1,
                ),
                status=make_step_status(step_count=1, stalled_steps=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=0.0, speed_kph=40.0),
                summary=_step_summary(
                    max_race_distance=0.0,
                    low_speed_frames=1,
                    consecutive_low_speed_frames=1,
                    final_frame_index=2,
                ),
                status=make_step_status(
                    step_count=2,
                    stalled_steps=2,
                    truncation_reason="stuck",
                ),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            max_episode_steps=100,
            stuck_step_limit=2,
            stuck_min_speed_kph=50.0,
            action=ActionConfig(name="steer_drive"),
        ),
    )

    env.reset(seed=4)
    _, _, terminated, truncated, _ = env.step(np.array([2, 0], dtype=np.int64))
    assert not terminated
    assert not truncated

    _, reward, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert not terminated
    assert truncated
    assert reward == pytest.approx(-300.99)
    assert info["truncation_reason"] == "stuck"
    assert info["stalled_steps"] == 2
    assert info["step_reward"] == pytest.approx(-300.99)
    reward_breakdown = info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert reward_breakdown["time"] == -0.005
    assert reward_breakdown["low_speed_time"] == -0.005
    assert reward_breakdown["stuck_truncation"] == pytest.approx(-300.98)


def test_step_truncates_when_driving_the_wrong_way() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=-3.0, reverse_timer=80),
                summary=_step_summary(
                    max_race_distance=0.0,
                    reverse_active_frames=1,
                    final_frame_index=1,
                ),
                status=make_step_status(step_count=1, reverse_timer=80),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=-6.5, reverse_timer=100),
                summary=_step_summary(
                    max_race_distance=0.0,
                    reverse_active_frames=1,
                    final_frame_index=2,
                ),
                status=make_step_status(
                    step_count=2,
                    reverse_timer=100,
                    truncation_reason="wrong_way",
                ),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            max_episode_steps=100,
            stuck_step_limit=10,
            wrong_way_timer_limit=100,
            action=ActionConfig(name="steer_drive"),
        ),
    )

    env.reset(seed=7)
    _, reward, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))
    assert not terminated
    assert not truncated
    assert reward == pytest.approx(-0.016)
    assert info["reverse_timer"] == 80

    _, reward, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert not terminated
    assert truncated
    assert reward == pytest.approx(-320.997)
    assert info["truncation_reason"] == "wrong_way"
    assert info["reverse_timer"] == 100
    reward_breakdown = info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert reward_breakdown["reverse_time"] == -0.005
    assert reward_breakdown["bootstrap_regress"] == pytest.approx(-0.007)
    assert reward_breakdown["wrong_way_truncation"] == pytest.approx(-320.98)


def test_step_truncates_when_progress_frontier_stalls() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=1_000.0, speed_kph=400.0),
                summary=_step_summary(
                    max_race_distance=1_000.0,
                    final_frame_index=1,
                ),
                status=make_step_status(
                    step_count=1,
                    progress_frontier_stalled_frames=0,
                ),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=1_000.0, speed_kph=300.0),
                summary=_step_summary(
                    max_race_distance=1_000.0,
                    final_frame_index=6,
                ),
                status=make_step_status(
                    step_count=6,
                    progress_frontier_stalled_frames=5,
                    truncation_reason="progress_stalled",
                ),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            max_episode_steps=100,
            progress_frontier_stall_limit_frames=5,
            progress_frontier_epsilon=100.0,
            action=ActionConfig(name="steer_drive"),
        ),
    )

    env.reset(seed=11)
    _, _, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))
    assert not terminated
    assert not truncated
    assert info["progress_frontier_stalled_frames"] == 0

    _, reward, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert not terminated
    assert truncated
    assert reward == pytest.approx(-300.945)
    assert info["truncation_reason"] == "progress_stalled"
    assert info["progress_frontier_stalled_frames"] == 5
    reward_breakdown = info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert reward_breakdown["progress_stalled_truncation"] == pytest.approx(-300.94)


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


def _telemetry(
    *,
    race_distance: float,
    state_labels: tuple[str, ...] = ("active",),
    speed_kph: float = 100.0,
    energy: float = 178.0,
    max_energy: float = 178.0,
    boost_timer: int = 0,
    reverse_timer: int = 0,
    lap: int = 1,
    laps_completed: int = 0,
    camera_setting_raw: int = 2,
    camera_setting_name: str = "regular",
) -> FZeroXTelemetry:
    return make_telemetry(
        race_distance=race_distance,
        state_labels=state_labels,
        speed_kph=speed_kph,
        energy=energy,
        max_energy=max_energy,
        boost_timer=boost_timer,
        reverse_timer=reverse_timer,
        lap=lap,
        laps_completed=laps_completed,
        camera_setting_raw=camera_setting_raw,
        camera_setting_name=camera_setting_name,
    )


def _camera_setting_name(camera_setting_raw: int) -> str:
    if camera_setting_raw == 0:
        return "overhead"
    if camera_setting_raw == 1:
        return "close_behind"
    if camera_setting_raw == 2:
        return "regular"
    if camera_setting_raw == 3:
        return "wide"
    return "unknown"


def _image_obs(observation: ObservationValue) -> np.ndarray:
    assert isinstance(observation, np.ndarray)
    return observation


def _step_summary(
    *,
    max_race_distance: float,
    frames_run: int = 1,
    reverse_active_frames: int = 0,
    low_speed_frames: int = 0,
    consecutive_low_speed_frames: int = 0,
    entered_state_labels: tuple[str, ...] = (),
    final_frame_index: int = 1,
) -> StepSummary:
    return make_step_summary(
        frames_run=frames_run,
        max_race_distance=max_race_distance,
        reverse_active_frames=reverse_active_frames,
        low_speed_frames=low_speed_frames,
        energy_loss_total=0.0,
        consecutive_low_speed_frames=consecutive_low_speed_frames,
        entered_state_labels=entered_state_labels,
        final_frame_index=final_frame_index,
    )


def _backend_step_result(
    *,
    telemetry: FZeroXTelemetry,
    summary: StepSummary,
    status: StepStatus | None = None,
) -> BackendStepResult:
    value = np.uint8(summary.final_frame_index % 255)
    observation = np.full((116, 164, 12), value, dtype=np.uint8)
    return BackendStepResult(
        observation=observation,
        summary=summary,
        status=(
            status
            if status is not None
            else make_step_status(
                step_count=summary.final_frame_index,
                termination_reason=telemetry.player.terminal_reason,
            )
        ),
        telemetry=telemetry,
    )
