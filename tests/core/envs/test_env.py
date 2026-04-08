# tests/core/envs/test_env.py
import numpy as np
import pytest
from gymnasium.spaces import MultiDiscrete

from fzerox_emulator import (
    BackendStepResult,
    ControllerState,
    FZeroXTelemetry,
    ResetState,
    StepSummary,
)
from rl_fzerox.core.config.schema import ActionConfig, EnvConfig, ObservationConfig
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.actions import THROTTLE_MASK
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import make_step_summary, make_telemetry


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
        reverse_progress_epsilon: float,
        energy_loss_epsilon: float,
        wrong_way_progress_epsilon: float,
    ) -> BackendStepResult:
        _ = (
            stuck_min_speed_kph,
            reverse_progress_epsilon,
            energy_loss_epsilon,
            wrong_way_progress_epsilon,
        )
        self.set_controller_state(controller_state)
        self._capture_video_flags.extend([False] * max(action_repeat - 1, 0))
        self._capture_video_flags.append(True)
        result = self._results.pop(0)
        self._state.frame_index = result.summary.final_frame_index
        self._state.progress = result.summary.max_race_distance
        self._last_frame = self._build_frame()
        if result.observation.shape[2] != frame_stack * 3:
            raise AssertionError("Scripted observation stack does not match frame_stack")
        if preset != "native_crop_v1":
            raise AssertionError(f"Unexpected preset {preset!r}")
        return result

    def try_read_telemetry(self) -> FZeroXTelemetry | None:
        return self._reset_telemetry


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

    assert obs.shape == (78, 222, 12)
    assert obs.dtype == np.uint8
    assert info["backend"] == "synthetic"
    assert info["seed"] == 123
    assert info["observation_shape"] == (78, 222, 12)
    assert info["observation_frame_shape"] == (78, 222, 3)
    assert info["observation_stack"] == 4
    assert np.array_equal(obs[:, :, 0:3], obs[:, :, 3:6])
    assert np.array_equal(obs[:, :, 3:6], obs[:, :, 6:9])
    assert np.array_equal(obs[:, :, 6:9], obs[:, :, 9:12])
    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [7, 2]


def test_step_advances_backend_by_action_repeat():
    backend = SyntheticBackend()
    env = FZeroXEnv(backend=backend, config=EnvConfig(action_repeat=3))

    env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(np.array([3, 1], dtype=np.int64))

    assert obs.shape == (78, 222, 12)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert backend.frame_index == 3
    assert backend.capture_video_flags == [False, False, True]
    assert info["repeat_index"] == 2
    assert backend.last_controller_state == ControllerState(
        joypad_mask=THROTTLE_MASK,
        left_stick_x=0.0,
    )


def test_step_shifts_the_frame_stack_forward():
    class DistinctFrameBackend(SyntheticBackend):
        def _build_frame(self) -> np.ndarray:
            value = np.uint8((self.frame_index * 40) % 255)
            return np.full((240, 640, 3), value, dtype=np.uint8)

    env = FZeroXEnv(
        backend=DistinctFrameBackend(),
        config=EnvConfig(action_repeat=1, observation=ObservationConfig(frame_stack=4)),
    )

    obs_before, _ = env.reset(seed=9)
    obs_after, _, _, _, _ = env.step(np.array([2, 0], dtype=np.int64))
    obs_later, _, _, _, _ = env.step(np.array([2, 0], dtype=np.int64))

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

    assert obs.shape == (78, 222, 12)
    assert info["observation_frame_shape"] == (78, 222, 3)
    assert backend.render_observation_calls == [("native_crop_v1", 4)]


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
    env = FZeroXEnv(backend=backend, config=EnvConfig(action_repeat=2))

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
    assert env.action_space.nvec.tolist() == [7, 2, 2, 3]


def test_boost_action_env_exposes_three_head_action_space() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(action=ActionConfig(name="steer_drive_boost")),
    )

    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [7, 2, 2]


def test_reset_can_boot_into_the_first_race_path():
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=2, reset_to_race=True),
    )

    obs, info = env.reset(seed=5)

    assert obs.shape == (78, 222, 12)
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
        "rl_fzerox.core.envs.engine.continue_to_next_race",
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
        "rl_fzerox.core.envs.engine.continue_to_next_race",
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
                    consecutive_low_speed_frames=1,
                    final_frame_index=1,
                ),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=0.0, speed_kph=40.0),
                summary=_step_summary(
                    max_race_distance=0.0,
                    consecutive_low_speed_frames=1,
                    final_frame_index=2,
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
        ),
    )

    env.reset(seed=4)
    _, _, terminated, truncated, _ = env.step(np.array([2, 0], dtype=np.int64))
    assert not terminated
    assert not truncated

    _, reward, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert not terminated
    assert truncated
    assert reward == -100.01
    assert info["truncation_reason"] == "stuck"
    assert info["stalled_steps"] == 2
    assert info["step_reward"] == -100.01
    reward_breakdown = info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert reward_breakdown["time"] == -0.01
    assert reward_breakdown["stuck_truncation"] == -100.0


def test_step_truncates_when_driving_the_wrong_way() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=-3.0),
                summary=_step_summary(
                    max_race_distance=0.0,
                    reverse_progress_total=3.0,
                    consecutive_reverse_frames=1,
                    final_frame_index=1,
                ),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=-6.5),
                summary=_step_summary(
                    max_race_distance=0.0,
                    reverse_progress_total=3.5,
                    consecutive_reverse_frames=1,
                    final_frame_index=2,
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
            wrong_way_step_limit=2,
            wrong_way_progress_epsilon=2.0,
        ),
    )

    env.reset(seed=7)
    _, reward, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))
    assert not terminated
    assert not truncated
    assert reward == pytest.approx(-0.013)
    assert info["reverse_steps"] == 1

    _, reward, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert not terminated
    assert truncated
    assert reward == pytest.approx(-120.0135)
    assert info["truncation_reason"] == "wrong_way"
    assert info["reverse_steps"] == 2
    reward_breakdown = info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert reward_breakdown["wrong_way_truncation"] == -120.0


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
        config=EnvConfig(action_repeat=1),
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
                    frames_run=3,
                    max_race_distance=42.0,
                    entered_state_labels=("finished",),
                    final_frame_index=3,
                ),
            )
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=3),
    )

    env.reset(seed=6)
    obs, _, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert obs.shape == (78, 222, 12)
    assert terminated
    assert not truncated
    assert info["repeat_index"] == 2
    assert info["termination_reason"] == "finished"


def _telemetry(
    *,
    race_distance: float,
    state_labels: tuple[str, ...] = ("active",),
    speed_kph: float = 100.0,
) -> FZeroXTelemetry:
    return make_telemetry(
        race_distance=race_distance,
        state_labels=state_labels,
        speed_kph=speed_kph,
    )


def _step_summary(
    *,
    max_race_distance: float,
    frames_run: int = 1,
    reverse_progress_total: float = 0.0,
    consecutive_reverse_frames: int = 0,
    consecutive_low_speed_frames: int = 0,
    entered_state_labels: tuple[str, ...] = (),
    final_frame_index: int = 1,
) -> StepSummary:
    return make_step_summary(
        frames_run=frames_run,
        max_race_distance=max_race_distance,
        reverse_progress_total=reverse_progress_total,
        consecutive_reverse_frames=consecutive_reverse_frames,
        energy_loss_total=0.0,
        consecutive_low_speed_frames=consecutive_low_speed_frames,
        entered_state_labels=entered_state_labels,
        final_frame_index=final_frame_index,
    )


def _backend_step_result(
    *,
    telemetry: FZeroXTelemetry,
    summary: StepSummary,
) -> BackendStepResult:
    value = np.uint8(summary.final_frame_index % 255)
    observation = np.full((78, 222, 12), value, dtype=np.uint8)
    return BackendStepResult(
        observation=observation,
        summary=summary,
        telemetry=telemetry,
    )
