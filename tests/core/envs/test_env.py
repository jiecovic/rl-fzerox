# tests/core/envs/test_env.py
import numpy as np
from gymnasium.spaces import MultiDiscrete

from rl_fzerox.core.config.schema import EnvConfig, ObservationConfig
from rl_fzerox.core.emulator.base import ResetState
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.actions import THROTTLE_MASK
from rl_fzerox.core.game import FZeroXTelemetry, PlayerTelemetry
from tests.support.fakes import SyntheticBackend


def test_reset_returns_stacked_observation():
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=2,
            observation=ObservationConfig(width=160, height=120, frame_stack=4),
        ),
    )

    obs, info = env.reset(seed=123)

    assert obs.shape == (120, 160, 12)
    assert obs.dtype == np.uint8
    assert info["backend"] == "synthetic"
    assert info["seed"] == 123
    assert info["observation_shape"] == (120, 160, 12)
    assert info["observation_frame_shape"] == (120, 160, 3)
    assert info["observation_stack"] == 4
    assert np.array_equal(obs[:, :, 0:3], obs[:, :, 3:6])
    assert np.array_equal(obs[:, :, 3:6], obs[:, :, 6:9])
    assert np.array_equal(obs[:, :, 6:9], obs[:, :, 9:12])
    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [5, 2]


def test_step_advances_backend_by_action_repeat():
    backend = SyntheticBackend()
    env = FZeroXEnv(backend=backend, config=EnvConfig(action_repeat=3))

    env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(np.array([3, 1], dtype=np.int64))

    assert obs.shape == (120, 160, 12)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert backend.frame_index == 3
    assert backend.capture_video_flags == [False, False, True]
    assert info["repeat_index"] == 2
    assert backend.last_controller_state == ControllerState(
        joypad_mask=THROTTLE_MASK,
        left_stick_x=0.5,
    )


def test_step_shifts_the_frame_stack_forward():
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action_repeat=1,
            observation=ObservationConfig(width=160, height=120, frame_stack=4),
        ),
    )

    obs_before, _ = env.reset(seed=9)
    obs_after, _, _, _, _ = env.step(np.array([2, 0], dtype=np.int64))
    obs_later, _, _, _, _ = env.step(np.array([2, 0], dtype=np.int64))

    assert not np.array_equal(obs_before, obs_after)
    assert np.array_equal(obs_later[:, :, 0:9], obs_after[:, :, 3:12])


def test_step_control_applies_manual_controller_state() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(backend=backend, config=EnvConfig(action_repeat=2))

    env.reset(seed=21)
    control_state = ControllerState(joypad_mask=5, left_stick_x=-1.0)
    env.step_control(control_state)

    assert backend.last_controller_state == control_state


def test_reset_can_boot_into_the_first_race_path():
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=2, reset_to_race=True),
    )

    obs, info = env.reset(seed=5)

    assert obs.shape == (120, 160, 12)
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


def test_step_truncates_on_timeout(monkeypatch) -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            max_episode_steps=2,
            stuck_grace_steps=10,
            stuck_step_limit=10,
        ),
    )
    monkeypatch.setattr(
        "rl_fzerox.core.envs.engine._read_live_telemetry",
        lambda _backend: _telemetry(race_distance=float(backend.frame_index * 10)),
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


def test_step_truncates_when_progress_is_stuck(monkeypatch) -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            max_episode_steps=100,
            stuck_grace_steps=1,
            stuck_step_limit=2,
            stuck_progress_epsilon=5.0,
        ),
    )
    monkeypatch.setattr(
        "rl_fzerox.core.envs.engine._read_live_telemetry",
        lambda _backend: _telemetry(race_distance=0.0),
    )

    env.reset(seed=4)
    _, _, terminated, truncated, _ = env.step(np.array([2, 0], dtype=np.int64))
    assert not terminated
    assert not truncated

    _, reward, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert not terminated
    assert truncated
    assert reward == -5.0
    assert info["truncation_reason"] == "stuck"
    assert info["stalled_steps"] == 2
    assert info["step_reward"] == -5.0
    reward_breakdown = info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert reward_breakdown["stuck_truncation"] == -5.0


def test_step_truncates_when_driving_the_wrong_way(monkeypatch) -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            max_episode_steps=100,
            stuck_grace_steps=10,
            stuck_step_limit=10,
            wrong_way_step_limit=2,
            wrong_way_progress_epsilon=2.0,
        ),
    )
    distances = iter((0.0, -3.0, -6.5))
    monkeypatch.setattr(
        "rl_fzerox.core.envs.engine._read_live_telemetry",
        lambda _backend: _telemetry(race_distance=next(distances)),
    )

    env.reset(seed=7)
    _, reward, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))
    assert not terminated
    assert not truncated
    assert reward == -0.003
    assert info["reverse_steps"] == 1

    _, reward, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert not terminated
    assert truncated
    assert reward == -10.0035
    assert info["truncation_reason"] == "wrong_way"
    assert info["reverse_steps"] == 2
    reward_breakdown = info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert reward_breakdown["wrong_way_truncation"] == -10.0


def test_terminal_step_exposes_monitor_info_keys(monkeypatch) -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=1),
    )
    monkeypatch.setattr(
        "rl_fzerox.core.envs.engine._read_live_telemetry",
        lambda _backend: _telemetry(race_distance=42.0, state_labels=("finished",)),
    )

    env.reset(seed=5)
    _, _, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert terminated
    assert not truncated
    assert info["termination_reason"] == "finished"
    assert "truncation_reason" in info
    assert info["truncation_reason"] is None
    assert isinstance(info["episode_return"], float)


def test_step_returns_a_frame_when_done_before_final_repeat(monkeypatch) -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=3),
    )
    monkeypatch.setattr(
        "rl_fzerox.core.envs.engine._read_live_telemetry",
        lambda _backend: _telemetry(race_distance=42.0, state_labels=("finished",)),
    )

    env.reset(seed=6)
    obs, _, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert obs.shape == (120, 160, 12)
    assert terminated
    assert not truncated
    assert info["repeat_index"] == 0
    assert info["termination_reason"] == "finished"


def _telemetry(
    *,
    race_distance: float,
    state_labels: tuple[str, ...] = ("active",),
) -> FZeroXTelemetry:
    state_flags = 1 << 30
    if "collision_recoil" in state_labels:
        state_flags |= 1 << 13
    if "spinning_out" in state_labels:
        state_flags |= 1 << 14
    if "retired" in state_labels:
        state_flags |= 1 << 18
    if "falling_off_track" in state_labels:
        state_flags |= 1 << 19
    if "finished" in state_labels:
        state_flags |= 1 << 25
    if "crashed" in state_labels:
        state_flags |= 1 << 27

    return FZeroXTelemetry(
        system_ram_size=0x00800000,
        game_frame_count=100,
        game_mode_raw=1,
        game_mode_name="gp_race",
        course_index=0,
        in_race_mode=True,
        player=PlayerTelemetry(
            state_flags=state_flags,
            state_labels=state_labels,
            speed_raw=0.0,
            speed_kph=0.0,
            energy=178.0,
            max_energy=178.0,
            boost_timer=0,
            race_distance=race_distance,
            laps_completed_distance=0.0,
            lap_distance=race_distance,
            race_distance_position=race_distance,
            race_time_ms=0,
            lap=1,
            laps_completed=0,
            position=30,
            character=0,
            machine_index=0,
        ),
    )
