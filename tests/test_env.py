# tests/test_env.py
import numpy as np
from gymnasium.spaces import MultiDiscrete

from rl_fzerox.core.actions.steer_drive import THROTTLE_MASK
from rl_fzerox.core.config.models import EnvConfig
from rl_fzerox.core.emulator.base import ResetState
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.envs import FZeroXEnv
from tests.fakes import SyntheticBackend


def test_reset_returns_raw_frame_observation():
    backend = SyntheticBackend()
    env = FZeroXEnv(backend=backend, config=EnvConfig(action_repeat=2))

    obs, info = env.reset(seed=123)

    assert obs.shape == backend.frame_shape
    assert obs.dtype == np.uint8
    assert info["backend"] == "synthetic"
    assert info["seed"] == 123
    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [5, 2]


def test_step_advances_backend_by_action_repeat():
    backend = SyntheticBackend()
    env = FZeroXEnv(backend=backend, config=EnvConfig(action_repeat=3))

    env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(np.array([3, 1], dtype=np.int64))

    assert obs.shape == backend.frame_shape
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert backend.frame_index == 3
    assert info["repeat_index"] == 2
    assert backend.last_controller_state == ControllerState(
        joypad_mask=THROTTLE_MASK,
        left_stick_x=0.5,
    )


def test_step_updates_raw_observation_during_headless_smoke_rollout():
    env = FZeroXEnv(backend=SyntheticBackend(), config=EnvConfig(action_repeat=1))

    obs_before, _ = env.reset(seed=9)
    obs_after, _, _, _, _ = env.step(np.array([3, 0], dtype=np.int64))

    assert not np.array_equal(obs_before, obs_after)


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

    assert obs.shape == backend.frame_shape
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
