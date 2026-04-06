# tests/test_env.py
from pathlib import Path

import numpy as np
from gymnasium.spaces import Discrete

from rl_fzerox.core.config.models import EnvConfig
from rl_fzerox.core.emulator.base import ResetState
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
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == 1


def test_step_advances_backend_by_action_repeat():
    backend = SyntheticBackend()
    env = FZeroXEnv(backend=backend, config=EnvConfig(action_repeat=3))

    env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(0)

    assert obs.shape == backend.frame_shape
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert backend.frame_index == 3
    assert info["repeat_index"] == 2


def test_step_updates_raw_observation_during_headless_smoke_rollout():
    env = FZeroXEnv(backend=SyntheticBackend(), config=EnvConfig(action_repeat=1))

    obs_before, _ = env.reset(seed=9)
    obs_after, _, _, _, _ = env.step(0)

    assert not np.array_equal(obs_before, obs_after)


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
    assert info["boot_state"] == "ready"
    assert backend.frame_index == 1_592


def test_reset_skips_bootstrap_when_a_saved_baseline_exists(tmp_path: Path):
    class BaselineBackend(SyntheticBackend):
        def reset(self) -> ResetState:
            reset_state = super().reset()
            info = dict(reset_state.info)
            info["baseline_state_path"] = str((tmp_path / "baseline.state").resolve())
            return ResetState(frame=reset_state.frame, info=info)

    baseline_path = tmp_path / "baseline.state"
    baseline_path.write_bytes(b"baseline")

    backend = BaselineBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=2, reset_to_race=True),
    )

    _, info = env.reset(seed=11)

    assert info["seed"] == 11
    assert "boot_state" not in info
    assert backend.frame_index == 0
