# tests/apps/test_training_monitor.py
from __future__ import annotations

from rl_fzerox.apps.run_manager.training_monitor import _estimated_env_step_rate


def test_estimated_env_step_rate_returns_none_without_start_timestamp() -> None:
    assert (
        _estimated_env_step_rate(
            started_at_monotonic=None,
            started_num_timesteps=0,
            current_num_timesteps=100,
            current_monotonic=5.0,
        )
        is None
    )


def test_estimated_env_step_rate_uses_elapsed_timesteps_since_start() -> None:
    value = _estimated_env_step_rate(
        started_at_monotonic=10.0,
        started_num_timesteps=25,
        current_num_timesteps=125,
        current_monotonic=14.0,
    )

    assert value == 25.0


def test_estimated_env_step_rate_clamps_negative_progress_to_zero() -> None:
    value = _estimated_env_step_rate(
        started_at_monotonic=3.0,
        started_num_timesteps=500,
        current_num_timesteps=450,
        current_monotonic=5.0,
    )

    assert value == 0.0
