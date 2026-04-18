# tests/core/envs/test_truncation.py
import numpy as np
import pytest

from fzerox_emulator import ResetState
from rl_fzerox.core.config.schema import ActionConfig, EnvConfig
from rl_fzerox.core.envs import FZeroXEnv
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
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import make_step_status


def test_reset_to_race_requires_custom_baseline():
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=2, reset_to_race=True),
    )

    with pytest.raises(RuntimeError, match="requires a custom baseline state"):
        env.reset(seed=5)


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
    assert reward == pytest.approx(-20.01)
    assert info["truncation_reason"] == "stuck"
    assert info["stalled_steps"] == 2
    assert info["step_reward"] == pytest.approx(-20.01)
    reward_breakdown = info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert reward_breakdown["time"] == -0.005
    assert reward_breakdown["low_speed_time"] == -0.005
    assert reward_breakdown["stuck_truncation"] == pytest.approx(-20.0)


def test_stuck_truncation_can_be_disabled() -> None:
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
        ],
        reset_telemetry=_telemetry(race_distance=0.0),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            max_episode_steps=100,
            stuck_truncation_enabled=False,
            stuck_step_limit=2,
            stuck_min_speed_kph=50.0,
            action=ActionConfig(name="steer_drive"),
        ),
    )

    env.reset(seed=4)
    _, _, _, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert not truncated
    assert backend.last_stuck_step_limit == 101
    assert info["stuck_truncation_enabled"] is False


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
    assert reward == pytest.approx(-0.010)
    assert info["reverse_timer"] == 80

    _, reward, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert not terminated
    assert truncated
    assert reward == pytest.approx(-20.010)
    assert info["truncation_reason"] == "wrong_way"
    assert info["reverse_timer"] == 100
    reward_breakdown = info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert reward_breakdown["time"] == pytest.approx(-0.005)
    assert reward_breakdown["reverse_time"] == -0.005
    assert reward_breakdown["wrong_way_truncation"] == pytest.approx(-20.0)


def test_step_disables_wrong_way_truncation_when_configured() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=-3.0, reverse_timer=10_000),
                summary=_step_summary(
                    max_race_distance=0.0,
                    reverse_active_frames=1,
                    final_frame_index=1,
                ),
                status=make_step_status(step_count=1, reverse_timer=10_000),
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
            wrong_way_truncation_enabled=False,
            wrong_way_timer_limit=100,
            action=ActionConfig(name="steer_drive"),
        ),
    )

    env.reset(seed=7)
    _, _, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert backend.last_wrong_way_timer_limit is None
    assert info["reverse_timer"] == 10_000
    assert not terminated
    assert not truncated


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
    assert reward == pytest.approx(-20.005)
    assert info["truncation_reason"] == "progress_stalled"
    assert info["progress_frontier_stalled_frames"] == 5
    reward_breakdown = info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert reward_breakdown["time"] == pytest.approx(-0.005)
    assert reward_breakdown["progress_stalled_truncation"] == pytest.approx(-20.0)
