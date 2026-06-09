# tests/core/envs/test_env_terminal_serialization.py
import pickle

import numpy as np
import pytest

from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.runtime_spec.schema import (
    EnvConfig,
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
)
from tests.support.native_objects import make_step_status


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
        config=EnvConfig(
            action_repeat=1,
            action=configured_discrete_action("steer", "gas"),
        ),
    )

    env.reset(seed=5)
    _, _, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))

    assert terminated
    assert not truncated
    assert info["termination_reason"] == "finished"
    assert info["entered_finished"] is True
    assert info["entered_retired"] is False
    assert info["entered_crashed"] is False
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
        config=EnvConfig(
            action_repeat=3,
            action=configured_discrete_action("steer", "gas"),
        ),
    )

    env.reset(seed=6)
    obs, _, terminated, truncated, info = env.step(np.array([2, 0], dtype=np.int64))
    obs = _image_obs(obs)

    assert obs.shape == (84, 84, 12)
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
        config=EnvConfig(
            action_repeat=1,
            action=configured_discrete_action("steer", "gas"),
        ),
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
