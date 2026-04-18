# tests/core/envs/test_lean_semantics.py
import numpy as np

from rl_fzerox.core.config.schema import ActionConfig, EnvConfig
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.actions import LEAN_LEFT_MASK, LEAN_RIGHT_MASK
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
from tests.support.native_objects import make_step_status


def test_env_releases_lean_input_without_python_side_latch() -> None:
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
            action=ActionConfig(
                name="steer_drive_boost_lean",
                lean_mode="timer_assist",
            ),
        ),
    )

    env.reset(seed=3)
    env.step(np.array([3, 0, 0, 1], dtype=np.int64))

    assert backend.last_controller_state.joypad_mask & LEAN_LEFT_MASK != 0
    assert backend.last_lean_timer_assist is True
    assert env.action_masks().tolist()[-3:] == [True, True, True]

    env.step(np.array([3, 0, 0, 0], dtype=np.int64))

    assert backend.last_controller_state.joypad_mask & LEAN_LEFT_MASK == 0
    assert env.action_masks().tolist()[-3:] == [True, True, True]


def test_env_minimum_hold_mode_keeps_lean_pressed_for_guard_window() -> None:
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
            action=ActionConfig(
                name="steer_drive_boost_lean",
                lean_mode="minimum_hold",
            ),
        ),
    )

    env.reset(seed=3)
    env.step(np.array([3, 0, 0, 1], dtype=np.int64))

    assert env.action_masks().tolist()[-3:] == [False, True, False]
    assert backend.last_lean_timer_assist is False

    env.step(np.array([3, 0, 0, 0], dtype=np.int64))

    assert backend.last_controller_state.joypad_mask & LEAN_LEFT_MASK != 0
    assert env.action_masks().tolist()[-3:] == [False, True, False]


def test_env_release_cooldown_mode_blocks_retap_after_short_lean() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=5, frames_run=5),
                status=make_step_status(step_count=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=20.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=20.0, final_frame_index=10, frames_run=5),
                status=make_step_status(step_count=2),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=30.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=30.0, final_frame_index=15, frames_run=5),
                status=make_step_status(step_count=3),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=40.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=40.0, final_frame_index=20, frames_run=5),
                status=make_step_status(step_count=4),
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
            action=ActionConfig(
                name="steer_drive_boost_lean",
                lean_mode="release_cooldown",
            ),
        ),
    )

    env.reset(seed=3)
    env.step(np.array([3, 0, 0, 1], dtype=np.int64))
    assert backend.last_controller_state.joypad_mask & LEAN_LEFT_MASK != 0
    assert backend.last_lean_timer_assist is False
    assert env.action_masks().tolist()[-3:] == [True, True, False]

    env.step(np.array([3, 0, 0, 0], dtype=np.int64))
    assert backend.last_controller_state.joypad_mask & LEAN_LEFT_MASK == 0
    assert env.action_masks().tolist()[-3:] == [True, False, False]

    env.step(np.array([3, 0, 0, 1], dtype=np.int64))
    assert backend.last_controller_state.joypad_mask & LEAN_LEFT_MASK == 0
    assert env.action_masks().tolist()[-3:] == [True, False, False]

    env.step(np.array([3, 0, 0, 0], dtype=np.int64))
    assert env.action_masks().tolist()[-3:] == [True, True, True]


def test_env_release_cooldown_mode_blocks_direct_lean_switch() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=5, frames_run=5),
                status=make_step_status(step_count=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=20.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=20.0, final_frame_index=10, frames_run=5),
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
            action_repeat=5,
            action=ActionConfig(
                name="steer_drive_boost_lean",
                lean_mode="release_cooldown",
            ),
        ),
    )

    env.reset(seed=3)
    env.step(np.array([3, 0, 0, 1], dtype=np.int64))
    assert env.action_masks().tolist()[-3:] == [True, True, False]

    env.step(np.array([3, 0, 0, 2], dtype=np.int64))
    assert backend.last_controller_state.joypad_mask & LEAN_RIGHT_MASK == 0
    assert env.action_masks().tolist()[-3:] == [True, False, False]


def test_env_keeps_lean_speed_mask_without_lean_latch() -> None:
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
            action=ActionConfig(
                name="steer_drive_boost_lean",
                lean_mode="timer_assist",
                lean_unmask_min_speed_kph=500.0,
            ),
        ),
    )

    env.reset(seed=4)
    assert env.action_masks().tolist()[-3:] == [True, False, False]

    env.step(np.array([3, 0, 0, 1], dtype=np.int64))
    assert env.action_masks().tolist()[-3:] == [True, False, False]

    env.step(np.array([3, 0, 0, 1], dtype=np.int64))
    assert env.action_masks().tolist()[-3:] == [True, False, False]

    env.step(np.array([3, 0, 0, 1], dtype=np.int64))
    assert env.action_masks().tolist()[-3:] == [True, False, False]
