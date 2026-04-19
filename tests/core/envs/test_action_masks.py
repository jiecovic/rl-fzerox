# tests/core/envs/test_action_masks.py
import numpy as np
import pytest

from rl_fzerox.core.config.schema import (
    ActionConfig,
    ActionMaskConfig,
    CurriculumConfig,
    CurriculumStageConfig,
    CurriculumTriggerConfig,
    EnvConfig,
)
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.actions import BOOST_MASK
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


def test_env_action_masks_reflect_base_action_mask_config() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=ActionConfig(
                name="steer_drive_boost_lean",
                mask=ActionMaskConfig(lean=(0,)),
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
                    name="hybrid_steer_drive_boost_lean",
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
            config=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_lean")),
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
                name="steer_drive_boost_lean",
                mask=ActionMaskConfig(lean=(0,)),
            )
        ),
        curriculum_config=CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(
                    name="basic_drive",
                    until=CurriculumTriggerConfig(race_laps_completed_mean_gte=3.0),
                    action_mask=ActionMaskConfig(lean=(0,)),
                ),
                CurriculumStageConfig(
                    name="full_controls",
                    action_mask=ActionMaskConfig(lean=(0, 1, 2)),
                ),
            ),
        ),
    )

    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 3) + ([True] * 2) + [True, False, False]
    )

    env.set_curriculum_stage(1)

    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2 + 3))


def test_hybrid_curriculum_stage_can_speed_gate_lean_temporarily() -> None:
    backend = ScriptedStepBackend(
        [],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
            speed_kph=800.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_boost_lean")),
        curriculum_config=CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(
                    name="high_speed_lean",
                    until=CurriculumTriggerConfig(race_laps_completed_mean_gte=1.0),
                    action_mask=ActionMaskConfig(boost=(0,), lean=(0, 1, 2)),
                    lean_unmask_min_speed_kph=900.0,
                ),
                CurriculumStageConfig(
                    name="full_controls",
                    action_mask=ActionMaskConfig(boost=(0, 1), lean=(0, 1, 2)),
                ),
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == [True, False, False, True, False]

    env.set_curriculum_stage(1)

    assert env.action_masks().tolist() == [True, True, True, True, True]


def test_env_sync_checkpoint_curriculum_stage_resets_to_default_stage() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action=ActionConfig(
                name="steer_drive_boost_lean",
                mask=ActionMaskConfig(lean=(0,)),
            )
        ),
        curriculum_config=CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(
                    name="basic_drive",
                    until=CurriculumTriggerConfig(race_laps_completed_mean_gte=3.0),
                    action_mask=ActionMaskConfig(lean=(0,)),
                ),
                CurriculumStageConfig(
                    name="full_controls",
                    action_mask=ActionMaskConfig(lean=(0, 1, 2)),
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
        config=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_boost_lean")),
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


def test_hybrid_lean_primitive_masks_boost_until_telemetry_unlocks_it() -> None:
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
        config=EnvConfig(action=ActionConfig(name="hybrid_steer_drive_boost_lean_primitive")),
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


def test_env_action_masks_disable_lean_below_speed_threshold() -> None:
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
                name="steer_drive_boost_lean",
                lean_unmask_min_speed_kph=500.0,
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 3) + ([True] * 2) + [True, False, False]
    )

    env.step(np.array([3, 1, 0, 0], dtype=np.int64))

    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2 + 3))


def test_env_action_masks_static_lean_mask_wins_over_speed_unmask() -> None:
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
                name="steer_drive_boost_lean",
                lean_unmask_min_speed_kph=500.0,
                mask=ActionMaskConfig(lean=(0,)),
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


def test_env_action_masks_disable_boost_above_speed_threshold() -> None:
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
            speed_kph=750.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=ActionConfig(
                name="steer_drive_boost",
                boost_unmask_max_speed_kph=700.0,
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 3) + [True, False])

    env.step(np.array([3, 1, 0], dtype=np.int64))

    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2))


def test_env_action_masks_keep_boost_masked_when_speed_cap_allows_before_unlock() -> None:
    backend = ScriptedStepBackend(
        [],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active",),
            speed_kph=650.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=ActionConfig(
                name="steer_drive_boost",
                boost_unmask_max_speed_kph=700.0,
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 3) + [True, False])


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


def test_env_action_masks_disable_boost_while_reversing() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "can_boost"),
                    reverse_timer=0,
                ),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "can_boost"),
            reverse_timer=12,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action=ActionConfig(name="steer_drive_boost")),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 3) + [True, False])

    env.step(np.array([3, 1, 1], dtype=np.int64))

    assert backend.last_controller_state.joypad_mask & BOOST_MASK == 0
    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2))


def test_env_action_masks_disable_boost_while_airborne() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(
                    race_distance=10.0,
                    state_labels=("active", "can_boost"),
                ),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            )
        ],
        reset_telemetry=_telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne", "can_boost"),
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action=ActionConfig(name="steer_drive_boost")),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 3) + [True, False])

    env.step(np.array([3, 1, 1], dtype=np.int64))

    assert backend.last_controller_state.joypad_mask & BOOST_MASK == 0
    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2))


def test_env_action_masks_expose_boost_only_on_decision_interval() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=20.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=20.0, final_frame_index=2),
                status=make_step_status(step_count=2),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=30.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=30.0, final_frame_index=3),
                status=make_step_status(step_count=3),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active", "can_boost")),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=ActionConfig(
                name="steer_drive_boost",
                boost_decision_interval_frames=3,
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2))

    env.step(np.array([3, 1, 0], dtype=np.int64))
    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 3) + [True, False])

    env.step(np.array([3, 1, 0], dtype=np.int64))
    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 3) + [True, False])

    env.step(np.array([3, 1, 0], dtype=np.int64))
    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2))


def test_env_action_masks_keep_boost_decision_slots_when_frame_skip_crosses_interval() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(
                    frames_run=2,
                    max_race_distance=10.0,
                    final_frame_index=2,
                ),
                status=make_step_status(step_count=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=20.0, state_labels=("active", "can_boost")),
                summary=_step_summary(
                    frames_run=2,
                    max_race_distance=20.0,
                    final_frame_index=4,
                ),
                status=make_step_status(step_count=2),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active", "can_boost")),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=2,
            action=ActionConfig(
                name="steer_drive_boost",
                boost_decision_interval_frames=3,
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2))

    env.step(np.array([3, 1, 0], dtype=np.int64))
    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 3) + [True, False])

    env.step(np.array([3, 1, 0], dtype=np.int64))
    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2))


def test_env_action_masks_lock_boost_after_manual_request() -> None:
    backend = ScriptedStepBackend(
        [
            _backend_step_result(
                telemetry=_telemetry(race_distance=10.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=10.0, final_frame_index=1),
                status=make_step_status(step_count=1),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=20.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=20.0, final_frame_index=2),
                status=make_step_status(step_count=2),
            ),
            _backend_step_result(
                telemetry=_telemetry(race_distance=30.0, state_labels=("active", "can_boost")),
                summary=_step_summary(max_race_distance=30.0, final_frame_index=3),
                status=make_step_status(step_count=3),
            ),
        ],
        reset_telemetry=_telemetry(race_distance=0.0, state_labels=("active", "can_boost")),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action=ActionConfig(
                name="steer_drive_boost",
                boost_request_lockout_frames=3,
            ),
        ),
    )

    env.reset(seed=1)

    assert env.action_masks().tolist() == ([True] * (7 + 3 + 2))

    env.step(np.array([3, 1, 1], dtype=np.int64))
    assert backend.last_controller_state.joypad_mask & BOOST_MASK == BOOST_MASK
    assert env.action_masks().tolist() == (([True] * 7) + ([True] * 3) + [True, False])

    env.step(np.array([3, 1, 0], dtype=np.int64))
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
                name="steer_drive_boost_lean",
                mask=ActionMaskConfig(lean=(0,)),
            )
        ),
        curriculum_config=CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(
                    name="basic_drive",
                    until=CurriculumTriggerConfig(race_laps_completed_mean_gte=3.0),
                    action_mask=ActionMaskConfig(lean=(0,)),
                ),
            ),
        ),
    )

    env.reset(seed=2)

    assert env.action_masks().tolist() == (
        ([True] * 7) + ([True] * 3) + [True, False] + [True, False, False]
    )
