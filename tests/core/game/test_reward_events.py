# tests/core/game/test_reward_events.py
from __future__ import annotations

import pytest

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.rewards import RewardActionContext, build_reward_tracker
from rl_fzerox.core.runtime_spec.schema import RewardConfig
from tests.support.native_objects import encode_state_flags, make_step_summary, make_telemetry

_COURSE_EFFECT_PIT = 1


def test_reward_main_treats_finish_as_final_lap_reward_only() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            lap_completion_bonus=5.0,
            lap_position_scale=1.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, laps_completed=2))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, entered_state_labels=("finished",)),
        _status(step_count=120, termination_reason="finished"),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "finished"),
            position=1,
            laps_completed=3,
        ),
    )

    assert step.reward == pytest.approx(34.0)
    assert step.breakdown == {"lap_completion": 5.0, "lap_position": 29.0}


def test_reward_main_applies_impact_penalty_per_frame_while_recoil_active() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            time_penalty_per_frame=0.0,
            impact_frame_penalty=-0.25,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    first_step = tracker.step_summary(
        _summary(
            max_race_distance=0.0,
            frames_run=3,
            collision_recoil_active_frames=3,
        ),
        _status(step_count=1),
        _telemetry(race_distance=0.0, state_labels=("active", "collision_recoil")),
    )
    second_step = tracker.step_summary(
        _summary(
            max_race_distance=0.0,
            frames_run=2,
            collision_recoil_active_frames=1,
        ),
        _status(step_count=2),
        _telemetry(race_distance=0.0),
    )

    assert first_step.reward == -0.75
    assert first_step.breakdown == {"impact": -0.75}
    assert second_step.reward == -0.25
    assert second_step.breakdown == {"impact": -0.25}


def test_reward_main_does_not_apply_impact_penalty_from_entry_flag_alone() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            time_penalty_per_frame=0.0,
            impact_frame_penalty=-0.25,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, entered_state_labels=("collision_recoil",)),
        _status(step_count=1),
        _telemetry(race_distance=0.0),
    )

    assert step.reward == 0.0
    assert step.breakdown == {}


def test_reward_main_uses_spinning_out_as_failure_not_raw_energy_depletion() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            time_penalty_per_frame=0.0,
            failure_penalty=-20.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    spinning = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1, termination_reason="spinning_out"),
        _telemetry(
            race_distance=0.0,
            energy=0.0,
            state_labels=("active", "spinning_out"),
        ),
    )
    energy_depleted = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=2, termination_reason="energy_depleted"),
        _telemetry(race_distance=0.0, energy=0.0),
    )

    assert spinning.breakdown == {"spinning_out": -20.0}
    assert energy_depleted.breakdown == {}


def test_reward_main_multiplies_time_penalty_by_frames_run() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            time_penalty_per_frame=-0.01,
            progress_bucket_reward=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=0.0),
    )

    assert step.reward == pytest.approx(-0.03)
    assert step.breakdown == {"time": -0.03}


def test_reward_main_penalizes_lean_request() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            lean_request_penalty=-0.001,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=0.0, speed_kph=1_500.0),
        RewardActionContext(lean_requested=True),
    )

    assert step.reward == pytest.approx(-0.003)
    assert step.breakdown == {"lean": -0.003}


def test_reward_main_penalizes_lean_activation_once() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            lean_request_penalty=-0.001,
            lean_activation_penalty=-0.01,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    first = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=0.0, speed_kph=1_500.0),
        RewardActionContext(lean_requested=True),
    )
    held = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=6),
        _telemetry(race_distance=0.0, speed_kph=1_500.0),
        RewardActionContext(lean_requested=True),
    )
    released = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=9),
        _telemetry(race_distance=0.0, speed_kph=1_500.0),
        RewardActionContext(lean_requested=False),
    )
    reactivated = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=12),
        _telemetry(race_distance=0.0, speed_kph=1_500.0),
        RewardActionContext(lean_requested=True),
    )

    assert first.reward == pytest.approx(-0.013)
    assert first.breakdown == {"lean": -0.003, "lean_activation": -0.01}
    assert held.reward == pytest.approx(-0.003)
    assert held.breakdown == {"lean": -0.003}
    assert released.reward == 0.0
    assert released.breakdown == {}
    assert reactivated.reward == pytest.approx(-0.013)
    assert reactivated.breakdown == {"lean": -0.003, "lean_activation": -0.01}


def test_reward_main_penalizes_grounded_pitch_outside_deadzone() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            grounded_pitch_penalty=-0.01,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=0.0),
        RewardActionContext(pitch_level=0.6, pitch_deadzone=0.1),
    )

    assert step.reward == pytest.approx(-0.0166666667)
    assert step.breakdown == {"grounded_pitch": pytest.approx(-0.0166666667)}


def test_reward_main_does_not_penalize_grounded_pitch_within_deadzone() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            grounded_pitch_penalty=-0.01,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=0.0),
        RewardActionContext(pitch_level=0.2, pitch_deadzone=0.25),
    )

    assert step.reward == 0.0
    assert step.breakdown == {}


def test_reward_main_penalizes_air_brake_request() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            air_brake_request_penalty=-0.01,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=0.0),
        RewardActionContext(air_brake_requested=True),
    )

    assert step.reward == pytest.approx(-0.03)
    assert step.breakdown == {"air_brake": -0.03}


def test_reward_main_penalizes_spin_request_once_per_env_step() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            spin_request_penalty=-0.02,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    requested = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=0.0),
        RewardActionContext(spin_requested=True),
    )
    idle = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=6),
        _telemetry(race_distance=0.0),
        RewardActionContext(spin_requested=False),
    )

    assert requested.reward == pytest.approx(-0.02)
    assert requested.breakdown == {"spin": -0.02}
    assert idle.reward == 0.0
    assert idle.breakdown == {}


def test_reward_main_rewards_manual_boost_request_once_per_env_step() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            manual_boost_reward=0.25,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=0.0),
        RewardActionContext(boost_requested=True),
    )

    assert step.reward == pytest.approx(0.25)
    assert step.breakdown == {"manual_boost": 0.25}


def test_reward_main_rewards_landing_once_per_frontier_bucket() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            airborne_landing_reward=5.0,
            airborne_landing_grace_frames=0,
            airborne_landing_min_peak_height=50.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            height_above_ground=140.0,
        )
    )

    first_landing = tracker.step_summary(
        _summary(max_race_distance=350.0),
        _status(step_count=1),
        _telemetry(race_distance=350.0),
    )
    takeoff_without_progress = tracker.step_summary(
        _summary(max_race_distance=350.0),
        _status(step_count=2),
        _telemetry(
            race_distance=350.0,
            state_labels=("active", "airborne"),
            height_above_ground=120.0,
        ),
    )
    repeated_landing = tracker.step_summary(
        _summary(max_race_distance=350.0),
        _status(step_count=3),
        _telemetry(race_distance=350.0),
    )
    takeoff_with_progress = tracker.step_summary(
        _summary(max_race_distance=450.0),
        _status(step_count=4),
        _telemetry(
            race_distance=450.0,
            state_labels=("active", "airborne"),
            height_above_ground=120.0,
        ),
    )
    next_bucket_landing = tracker.step_summary(
        _summary(max_race_distance=450.0),
        _status(step_count=5),
        _telemetry(race_distance=450.0),
    )

    assert first_landing.reward == 5.0
    assert first_landing.breakdown == {"landing": 5.0}
    assert takeoff_without_progress.reward == 0.0
    assert takeoff_without_progress.breakdown == {}
    assert repeated_landing.reward == 0.0
    assert repeated_landing.breakdown == {}
    assert takeoff_with_progress.reward == 0.0
    assert takeoff_with_progress.breakdown == {}
    assert next_bucket_landing.reward == 5.0
    assert next_bucket_landing.breakdown == {"landing": 5.0}


def test_reward_main_requires_airborne_grace_for_landing_reward() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            airborne_landing_reward=5.0,
            airborne_landing_grace_frames=50,
            airborne_landing_min_peak_height=50.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    tracker.step_summary(
        _summary(max_race_distance=350.0, airborne_frames=20),
        _status(step_count=1),
        _telemetry(
            race_distance=350.0,
            state_labels=("active", "airborne"),
            height_above_ground=120.0,
        ),
    )
    short_jump_landing = tracker.step_summary(
        _summary(max_race_distance=350.0, airborne_frames=20),
        _status(step_count=2),
        _telemetry(race_distance=350.0),
    )

    tracker.reset(_telemetry(race_distance=0.0))
    tracker.step_summary(
        _summary(max_race_distance=350.0, airborne_frames=25),
        _status(step_count=1),
        _telemetry(
            race_distance=350.0,
            state_labels=("active", "airborne"),
            height_above_ground=120.0,
        ),
    )
    long_jump_landing = tracker.step_summary(
        _summary(max_race_distance=350.0, airborne_frames=25),
        _status(step_count=2),
        _telemetry(race_distance=350.0),
    )

    assert short_jump_landing.reward == 0.0
    assert short_jump_landing.breakdown == {}
    assert long_jump_landing.reward == 5.0
    assert long_jump_landing.breakdown == {"landing": 5.0}


def test_reward_main_requires_peak_height_for_landing_reward() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            airborne_landing_reward=5.0,
            airborne_landing_grace_frames=0,
            airborne_landing_min_peak_height=50.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    tracker.step_summary(
        _summary(max_race_distance=350.0, airborne_frames=10),
        _status(step_count=1),
        _telemetry(
            race_distance=350.0,
            state_labels=("active", "airborne"),
            height_above_ground=20.0,
        ),
    )
    shallow_landing = tracker.step_summary(
        _summary(max_race_distance=350.0, airborne_frames=10),
        _status(step_count=2),
        _telemetry(race_distance=350.0),
    )

    assert shallow_landing.reward == 0.0
    assert shallow_landing.breakdown == {}


def _telemetry(
    *,
    race_distance: float,
    state_labels: tuple[str, ...] = ("active",),
    position: int = 30,
    energy: float = 178.0,
    boost_timer: int = 0,
    race_time_ms: int = 0,
    speed_kph: float = 100.0,
    laps_completed: int = 0,
    lap: int | None = None,
    reverse_timer: int = 0,
    on_energy_refill: bool = False,
    height_above_ground: float = 0.0,
) -> FZeroXTelemetry:
    state_flags = encode_state_flags(state_labels)
    if on_energy_refill:
        state_flags |= _COURSE_EFFECT_PIT
    return make_telemetry(
        race_distance=race_distance,
        state_labels=state_labels,
        state_flags=state_flags,
        speed_kph=speed_kph,
        energy=energy,
        boost_timer=boost_timer,
        race_time_ms=race_time_ms,
        position=position,
        laps_completed=laps_completed,
        lap=max(laps_completed + 1, 1) if lap is None else lap,
        reverse_timer=reverse_timer,
        height_above_ground=height_above_ground,
    )


def _summary(
    *,
    max_race_distance: float,
    frames_run: int = 1,
    reverse_active_frames: int = 0,
    collision_recoil_active_frames: int = 0,
    energy_gain_total: float = 0.0,
    damage_taken_frames: int = 0,
    entered_state_labels: tuple[str, ...] = (),
    airborne_frames: int = 0,
) -> StepSummary:
    return make_step_summary(
        frames_run=frames_run,
        max_race_distance=max_race_distance,
        reverse_active_frames=reverse_active_frames,
        collision_recoil_active_frames=collision_recoil_active_frames,
        energy_gain_total=energy_gain_total,
        damage_taken_frames=damage_taken_frames,
        entered_state_labels=entered_state_labels,
        final_frame_index=frames_run,
        airborne_frames=airborne_frames,
    )


def _status(
    *,
    step_count: int,
    stalled_steps: int = 0,
    reverse_timer: int = 0,
    termination_reason: str | None = None,
    truncation_reason: str | None = None,
) -> StepStatus:
    return StepStatus(
        {
            "step_count": step_count,
            "stalled_steps": stalled_steps,
            "reverse_timer": reverse_timer,
            "termination_reason": termination_reason,
            "truncation_reason": truncation_reason,
        }
    )
