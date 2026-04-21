# tests/core/game/test_reward_events.py
from __future__ import annotations

import pytest

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.config.schema import RewardConfig
from rl_fzerox.core.envs.rewards import (
    RewardActionContext,
    build_reward_tracker,
)
from tests.support.native_objects import encode_state_flags, make_step_summary, make_telemetry

_COURSE_EFFECT_PIT = 1


def test_race_v3_treats_finish_as_final_lap_reward_only() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            lap_completion_bonus=5.0,
            lap_position_scale=1.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
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
    assert step.breakdown == {
        "lap_completion": 5.0,
        "lap_position": 29.0,
    }


def test_race_v3_applies_small_collision_recoil_entry_penalty() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
            collision_recoil_penalty=-0.25,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, entered_state_labels=("collision_recoil",)),
        _status(step_count=1),
        _telemetry(race_distance=0.0),
    )

    assert step.reward == -0.25
    assert step.breakdown == {"collision_recoil": -0.25}


def test_race_v3_uses_spinning_out_as_failure_not_raw_energy_depletion() -> None:
    spinning_tracker = build_reward_tracker(
        RewardConfig(
            time_penalty_per_frame=0.0,
            failure_penalty=-20.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    spinning_tracker.reset(_telemetry(race_distance=0.0))

    spinning = spinning_tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1, termination_reason="spinning_out"),
        _telemetry(
            race_distance=0.0,
            energy=0.0,
            state_labels=("active", "spinning_out"),
        ),
    )

    energy_tracker = build_reward_tracker(
        RewardConfig(
            time_penalty_per_frame=0.0,
            failure_penalty=-20.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    energy_tracker.reset(_telemetry(race_distance=0.0))
    energy_depleted = energy_tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1, termination_reason="energy_depleted"),
        _telemetry(race_distance=0.0, energy=0.0),
    )

    assert spinning.reward == -20.0
    assert spinning.breakdown == {"spinning_out": -20.0}
    assert energy_depleted.reward == 0.0
    assert energy_depleted.breakdown == {}


def test_race_v3_multiplies_time_penalty_by_frames_run() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            time_penalty_per_frame=-0.01,
            progress_bucket_reward=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
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


def test_race_v3_penalizes_lean_request_below_speed_threshold() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            lean_low_speed_penalty=-0.01,
            lean_low_speed_penalty_max_speed_kph=800.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=0.0, speed_kph=799.0),
        RewardActionContext(lean_requested=True),
    )

    assert step.reward == pytest.approx(-0.03)
    assert step.breakdown == {"lean_low_speed": -0.03}


def test_race_v3_penalizes_missing_discrete_gas_request() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            gas_underuse_penalty=-0.02,
            gas_underuse_threshold=0.5,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=0.0),
        RewardActionContext(gas_level=0.0),
    )

    assert step.reward == pytest.approx(-0.06)
    assert step.breakdown == {"gas_underuse": -0.06}


def test_race_v3_scales_gas_underuse_penalty_below_threshold() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            gas_underuse_penalty=-0.02,
            gas_underuse_threshold=0.5,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    under_threshold = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=2),
        _status(step_count=2),
        _telemetry(race_distance=0.0),
        RewardActionContext(gas_level=0.25),
    )
    at_threshold = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=2),
        _status(step_count=4),
        _telemetry(race_distance=0.0),
        RewardActionContext(gas_level=0.5),
    )

    assert under_threshold.reward == pytest.approx(-0.02)
    assert under_threshold.breakdown == {"gas_underuse": -0.02}
    assert at_threshold.reward == 0.0
    assert at_threshold.breakdown == {}


def test_race_v3_rewards_manual_boost_request_once_per_env_step() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            manual_boost_reward=0.25,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
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


def test_race_v3_penalizes_steering_oscillation_acceleration() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            steer_oscillation_penalty=-0.001,
            steer_oscillation_deadzone=0.0,
            steer_oscillation_cap=2.0,
            steer_oscillation_power=2.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    first = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=1),
        _telemetry(race_distance=0.0),
        RewardActionContext(steer_level=-1.0),
    )
    second = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=2),
        _telemetry(race_distance=0.0),
        RewardActionContext(steer_level=1.0),
    )
    third = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=0.0),
        RewardActionContext(steer_level=-1.0),
    )

    assert first.reward == 0.0
    assert second.reward == 0.0
    assert third.reward == pytest.approx(-0.001)
    assert third.breakdown == {"steer_oscillation": pytest.approx(-0.001)}


def test_race_v3_does_not_penalize_smooth_steering_ramp() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            steer_oscillation_penalty=-0.001,
            steer_oscillation_deadzone=0.0,
            steer_oscillation_cap=2.0,
            steer_oscillation_power=2.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))
    step = None

    for step_count, steer_level in enumerate((0.0, 0.2, 0.4), start=1):
        step = tracker.step_summary(
            _summary(max_race_distance=0.0),
            _status(step_count=step_count),
            _telemetry(race_distance=0.0),
            RewardActionContext(steer_level=steer_level),
        )

    assert step is not None
    assert step.reward == 0.0
    assert step.breakdown == {}


def test_race_v3_does_not_penalize_idle_or_high_speed_lean() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            lean_low_speed_penalty=-0.01,
            lean_low_speed_penalty_max_speed_kph=800.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    idle_step = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1),
        _telemetry(race_distance=0.0, speed_kph=700.0),
        RewardActionContext(lean_requested=False),
    )
    fast_step = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=2),
        _telemetry(race_distance=0.0, speed_kph=800.0),
        RewardActionContext(lean_requested=True),
    )

    assert idle_step.reward == 0.0
    assert idle_step.breakdown == {}
    assert fast_step.reward == 0.0
    assert fast_step.breakdown == {}


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
    )


def _summary(
    *,
    max_race_distance: float,
    frames_run: int = 1,
    reverse_active_frames: int = 0,
    low_speed_frames: int = 0,
    energy_loss_total: float = 0.0,
    energy_gain_total: float = 0.0,
    damage_taken_frames: int = 0,
    entered_state_labels: tuple[str, ...] = (),
) -> StepSummary:
    return make_step_summary(
        frames_run=frames_run,
        max_race_distance=max_race_distance,
        reverse_active_frames=reverse_active_frames,
        low_speed_frames=low_speed_frames,
        energy_loss_total=energy_loss_total,
        energy_gain_total=energy_gain_total,
        damage_taken_frames=damage_taken_frames,
        entered_state_labels=entered_state_labels,
        final_frame_index=frames_run,
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
        step_count=step_count,
        stalled_steps=stalled_steps,
        reverse_timer=reverse_timer,
        termination_reason=termination_reason,
        truncation_reason=truncation_reason,
    )
