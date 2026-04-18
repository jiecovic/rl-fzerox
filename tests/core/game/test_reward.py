# tests/core/game/test_reward.py
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.config.schema import RewardConfig
from rl_fzerox.core.envs.rewards import (
    DEFAULT_REWARD_NAME,
    RaceV3RewardTracker,
    RaceV3RewardWeights,
    build_reward_tracker,
    reward_tracker_names,
)
from tests.support.native_objects import encode_state_flags, make_step_summary, make_telemetry

_COURSE_EFFECT_PIT = 1


def test_reward_yamls_use_known_reward_config_keys() -> None:
    schema_keys = set(RewardConfig.model_fields)
    for yaml_path in Path("conf/reward").glob("*.yaml"):
        config_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

        assert set(config_data) <= schema_keys
        assert RewardConfig.model_validate(config_data).name in reward_tracker_names()


def test_race_v3_is_the_only_registered_reward_profile() -> None:
    assert DEFAULT_REWARD_NAME == "race_v3"
    assert reward_tracker_names() == ("race_v3",)
    assert isinstance(build_reward_tracker(), RaceV3RewardTracker)


def test_race_v3_weight_fields_match_reward_config_schema() -> None:
    weight_fields = {field.name for field in fields(RaceV3RewardWeights)}

    assert weight_fields == set(RewardConfig.model_fields) - {"name"}


def test_build_reward_tracker_wires_all_race_v3_weight_fields() -> None:
    overrides = {
        "energy_loss_epsilon": 0.04,
        "progress_bucket_distance": 123.0,
        "progress_bucket_reward": 2.5,
        "progress_reward_interval_frames": 7,
        "time_penalty_per_frame": -0.002,
        "reverse_time_penalty_scale": 1.25,
        "low_speed_time_penalty_scale": 1.5,
        "lap_completion_bonus": 9.0,
        "lap_position_scale": 0.33,
        "damage_taken_frame_penalty": -0.02,
        "damage_taken_streak_ramp_penalty": -0.001,
        "damage_taken_streak_cap_frames": 120,
        "boost_pad_reward": 10.0,
        "boost_pad_reward_progress_window": 800.0,
        "energy_refill_progress_multiplier": 3.0,
        "energy_refill_collision_cooldown_frames": 17,
        "energy_full_refill_lap_bonus": 4.0,
        "energy_full_refill_min_gain_fraction": 0.12,
        "gas_underuse_penalty": -0.03,
        "gas_underuse_threshold": 0.25,
        "steer_oscillation_penalty": -0.004,
        "steer_oscillation_deadzone": 0.05,
        "steer_oscillation_cap": 1.5,
        "steer_oscillation_power": 1.5,
        "lean_low_speed_penalty": -0.01,
        "lean_low_speed_penalty_max_speed_kph": 800.0,
        "airborne_landing_reward": 5.0,
        "collision_recoil_penalty": -0.25,
        "failure_penalty": -30.0,
        "truncation_penalty": -15.0,
    }
    assert set(overrides) == {field.name for field in fields(RaceV3RewardWeights)}

    tracker = build_reward_tracker(RewardConfig(**overrides))

    assert isinstance(tracker, RaceV3RewardTracker)
    weights = tracker._weights
    actual = {field.name: getattr(weights, field.name) for field in fields(RaceV3RewardWeights)}
    assert actual == overrides


def test_race_v3_rewards_each_frontier_bucket_once() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=1_000.0,
            progress_bucket_reward=2.0,
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    before_first_bucket = tracker.step_summary(
        _summary(max_race_distance=999.0),
        _status(step_count=1),
        _telemetry(race_distance=999.0),
    )
    first_step = tracker.step_summary(
        _summary(max_race_distance=2_500.0),
        _status(step_count=2),
        _telemetry(race_distance=2_500.0),
    )
    repeated_step = tracker.step_summary(
        _summary(max_race_distance=2_500.0),
        _status(step_count=3),
        _telemetry(race_distance=500.0),
    )

    assert before_first_bucket.reward == 0.0
    assert first_step.reward == 4.0
    assert first_step.breakdown == {"frontier_progress": 4.0}
    assert repeated_step.reward == 0.0
    info = tracker.info(_telemetry(race_distance=2_500.0))
    assert info["frontier_progress_bucket_index"] == 2
    assert info["frontier_progress_distance"] == 2_000.0


def test_race_v3_can_delay_frontier_rewards_by_interval() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            progress_reward_interval_frames=3,
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    held_step = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=1),
        _status(step_count=1),
        _telemetry(race_distance=100.0),
    )
    flushed_step = tracker.step_summary(
        _summary(max_race_distance=300.0, frames_run=2),
        _status(step_count=3),
        _telemetry(race_distance=300.0),
    )

    assert held_step.reward == 0.0
    assert flushed_step.breakdown == {"frontier_progress": 3.0}


def test_race_v3_multiplies_frontier_progress_when_energy_refills() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            energy_refill_progress_multiplier=2.0,
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, energy=89.0))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_gain_total=10.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=99.0, on_energy_refill=True),
    )

    refill_bonus = 1.0
    assert step.reward == pytest.approx(1.0 + refill_bonus)
    assert step.breakdown["frontier_progress"] == 1.0
    assert step.breakdown["energy_refill_progress"] == pytest.approx(refill_bonus)


def test_race_v3_does_not_reward_refill_without_new_frontier_progress() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            energy_refill_progress_multiplier=2.0,
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, energy=89.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, energy_gain_total=10.0),
        _status(step_count=1),
        _telemetry(race_distance=0.0, energy=99.0, on_energy_refill=True),
    )

    assert step.reward == 0.0
    assert step.breakdown == {}


def test_race_v3_suppresses_refill_multiplier_while_reversing() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            energy_refill_progress_multiplier=2.0,
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, energy=89.0))

    step = tracker.step_summary(
        _summary(
            max_race_distance=100.0,
            energy_gain_total=10.0,
            reverse_active_frames=1,
        ),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=99.0, on_energy_refill=True),
    )

    assert step.reward == 1.0
    assert step.breakdown == {"frontier_progress": 1.0}


def test_race_v3_rewards_full_energy_refill_once_per_lap() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            energy_refill_progress_multiplier=1.0,
            energy_full_refill_lap_bonus=2.5,
            energy_full_refill_min_gain_fraction=0.1,
            lap_completion_bonus=0.0,
            lap_position_scale=0.0,
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, energy=0.0))

    first_full = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_gain_total=178.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=178.0),
    )
    tracker.step_summary(
        _summary(max_race_distance=200.0),
        _status(step_count=2),
        _telemetry(race_distance=200.0, energy=0.0),
    )
    blocked_same_lap = tracker.step_summary(
        _summary(max_race_distance=300.0, energy_gain_total=178.0),
        _status(step_count=3),
        _telemetry(race_distance=300.0, energy=178.0),
    )
    tracker.step_summary(
        _summary(max_race_distance=400.0),
        _status(step_count=4),
        _telemetry(race_distance=400.0, energy=0.0, laps_completed=1),
    )
    next_lap_full = tracker.step_summary(
        _summary(max_race_distance=500.0, energy_gain_total=178.0),
        _status(step_count=5),
        _telemetry(race_distance=500.0, energy=178.0, laps_completed=1),
    )

    assert first_full.breakdown == {"energy_full_refill_lap": 2.5}
    assert blocked_same_lap.breakdown == {}
    assert next_lap_full.breakdown == {"energy_full_refill_lap": 2.5}
    assert tracker.info(_telemetry(race_distance=500.0))["rewarded_full_refill_laps"] == 2


def test_race_v3_scales_full_refill_bonus_by_recovered_energy_fraction() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            energy_refill_progress_multiplier=1.0,
            energy_full_refill_lap_bonus=20.0,
            energy_full_refill_min_gain_fraction=0.1,
            lap_completion_bonus=0.0,
            lap_position_scale=0.0,
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, energy=140.0))

    partial = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_gain_total=20.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=160.0),
    )
    full = tracker.step_summary(
        _summary(max_race_distance=200.0, energy_gain_total=18.0),
        _status(step_count=2),
        _telemetry(race_distance=200.0, energy=178.0),
    )

    assert partial.breakdown == {}
    assert full.breakdown == {"energy_full_refill_lap": pytest.approx(20.0 * (38.0 / 178.0))}


def test_race_v3_suppresses_tiny_full_refill_bonus() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            energy_refill_progress_multiplier=1.0,
            energy_full_refill_lap_bonus=20.0,
            energy_full_refill_min_gain_fraction=0.1,
            lap_completion_bonus=0.0,
            lap_position_scale=0.0,
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, energy=177.0))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_gain_total=1.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=178.0),
    )

    assert step.breakdown == {}


def test_race_v3_suppresses_full_energy_refill_while_reversing() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            energy_refill_progress_multiplier=1.0,
            energy_full_refill_lap_bonus=2.5,
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, energy=100.0))

    step = tracker.step_summary(
        _summary(
            max_race_distance=100.0,
            energy_gain_total=78.0,
            reverse_active_frames=1,
        ),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=178.0, reverse_timer=1),
    )

    assert step.reward == 0.0
    assert step.breakdown == {}


def test_race_v3_rewards_dash_pad_boost_entries_once_per_progress_window() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            boost_pad_reward=0.5,
            boost_pad_reward_progress_window=1000.0,
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    first = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=1, entered_state_labels=("dash_pad_boost",)),
        _status(step_count=1),
        _telemetry(race_distance=100.0, state_labels=("active", "dash_pad_boost")),
    )
    blocked_same_window = tracker.step_summary(
        _summary(max_race_distance=900.0, frames_run=1, entered_state_labels=("dash_pad_boost",)),
        _status(step_count=2),
        _telemetry(race_distance=900.0, state_labels=("active", "dash_pad_boost")),
    )
    rewarded_next_window = tracker.step_summary(
        _summary(max_race_distance=1100.0, frames_run=1, entered_state_labels=("dash_pad_boost",)),
        _status(step_count=3),
        _telemetry(race_distance=1100.0, state_labels=("active", "dash_pad_boost")),
    )

    assert first.breakdown["boost_pad"] == 0.5
    assert "boost_pad" not in blocked_same_window.breakdown
    assert rewarded_next_window.breakdown["boost_pad"] == 0.5
    info = tracker.info(_telemetry(race_distance=1100.0))
    assert info["rewarded_boost_pad_progress_windows"] == 2


def test_race_v3_blocks_dash_pad_boost_reward_while_reversing() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            boost_pad_reward=0.5,
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(
            max_race_distance=0.0,
            reverse_active_frames=1,
            entered_state_labels=("dash_pad_boost",),
        ),
        _status(step_count=1),
        _telemetry(race_distance=0.0, state_labels=("active", "dash_pad_boost")),
    )

    assert step.reward == 0.0
    assert step.breakdown == {}


def test_race_v3_rewards_airborne_landing_transition() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            airborne_landing_reward=3.0,
            time_penalty_per_frame=0.0,
            damage_taken_frame_penalty=0.0,
            damage_taken_streak_ramp_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, state_labels=("active", "airborne")))

    still_airborne = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1),
        _telemetry(race_distance=0.0, state_labels=("active", "airborne")),
    )
    landing = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=2),
        _telemetry(race_distance=0.0, state_labels=("active",)),
    )
    grounded = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=3),
        _telemetry(race_distance=0.0, state_labels=("active",)),
    )

    assert still_airborne.reward == 0.0
    assert still_airborne.breakdown == {}
    assert landing.reward == 3.0
    assert landing.breakdown == {"landing": 3.0}
    assert grounded.reward == 0.0
    assert grounded.breakdown == {}


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
