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
    REWARD_TRACKER_REGISTRY,
    RaceV2RewardTracker,
    RaceV2RewardWeights,
    RewardActionContext,
    build_reward_tracker,
    reward_tracker_names,
)
from tests.support.native_objects import make_step_summary, make_telemetry


def test_race_v2_yaml_keys_match_reward_config_schema() -> None:
    yaml_path = Path("conf/reward/race_v2.yaml")
    yaml_keys = set(yaml.safe_load(yaml_path.read_text(encoding="utf-8")))

    assert yaml_keys == set(RewardConfig.model_fields)


def test_race_v2_weight_fields_match_reward_config_schema() -> None:
    weight_fields = {field.name for field in fields(RaceV2RewardWeights)}

    assert weight_fields == set(RewardConfig.model_fields) - {"name"}


def test_reward_config_accepts_legacy_redundant_boost_penalty_key() -> None:
    config = RewardConfig.model_validate({"boost_redundant_press_penalty": -0.013})

    assert config.boost_press_penalty == -0.013


def test_build_reward_tracker_wires_all_race_v2_weight_fields() -> None:
    overrides = {
        "time_penalty_per_frame": -0.123,
        "reverse_time_penalty_scale": 1.25,
        "low_speed_time_penalty_scale": 1.5,
        "milestone_distance": 1234.0,
        "randomize_milestone_phase_on_reset": True,
        "milestone_bonus": 2.5,
        "milestone_speed_scale": 0.05,
        "milestone_speed_bonus_cap": 1.25,
        "bootstrap_progress_scale": 0.003,
        "bootstrap_regress_penalty_scale": 0.007,
        "bootstrap_position_multiplier_scale": 0.11,
        "bootstrap_lap_count": 2,
        "lap_1_completion_bonus": 21.0,
        "lap_2_completion_bonus": 34.0,
        "final_lap_completion_bonus": 89.0,
        "lap_position_scale": 0.33,
        "remaining_step_penalty_per_frame": 0.017,
        "remaining_lap_penalty": 43.0,
        "energy_loss_epsilon": 0.04,
        "energy_loss_penalty_scale": 0.12,
        "energy_loss_safe_fraction": 0.83,
        "energy_loss_danger_power": 2.75,
        "energy_gain_reward_scale": 0.018,
        "energy_gain_collision_cooldown_frames": 17,
        "energy_full_refill_bonus": 1.25,
        "energy_full_refill_cooldown_frames": 23,
        "airborne_landing_reward": 0.42,
        "grounded_air_brake_penalty": -0.014,
        "boost_pad_reward": 0.33,
        "boost_pad_reward_cooldown_frames": 19,
        "boost_press_penalty": -0.013,
        "collision_recoil_penalty": -2.25,
        "spinning_out_penalty": -4.5,
        "terminal_failure_base_penalty": -111.0,
        "stuck_truncation_base_penalty": -101.0,
        "wrong_way_truncation_base_penalty": -121.0,
        "progress_stalled_truncation_base_penalty": -131.0,
        "timeout_truncation_base_penalty": -102.0,
        "finish_position_scale": 3.5,
    }
    assert set(overrides) == {field.name for field in fields(RaceV2RewardWeights)}

    tracker = build_reward_tracker(RewardConfig(**overrides))

    assert isinstance(tracker, RaceV2RewardTracker)
    weights = tracker._weights
    actual = {field.name: getattr(weights, field.name) for field in fields(RaceV2RewardWeights)}
    assert actual == overrides


def test_race_v2_rewards_each_progress_milestone_once() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_distance=25.0,
            milestone_bonus=3.0,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    first = tracker.step_summary(
        _summary(max_race_distance=60.0),
        _status(step_count=1),
        _telemetry(race_distance=60.0),
    )
    repeated = tracker.step_summary(
        _summary(max_race_distance=70.0),
        _status(step_count=2),
        _telemetry(race_distance=70.0),
    )

    assert first.reward == pytest.approx(6.0)
    assert first.breakdown == {"milestone": 6.0}
    assert repeated.reward == 0.0
    assert repeated.breakdown == {}


def test_race_v2_adds_capped_milestone_speed_bonus() -> None:
    weights = RaceV2RewardWeights(
        time_penalty_per_frame=0.0,
        milestone_distance=10_000.0,
        milestone_bonus=2.0,
        milestone_speed_scale=0.05,
        milestone_speed_bonus_cap=2.0,
        bootstrap_progress_scale=0.0,
    )
    slow = RaceV2RewardTracker(weights)
    fast = RaceV2RewardTracker(weights)
    multi = RaceV2RewardTracker(weights)
    slow.reset(_telemetry(race_distance=0.0))
    fast.reset(_telemetry(race_distance=0.0))
    multi.reset(_telemetry(race_distance=0.0))

    slow_step = slow.step_summary(
        _summary(max_race_distance=10_000.0),
        _status(step_count=1_000),
        _telemetry(race_distance=10_000.0),
    )
    fast_step = fast.step_summary(
        _summary(max_race_distance=10_000.0),
        _status(step_count=250),
        _telemetry(race_distance=10_000.0),
    )
    multi_step = multi.step_summary(
        _summary(max_race_distance=20_000.0),
        _status(step_count=250),
        _telemetry(race_distance=20_000.0),
    )

    assert slow_step.breakdown == {"milestone": 2.0, "milestone_speed": 0.5}
    assert slow_step.reward == pytest.approx(2.5)
    assert fast_step.breakdown == {"milestone": 2.0, "milestone_speed": 2.0}
    assert fast_step.reward == pytest.approx(4.0)
    assert multi_step.breakdown == {"milestone": 4.0, "milestone_speed": 4.0}
    assert multi_step.reward == pytest.approx(8.0)


def test_race_v2_rewards_landing_once_when_airborne_clears() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            bootstrap_progress_scale=0.0,
            airborne_landing_reward=1.5,
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
        _telemetry(race_distance=0.0),
    )
    grounded = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=3),
        _telemetry(race_distance=0.0),
    )

    assert still_airborne.reward == 0.0
    assert still_airborne.breakdown == {}
    assert landing.reward == pytest.approx(1.5)
    assert landing.breakdown == {"landing": 1.5}
    assert grounded.reward == 0.0
    assert grounded.breakdown == {}


def test_race_v2_randomized_milestone_phase_shifts_first_threshold() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_distance=100.0,
            randomize_milestone_phase_on_reset=True,
            milestone_bonus=3.0,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0), episode_seed=1)

    info = tracker.info(_telemetry(race_distance=0.0))
    phase_offset_value = info["milestone_phase_offset"]
    assert isinstance(phase_offset_value, float)
    phase_offset = phase_offset_value

    before_threshold = max(phase_offset - 1.0, 0.0)
    before = tracker.step_summary(
        _summary(max_race_distance=before_threshold),
        _status(step_count=1),
        _telemetry(race_distance=before_threshold),
    )
    at_threshold = tracker.step_summary(
        _summary(max_race_distance=phase_offset + 1.0),
        _status(step_count=2),
        _telemetry(race_distance=phase_offset + 1.0),
    )

    assert 0.0 < phase_offset < 100.0
    assert before.reward == 0.0
    assert at_threshold.reward == pytest.approx(3.0)
    assert at_threshold.breakdown == {"milestone": 3.0}


def test_race_v2_measures_milestones_from_episode_start_progress() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_distance=3_000.0,
            milestone_bonus=2.0,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=-500.0))

    first = tracker.step_summary(
        _summary(max_race_distance=2_600.0),
        _status(step_count=1),
        _telemetry(race_distance=2_600.0),
    )
    repeated = tracker.step_summary(
        _summary(max_race_distance=2_900.0),
        _status(step_count=2),
        _telemetry(race_distance=2_900.0),
    )

    assert first.reward == pytest.approx(2.0)
    assert first.breakdown == {"milestone": 2.0}
    assert repeated.reward == 0.0
    assert repeated.breakdown == {}


def test_race_v2_initializes_progress_origin_from_first_in_race_sample() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_distance=3_000.0,
            milestone_bonus=2.0,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(None)

    first = tracker.step_summary(
        _summary(max_race_distance=2_600.0),
        _status(step_count=1),
        _telemetry(race_distance=2_600.0),
    )
    repeated = tracker.step_summary(
        _summary(max_race_distance=5_700.0),
        _status(step_count=2),
        _telemetry(race_distance=5_700.0),
    )

    assert first.reward == 0.0
    assert first.breakdown == {}
    assert repeated.reward == pytest.approx(2.0)
    assert repeated.breakdown == {"milestone": 2.0}


def test_race_v2_rewards_completed_laps() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            lap_1_completion_bonus=20.0,
            lap_2_completion_bonus=35.0,
            final_lap_completion_bonus=60.0,
            lap_position_scale=1.0,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, laps_completed=0))

    step = tracker.step_summary(
        _summary(max_race_distance=80_000.0),
        _status(step_count=200),
        _telemetry(race_distance=80_000.0, laps_completed=1, position=1),
    )

    assert step.reward == pytest.approx(49.0)
    assert step.breakdown == {
        "lap_completion": 20.0,
        "lap_position": 29.0,
    }


def test_race_v2_ignores_initial_start_line_crossing_for_completed_laps() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            bootstrap_progress_scale=0.1,
            bootstrap_lap_count=1,
            lap_1_completion_bonus=20.0,
            lap_position_scale=1.0,
        )
    )
    tracker.reset(_telemetry(race_distance=-100.0, lap=1, laps_completed=0))

    step = tracker.step_summary(
        _summary(max_race_distance=10.0),
        _status(step_count=20),
        _telemetry(race_distance=10.0, lap=1, laps_completed=1, position=1),
    )
    info = tracker.info(_telemetry(race_distance=10.0, lap=1, laps_completed=1))

    assert step.reward == pytest.approx(11.0)
    assert step.breakdown == {"bootstrap_progress": 11.0}
    assert info["bootstrap_progress_active"] is True
    assert info["race_laps_completed"] == 0
    assert info["rewarded_laps_completed"] == 0


def test_race_v2_scales_time_penalty_while_reverse_timer_is_active() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
            reverse_time_penalty_scale=2.0,
            milestone_bonus=0.0,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3, reverse_active_frames=2),
        _status(step_count=3, reverse_timer=20),
        _telemetry(race_distance=0.0, reverse_timer=20),
    )

    assert step.reward == pytest.approx(-0.05)
    assert step.breakdown == {
        "time": -0.03,
        "reverse_time": -0.02,
    }


def test_race_v2_scales_time_penalty_while_below_stuck_speed_threshold() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
            low_speed_time_penalty_scale=2.0,
            milestone_bonus=0.0,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, speed_kph=30.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3, low_speed_frames=2),
        _status(step_count=3, stalled_steps=2),
        _telemetry(race_distance=0.0, speed_kph=30.0),
    )

    assert step.reward == pytest.approx(-0.05)
    assert step.breakdown == {
        "time": -0.03,
        "low_speed_time": -0.02,
    }


def test_race_v2_penalizes_energy_loss_only_below_safe_fraction() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            energy_loss_epsilon=0.1,
            energy_loss_penalty_scale=0.10,
            energy_loss_safe_fraction=0.90,
            energy_loss_danger_power=2.0,
            energy_gain_reward_scale=0.02,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=178.0))

    safe_loss = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_loss_total=4.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=174.0),
    )
    dangerous_loss = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_loss_total=4.0),
        _status(step_count=2),
        _telemetry(race_distance=100.0, energy=89.0),
    )
    gain = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_gain_total=2.0),
        _status(step_count=3),
        _telemetry(race_distance=100.0, energy=176.0),
    )

    assert safe_loss.reward == 0.0
    assert safe_loss.breakdown == {}
    assert dangerous_loss.reward == pytest.approx(-0.0790123457)
    assert dangerous_loss.breakdown == {"energy_loss": pytest.approx(-0.0790123457)}
    assert gain.reward == pytest.approx(0.04)
    assert gain.breakdown == {"energy_gain": 0.04}


def test_race_v2_energy_gain_stays_net_negative_against_equal_loss() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            energy_loss_penalty_scale=0.05,
            energy_loss_safe_fraction=0.90,
            energy_loss_danger_power=2.0,
            energy_gain_reward_scale=0.02,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=178.0))

    step = tracker.step_summary(
        _summary(
            max_race_distance=100.0,
            energy_loss_total=4.0,
            energy_gain_total=4.0,
        ),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=0.0),
    )

    assert step.reward == pytest.approx(-0.12)
    assert step.breakdown == {
        "energy_loss": -0.2,
        "energy_gain": 0.08,
    }


def test_race_v2_suppresses_energy_gain_after_collision_recoil() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            energy_gain_reward_scale=0.1,
            energy_gain_collision_cooldown_frames=3,
            collision_recoil_penalty=-2.0,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=120.0))

    collision_step = tracker.step_summary(
        _summary(
            max_race_distance=100.0,
            frames_run=1,
            energy_gain_total=10.0,
            entered_state_labels=("collision_recoil",),
        ),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=130.0),
    )
    cooldown_step = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=2, energy_gain_total=10.0),
        _status(step_count=3),
        _telemetry(race_distance=100.0, energy=140.0),
    )
    reward_step = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=1, energy_gain_total=10.0),
        _status(step_count=4),
        _telemetry(race_distance=100.0, energy=150.0),
    )
    info = tracker.info(_telemetry(race_distance=100.0, energy=150.0))

    assert collision_step.reward == -2.0
    assert collision_step.breakdown == {"collision_recoil": -2.0}
    assert cooldown_step.reward == 0.0
    assert cooldown_step.breakdown == {}
    assert reward_step.reward == pytest.approx(1.0)
    assert reward_step.breakdown == {"energy_gain": 1.0}
    assert info["energy_gain_cooldown_frames_remaining"] == 0


def test_race_v2_rewards_full_energy_refill_once() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            energy_gain_reward_scale=0.0,
            energy_full_refill_bonus=1.5,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=120.0))

    partial = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_gain_total=20.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=140.0),
    )
    full = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_gain_total=38.0),
        _status(step_count=2),
        _telemetry(race_distance=100.0, energy=178.0),
    )
    still_full = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=3),
        _telemetry(race_distance=100.0, energy=178.0),
    )

    assert partial.reward == 0.0
    assert partial.breakdown == {}
    assert full.reward == pytest.approx(1.5)
    assert full.breakdown == {"energy_full_refill": 1.5}
    assert still_full.reward == 0.0
    assert still_full.breakdown == {}


def test_race_v2_full_energy_refill_bonus_obeys_cooldown() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            energy_gain_reward_scale=0.0,
            energy_full_refill_bonus=1.5,
            energy_full_refill_cooldown_frames=10,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=120.0))

    first_full = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=1, energy_gain_total=58.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=178.0),
    )
    tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=3, energy_loss_total=10.0),
        _status(step_count=2),
        _telemetry(race_distance=100.0, energy=168.0),
    )
    cooldown_blocked = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=3, energy_gain_total=10.0),
        _status(step_count=3),
        _telemetry(race_distance=100.0, energy=178.0),
    )
    tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=4, energy_loss_total=10.0),
        _status(step_count=4),
        _telemetry(race_distance=100.0, energy=168.0),
    )
    rewarded_after_cooldown = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=1, energy_gain_total=10.0),
        _status(step_count=5),
        _telemetry(race_distance=100.0, energy=178.0),
    )

    assert first_full.breakdown == {"energy_full_refill": 1.5}
    assert cooldown_blocked.breakdown == {}
    assert rewarded_after_cooldown.breakdown == {"energy_full_refill": 1.5}
    assert tracker.info(_telemetry(race_distance=100.0, energy=178.0))[
        "energy_full_refill_cooldown_frames_remaining"
    ] == 10


def test_build_reward_tracker_passes_energy_gain_collision_cooldown_from_config() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            time_penalty_per_frame=0.0,
            energy_gain_reward_scale=0.1,
            energy_gain_collision_cooldown_frames=2,
            collision_recoil_penalty=-2.0,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=120.0))

    step = tracker.step_summary(
        _summary(
            max_race_distance=100.0,
            frames_run=1,
            energy_gain_total=10.0,
            entered_state_labels=("collision_recoil",),
        ),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=130.0),
    )

    assert step.reward == -2.0
    assert step.breakdown == {"collision_recoil": -2.0}


def test_race_v2_penalizes_boost_requests() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            boost_press_penalty=-0.25,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, boost_timer=0))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, boost_timer=20),
        RewardActionContext(boost_requested=True),
    )

    assert step.reward == -0.25
    assert step.breakdown == {"boost_press": -0.25}


def test_race_v2_penalizes_boost_requests_while_boost_is_already_active() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            boost_press_penalty=-0.25,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, boost_timer=12))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, boost_timer=11),
        RewardActionContext(boost_requested=True),
    )

    assert step.reward == -0.25
    assert step.breakdown == {"boost_press": -0.25}


def test_race_v2_does_not_penalize_when_boost_is_not_requested() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            boost_press_penalty=-0.25,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, boost_timer=0))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, boost_timer=20),
        RewardActionContext(boost_requested=False),
    )

    assert step.reward == 0.0
    assert step.breakdown == {}


def test_race_v2_penalizes_grounded_air_brake_requests() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            grounded_air_brake_penalty=-0.125,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0),
        RewardActionContext(air_brake_requested=True),
    )

    assert step.reward == -0.125
    assert step.breakdown == {"grounded_air_brake": -0.125}


def test_race_v2_does_not_penalize_air_brake_requests_while_airborne() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            grounded_air_brake_penalty=-0.125,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, state_labels=("active", "airborne")))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, state_labels=("active", "airborne")),
        RewardActionContext(air_brake_requested=True),
    )

    assert step.reward == 0.0
    assert step.breakdown == {}


def test_race_v2_rewards_dash_pad_boost_entries_with_cooldown() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            boost_pad_reward=0.5,
            boost_pad_reward_cooldown_frames=3,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    first = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=1, entered_state_labels=("dash_pad_boost",)),
        _status(step_count=1),
        _telemetry(race_distance=100.0, state_labels=("active", "dash_pad_boost")),
    )
    blocked = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=1, entered_state_labels=("dash_pad_boost",)),
        _status(step_count=2),
        _telemetry(race_distance=100.0, state_labels=("active", "dash_pad_boost")),
    )
    rewarded_after_cooldown = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=2, entered_state_labels=("dash_pad_boost",)),
        _status(step_count=4),
        _telemetry(race_distance=100.0, state_labels=("active", "dash_pad_boost")),
    )
    info = tracker.info(_telemetry(race_distance=100.0))

    assert first.reward == 0.5
    assert first.breakdown == {"boost_pad": 0.5}
    assert blocked.reward == 0.0
    assert blocked.breakdown == {}
    assert rewarded_after_cooldown.reward == 0.5
    assert rewarded_after_cooldown.breakdown == {"boost_pad": 0.5}
    assert info["boost_pad_reward_cooldown_frames_remaining"] == 3


def test_race_v2_applies_event_penalties_once_per_entry() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            collision_recoil_penalty=-2.0,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    first = tracker.step_summary(
        _summary(
            max_race_distance=100.0,
            entered_state_labels=("collision_recoil",),
        ),
        _status(step_count=1),
        _telemetry(
            race_distance=100.0,
            state_labels=("active", "collision_recoil"),
        ),
    )
    repeated = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=2),
        _telemetry(
            race_distance=100.0,
            state_labels=("active", "collision_recoil"),
        ),
    )

    assert first.reward == -2.0
    assert first.breakdown == {"collision_recoil": -2.0}
    assert repeated.reward == 0.0
    assert repeated.breakdown == {}


def test_race_v2_scales_terminal_failure_penalty_by_remaining_steps_and_laps() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
            milestone_bonus=0.0,
            lap_1_completion_bonus=0.0,
            lap_2_completion_bonus=0.0,
            final_lap_completion_bonus=0.0,
            lap_position_scale=0.0,
            remaining_step_penalty_per_frame=0.01,
            remaining_lap_penalty=50.0,
            terminal_failure_base_penalty=-120.0,
            bootstrap_progress_scale=0.0,
        ),
        max_episode_steps=100,
    )
    tracker.reset(_telemetry(race_distance=0.0, laps_completed=0))

    early = tracker.step_summary(
        _summary(max_race_distance=10_000.0),
        _status(step_count=10, termination_reason="crashed"),
        _telemetry(race_distance=10_000.0, laps_completed=0, state_labels=("crashed",)),
    )
    late = tracker.step_summary(
        _summary(max_race_distance=150_000.0),
        _status(step_count=90, termination_reason="crashed"),
        _telemetry(race_distance=150_000.0, laps_completed=2, state_labels=("crashed",)),
    )

    assert early.reward == pytest.approx(-270.91)
    assert early.breakdown["crashed"] == pytest.approx(-270.9)
    assert late.reward == pytest.approx(-170.11)
    assert late.breakdown["crashed"] == pytest.approx(-170.1)
    assert early.reward < late.reward


def test_race_v2_treats_energy_depletion_as_terminal_failure() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            lap_1_completion_bonus=0.0,
            lap_2_completion_bonus=0.0,
            final_lap_completion_bonus=0.0,
            lap_position_scale=0.0,
            remaining_step_penalty_per_frame=0.0,
            remaining_lap_penalty=0.0,
            terminal_failure_base_penalty=-120.0,
            bootstrap_progress_scale=0.0,
        ),
        max_episode_steps=100,
    )
    tracker.reset(_telemetry(race_distance=0.0, laps_completed=0))

    result = tracker.step_summary(
        _summary(max_race_distance=10_000.0),
        _status(step_count=10, termination_reason="energy_depleted"),
        _telemetry(race_distance=10_000.0, energy=0.0),
    )

    assert result.reward == pytest.approx(-120.0)
    assert result.breakdown["energy_depleted"] == pytest.approx(-120.0)


def test_race_v2_scales_truncation_penalty_by_remaining_steps_and_laps() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
            milestone_bonus=0.0,
            lap_1_completion_bonus=0.0,
            lap_2_completion_bonus=0.0,
            final_lap_completion_bonus=0.0,
            lap_position_scale=0.0,
            remaining_step_penalty_per_frame=0.01,
            remaining_lap_penalty=50.0,
            wrong_way_truncation_base_penalty=-170.0,
            bootstrap_progress_scale=0.0,
        ),
        max_episode_steps=100,
    )
    tracker.reset(_telemetry(race_distance=0.0, laps_completed=0))

    early = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=10, reverse_timer=100, truncation_reason="wrong_way"),
        _telemetry(race_distance=0.0, laps_completed=0),
    )
    late = tracker.step_summary(
        _summary(max_race_distance=140_000.0),
        _status(step_count=90, reverse_timer=100, truncation_reason="wrong_way"),
        _telemetry(race_distance=140_000.0, laps_completed=2),
    )

    assert early.breakdown["wrong_way_truncation"] == pytest.approx(-320.9)
    assert late.breakdown["wrong_way_truncation"] == pytest.approx(-220.1)
    assert early.reward == pytest.approx(-320.91)
    assert late.reward == pytest.approx(-220.11)
    assert early.reward < late.reward


def test_race_v2_supports_progress_stalled_truncation_penalty() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            lap_1_completion_bonus=0.0,
            lap_2_completion_bonus=0.0,
            final_lap_completion_bonus=0.0,
            lap_position_scale=0.0,
            remaining_step_penalty_per_frame=0.0,
            remaining_lap_penalty=0.0,
            progress_stalled_truncation_base_penalty=-99.0,
            bootstrap_progress_scale=0.0,
        ),
        max_episode_steps=100,
    )
    tracker.reset(_telemetry(race_distance=0.0, laps_completed=0))

    result = tracker.step_summary(
        _summary(max_race_distance=10_000.0),
        _status(step_count=25, truncation_reason="progress_stalled"),
        _telemetry(race_distance=10_000.0, laps_completed=0),
    )

    assert result.reward == pytest.approx(-99.0)
    assert result.breakdown["progress_stalled_truncation"] == pytest.approx(-99.0)


def test_race_v2_applies_final_lap_reward_and_finish_position_bonus() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
            milestone_bonus=0.0,
            lap_1_completion_bonus=20.0,
            lap_2_completion_bonus=35.0,
            final_lap_completion_bonus=60.0,
            lap_position_scale=0.0,
            finish_position_scale=4.0,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, state_labels=("active",), laps_completed=2))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0, entered_state_labels=("finished",)),
        _status(step_count=120, termination_reason="finished"),
        _telemetry(
            race_distance=100.0,
            state_labels=("active", "finished"),
            position=1,
            laps_completed=3,
        ),
    )

    assert step.reward == pytest.approx(175.99)
    assert step.breakdown == {
        "time": -0.01,
        "lap_completion": 60.0,
        "finish_position": 116.0,
    }


def test_race_v2_applies_low_speed_time_penalty_when_dash_pad_reward_is_disabled() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
            low_speed_time_penalty_scale=2.0,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, speed_kph=30.0))

    step = tracker.step_summary(
        _summary(
            max_race_distance=100.0,
            low_speed_frames=1,
            entered_state_labels=("dash_pad_boost",),
        ),
        _status(step_count=1),
        _telemetry(
            race_distance=100.0,
            speed_kph=30.0,
            state_labels=("active", "dash_pad_boost"),
        ),
    )

    assert step.reward == -0.02
    assert step.breakdown == {
        "time": -0.01,
        "low_speed_time": -0.01,
    }


def test_reward_tracker_registry_exposes_registered_names() -> None:
    assert DEFAULT_REWARD_NAME == "race_v2"
    assert reward_tracker_names() == tuple(REWARD_TRACKER_REGISTRY)
    assert isinstance(build_reward_tracker(), RaceV2RewardTracker)


def test_race_v2_multiplies_time_penalty_by_frames_run() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
            bootstrap_progress_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=100.0),
    )

    assert step.reward == pytest.approx(-0.03)
    assert step.breakdown == {"time": -0.03}


def test_race_v2_bootstrap_progress_applies_until_configured_lap_is_completed() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_distance=25.0,
            milestone_bonus=0.0,
            bootstrap_progress_scale=0.1,
            bootstrap_lap_count=1,
            lap_1_completion_bonus=0.0,
            lap_position_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    first = tracker.step_summary(
        _summary(max_race_distance=10.0),
        _status(step_count=1),
        _telemetry(race_distance=10.0),
    )
    crossing = tracker.step_summary(
        _summary(max_race_distance=30.0),
        _status(step_count=2),
        _telemetry(race_distance=30.0),
    )
    later_same_lap = tracker.step_summary(
        _summary(max_race_distance=40.0),
        _status(step_count=3),
        _telemetry(race_distance=40.0),
    )
    lap_crossing = tracker.step_summary(
        _summary(max_race_distance=80.0),
        _status(step_count=4),
        _telemetry(race_distance=80.0, laps_completed=1),
    )
    after_lap = tracker.step_summary(
        _summary(max_race_distance=90.0),
        _status(step_count=5),
        _telemetry(race_distance=90.0, laps_completed=1),
    )

    assert first.reward == pytest.approx(1.0)
    assert first.breakdown == {"bootstrap_progress": 1.0}
    assert crossing.reward == pytest.approx(2.0)
    assert crossing.breakdown == {"bootstrap_progress": 2.0}
    assert later_same_lap.reward == pytest.approx(1.0)
    assert later_same_lap.breakdown == {"bootstrap_progress": 1.0}
    assert lap_crossing.reward == pytest.approx(4.0)
    assert lap_crossing.breakdown == {"bootstrap_progress": 4.0}
    assert after_lap.reward == 0.0
    assert after_lap.breakdown == {}


def test_race_v2_bootstrap_progress_rewards_recovery_and_penalizes_regress() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_distance=1_000.0,
            milestone_bonus=0.0,
            bootstrap_progress_scale=0.1,
            bootstrap_regress_penalty_scale=0.2,
            bootstrap_lap_count=3,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    forward = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0),
    )
    backward = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=2),
        _telemetry(race_distance=60.0),
    )
    recovery = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=3),
        _telemetry(race_distance=90.0),
    )
    new_progress = tracker.step_summary(
        _summary(max_race_distance=110.0),
        _status(step_count=4),
        _telemetry(race_distance=110.0),
    )

    assert forward.reward == pytest.approx(10.0)
    assert forward.breakdown == {"bootstrap_progress": 10.0}
    assert backward.reward == pytest.approx(-8.0)
    assert backward.breakdown == {"bootstrap_regress": -8.0}
    assert recovery.reward == pytest.approx(3.0)
    assert recovery.breakdown == {"bootstrap_progress": 3.0}
    assert new_progress.reward == pytest.approx(2.0)
    assert new_progress.breakdown == {"bootstrap_progress": 2.0}


def test_race_v2_bootstrap_progress_can_span_multiple_initial_laps() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_distance=25.0,
            milestone_bonus=0.0,
            bootstrap_progress_scale=0.1,
            bootstrap_lap_count=2,
            lap_1_completion_bonus=0.0,
            lap_2_completion_bonus=0.0,
            lap_position_scale=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    first = tracker.step_summary(
        _summary(max_race_distance=10.0),
        _status(step_count=1),
        _telemetry(race_distance=10.0),
    )
    second = tracker.step_summary(
        _summary(max_race_distance=30.0),
        _status(step_count=2),
        _telemetry(race_distance=30.0),
    )
    lap_1_crossing = tracker.step_summary(
        _summary(max_race_distance=45.0),
        _status(step_count=3),
        _telemetry(race_distance=45.0, laps_completed=1),
    )
    lap_2_progress = tracker.step_summary(
        _summary(max_race_distance=60.0),
        _status(step_count=4),
        _telemetry(race_distance=60.0, laps_completed=1),
    )
    lap_2_crossing = tracker.step_summary(
        _summary(max_race_distance=70.0),
        _status(step_count=5),
        _telemetry(race_distance=70.0, laps_completed=2),
    )
    after_lap_2 = tracker.step_summary(
        _summary(max_race_distance=80.0),
        _status(step_count=6),
        _telemetry(race_distance=80.0, laps_completed=2),
    )

    assert first.reward == pytest.approx(1.0)
    assert first.breakdown == {"bootstrap_progress": 1.0}
    assert second.reward == pytest.approx(2.0)
    assert second.breakdown == {"bootstrap_progress": 2.0}
    assert lap_1_crossing.reward == pytest.approx(1.5)
    assert lap_1_crossing.breakdown == {"bootstrap_progress": 1.5}
    assert lap_2_progress.reward == pytest.approx(1.5)
    assert lap_2_progress.breakdown == {"bootstrap_progress": 1.5}
    assert lap_2_crossing.reward == pytest.approx(1.0)
    assert lap_2_crossing.breakdown == {"bootstrap_progress": 1.0}
    assert after_lap_2.reward == 0.0
    assert after_lap_2.breakdown == {}


def test_race_v2_bootstrap_progress_uses_episode_start_as_zero() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_distance=1_000.0,
            milestone_bonus=0.0,
            bootstrap_progress_scale=0.1,
            bootstrap_lap_count=1,
        )
    )
    tracker.reset(_telemetry(race_distance=-300.0))

    first = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0),
    )
    crossing = tracker.step_summary(
        _summary(max_race_distance=1_200.0),
        _status(step_count=2),
        _telemetry(race_distance=1_200.0),
    )

    assert first.reward == pytest.approx(40.0)
    assert first.breakdown == {"bootstrap_progress": 40.0}
    assert crossing.reward == pytest.approx(110.0)
    assert crossing.breakdown == {"bootstrap_progress": 110.0}


def test_race_v2_scales_bootstrap_progress_by_race_position() -> None:
    weights = RaceV2RewardWeights(
        time_penalty_per_frame=0.0,
        milestone_bonus=0.0,
        bootstrap_progress_scale=0.01,
        bootstrap_regress_penalty_scale=0.02,
        bootstrap_position_multiplier_scale=0.5,
    )
    last_place = RaceV2RewardTracker(weights)
    first_place = RaceV2RewardTracker(weights)
    last_place.reset(_telemetry(race_distance=0.0, position=30))
    first_place.reset(_telemetry(race_distance=0.0, position=1))

    last_place_forward = last_place.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, position=30),
    )
    first_place_forward = first_place.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, position=1),
    )
    first_place_backtrack = first_place.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=2),
        _telemetry(race_distance=50.0, position=1),
    )
    info = first_place.info(_telemetry(race_distance=50.0, position=1))

    assert last_place_forward.breakdown == {"bootstrap_progress": 1.0}
    assert first_place_forward.breakdown == {"bootstrap_progress": 1.5}
    assert first_place_backtrack.breakdown == {"bootstrap_regress": -1.5}
    assert info["bootstrap_position_multiplier"] == pytest.approx(1.5)


def test_build_reward_tracker_passes_bootstrap_lap_count_from_config() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            time_penalty_per_frame=0.0,
            bootstrap_lap_count=2,
            bootstrap_progress_scale=0.1,
            bootstrap_regress_penalty_scale=0.3,
            bootstrap_position_multiplier_scale=0.5,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, position=1))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1),
        _telemetry(race_distance=-10.0, position=1),
    )
    info = tracker.info(_telemetry(race_distance=0.0, position=1))

    assert info["bootstrap_lap_count"] == 2
    assert step.breakdown == {"bootstrap_regress": -4.5}


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
) -> FZeroXTelemetry:
    return make_telemetry(
        race_distance=race_distance,
        state_labels=state_labels,
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
    entered_state_labels: tuple[str, ...] = (),
) -> StepSummary:
    return make_step_summary(
        frames_run=frames_run,
        max_race_distance=max_race_distance,
        reverse_active_frames=reverse_active_frames,
        low_speed_frames=low_speed_frames,
        energy_loss_total=energy_loss_total,
        energy_gain_total=energy_gain_total,
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
