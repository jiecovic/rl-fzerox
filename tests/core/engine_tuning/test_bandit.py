# tests/core/engine_tuning/test_bandit.py
from __future__ import annotations

import pytest

from rl_fzerox.core.engine_tuning import (
    AdaptiveEngineBandit,
    EngineBanditSettings,
    EngineTuningContext,
    EngineTuningEpisodeOutcome,
)
from rl_fzerox.core.engine_tuning.bandit import engine_bins


def test_engine_bins_are_inclusive_and_clamped() -> None:
    assert engine_bins(minimum=-10, maximum=13, bin_size=5) == (0, 5, 10, 13)
    assert engine_bins(minimum=95, maximum=120, bin_size=10) == (95, 100)


def test_engine_bins_reject_inverted_range() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        engine_bins(minimum=80, maximum=20, bin_size=5)


def test_bandit_recommends_observed_better_engine() -> None:
    context = EngineTuningContext(
        course_key="mute_city",
        vehicle_id="blue_falcon",
    )
    bandit = AdaptiveEngineBandit(
        settings=EngineBanditSettings(
            min_raw_value=40,
            max_raw_value=60,
            bin_size=10,
            prior_mean=0.0,
            prior_strength=0.0,
            uniform_exploration=0.0,
        ),
    )

    bandit.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=40,
            completion_fraction=0.25,
            finished=False,
        )
    )
    bandit.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=60,
            completion_fraction=1.0,
            finished=True,
            finish_position=1,
            total_racers=30,
        )
    )

    choice = bandit.recommendation(context)

    assert choice.engine_setting_raw_value == 60
    assert choice.attempts == 1
    assert choice.mean_score > 1.0


def test_discounted_state_prefers_recent_observations() -> None:
    context = EngineTuningContext(
        course_key="silence",
        vehicle_id="deep_claw",
    )
    bandit = AdaptiveEngineBandit(
        settings=EngineBanditSettings(
            min_raw_value=50,
            max_raw_value=50,
            stat_decay=0.5,
            prior_mean=0.0,
            prior_strength=0.0,
            finish_bonus=0.0,
            position_weight=0.0,
        )
    )
    bandit.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=50,
            completion_fraction=0.0,
            finished=False,
        )
    )
    bandit.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=50,
            completion_fraction=1.0,
            finished=False,
        )
    )

    choice = bandit.recommendation(context)

    assert choice.mean_score == pytest.approx(2.0 / 3.0)
