# tests/core/engine_tuning/test_ordered_tuner.py
from __future__ import annotations

import pytest

from rl_fzerox.core.engine_tuning import (
    EngineTunerSettings,
    EngineTuningContext,
    EngineTuningEpisodeOutcome,
    OrderedEngineTuner,
)
from rl_fzerox.core.engine_tuning.tuner import engine_candidates


def test_engine_candidates_are_inclusive_and_clamped() -> None:
    assert engine_candidates(minimum=-10, maximum=3) == (0, 1, 2, 3)
    assert engine_candidates(minimum=98, maximum=120) == (98, 99, 100)


def test_engine_candidates_reject_inverted_range() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        engine_candidates(minimum=80, maximum=20)


def test_ordered_tuner_recommends_lower_finish_time() -> None:
    context = EngineTuningContext(
        course_key="mute_city",
        vehicle_id="blue_falcon",
    )
    tuner = OrderedEngineTuner(
        settings=EngineTunerSettings(
            min_raw_value=40,
            max_raw_value=60,
            prior_finish_time_seconds=200.0,
            observation_noise_seconds=0.25,
            curve_lengthscale_raw=1.0,
            uniform_exploration=0.0,
        ),
    )

    tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=40,
            completion_fraction=1.0,
            finished=True,
            race_time_ms=110_000,
            finish_position=1,
            total_racers=30,
        )
    )
    tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=60,
            completion_fraction=1.0,
            finished=True,
            race_time_ms=90_000,
            finish_position=1,
            total_racers=30,
        )
    )

    choice = tuner.recommendation(context)

    assert choice.engine_setting_raw_value == 60
    assert choice.finish_count == 1
    assert choice.mean_score == pytest.approx(-90.0, abs=0.01)
    assert choice.estimated_finish_time_ms == pytest.approx(90_000, abs=10)


def test_ordered_tuner_ignores_failed_engine_samples() -> None:
    context = EngineTuningContext(
        course_key="silence",
        vehicle_id="deep_claw",
    )
    tuner = OrderedEngineTuner(
        settings=EngineTunerSettings(
            min_raw_value=50,
            max_raw_value=50,
            prior_finish_time_seconds=200.0,
            uniform_exploration=0.0,
        )
    )

    state = tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=50,
            completion_fraction=0.8,
            finished=False,
            race_time_ms=80_000,
            finish_position=1,
            total_racers=30,
        )
    )

    assert state.update_count == 0
    assert state.candidates == ()


def test_discounted_state_prefers_recent_observations() -> None:
    context = EngineTuningContext(
        course_key="silence",
        vehicle_id="deep_claw",
    )
    tuner = OrderedEngineTuner(
        settings=EngineTunerSettings(
            min_raw_value=50,
            max_raw_value=50,
            stat_decay=0.5,
            prior_finish_time_seconds=200.0,
            observation_noise_seconds=0.25,
            curve_lengthscale_raw=1.0,
        )
    )
    tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=50,
            completion_fraction=0.0,
            finished=True,
            race_time_ms=120_000,
        )
    )
    tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=50,
            completion_fraction=1.0,
            finished=True,
            race_time_ms=60_000,
        )
    )

    choice = tuner.recommendation(context)

    assert choice.mean_score == pytest.approx(-80.0, abs=0.01)


def test_discounting_applies_to_all_candidates_on_new_success() -> None:
    context = EngineTuningContext(
        course_key="silence",
        vehicle_id="deep_claw",
    )
    tuner = OrderedEngineTuner(
        settings=EngineTunerSettings(
            min_raw_value=40,
            max_raw_value=60,
            stat_decay=0.5,
            prior_finish_time_seconds=200.0,
        )
    )
    tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=40,
            completion_fraction=1.0,
            finished=True,
            race_time_ms=120_000,
        )
    )
    tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=60,
            completion_fraction=1.0,
            finished=True,
            race_time_ms=80_000,
        )
    )

    candidates = tuner.state.candidate_map()

    assert candidates[(context.key, 40)].decayed_count == pytest.approx(0.5)
    assert candidates[(context.key, 40)].finish_count == 1
    assert candidates[(context.key, 60)].decayed_count == pytest.approx(1.0)


def test_ordered_tuner_shares_information_with_nearby_candidates() -> None:
    context = EngineTuningContext(
        course_key="silence",
        vehicle_id="deep_claw",
    )
    tuner = OrderedEngineTuner(
        settings=EngineTunerSettings(
            min_raw_value=50,
            max_raw_value=90,
            prior_finish_time_seconds=200.0,
            observation_noise_seconds=1.0,
            curve_lengthscale_raw=20.0,
            exploration_seconds=0.0,
            uniform_exploration=0.0,
        )
    )
    tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=70,
            completion_fraction=1.0,
            finished=True,
            race_time_ms=80_000,
        )
    )

    estimates = {
        estimate.engine_setting_raw_value: estimate
        for estimate in tuner.distribution(context, seed=123, draws=16)
    }

    assert estimates[70].posterior_mean > estimates[50].posterior_mean
    assert estimates[80].posterior_mean > estimates[50].posterior_mean


def test_distribution_samples_correlated_posterior_functions() -> None:
    context = EngineTuningContext(
        course_key="big_blue_2",
        vehicle_id="blue_falcon",
    )
    tuner = OrderedEngineTuner(
        settings=EngineTunerSettings(
            min_raw_value=60,
            max_raw_value=80,
            prior_finish_time_seconds=200.0,
            observation_noise_seconds=1.0,
            curve_lengthscale_raw=10.0,
            exploration_seconds=30.0,
            uniform_exploration=0.0,
        )
    )
    tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=70,
            completion_fraction=1.0,
            finished=True,
            race_time_ms=70_000,
        )
    )

    estimates = tuple(tuner.distribution(context, seed=123, draws=128))
    peak = max(estimates, key=lambda estimate: estimate.probability)

    assert 60 < peak.engine_setting_raw_value < 80
    assert estimates[0].probability < peak.probability
    assert estimates[-1].probability < peak.probability
