# tests/core/engine_tuning/test_ordered_tuner.py
from __future__ import annotations

import pytest

from rl_fzerox.core.engine_tuning import (
    ENGINE_TUNING_STATE_VERSION,
    BanditEngineTunerSettings,
    EngineTuningCandidateEstimate,
    EngineTuningCandidateState,
    EngineTuningContext,
    EngineTuningEpisodeOutcome,
    EngineTuningRuntimeState,
    GaussianProcessEngineTunerSettings,
    MlpEnsembleEngineTunerSettings,
    OrderedEngineTuner,
)
from rl_fzerox.core.engine_tuning.sampling import (
    StableGreedySelection,
    stable_greedy_engine_setting,
)
from rl_fzerox.core.engine_tuning.tuner import engine_candidates
from rl_fzerox.core.engine_tuning.types import engine_bucket_candidates


def test_engine_candidates_are_inclusive_and_clamped() -> None:
    assert engine_candidates(minimum=-10, maximum=3) == (0, 1, 2, 3)
    assert engine_candidates(minimum=126, maximum=140) == (126, 127, 128)


def test_engine_bucket_candidates_are_inclusive_centered_on_neutral_step() -> None:
    assert engine_bucket_candidates(minimum=0, maximum=128, slider_spacing=16) == (
        0,
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128,
    )
    assert engine_bucket_candidates(minimum=0, maximum=128, slider_spacing=32) == (
        0,
        32,
        64,
        96,
        128,
    )
    assert engine_bucket_candidates(minimum=0, maximum=128, slider_spacing=20) == (
        0,
        4,
        24,
        44,
        64,
        84,
        104,
        124,
        128,
    )
    assert engine_bucket_candidates(minimum=44, maximum=84, slider_spacing=10) == (
        44,
        54,
        64,
        74,
        84,
    )
    assert engine_bucket_candidates(minimum=26, maximum=102, slider_spacing=32) == (
        26,
        32,
        64,
        96,
        102,
    )
    assert engine_bucket_candidates(minimum=65, maximum=70, slider_spacing=10) == (65, 70)


def test_engine_candidates_reject_inverted_range() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        engine_candidates(minimum=80, maximum=20)


def test_stable_greedy_uses_soft_plateau_center() -> None:
    estimates = tuple(
        EngineTuningCandidateEstimate(
            engine_setting_raw_value=engine_raw,
            probability=0.0,
            mean_score=-90.0 if engine_raw != 58 else -89.95,
            uncertainty_score=0.0,
            estimated_finish_time_ms=90_000,
            finish_count=1,
            best_finish_time_ms=90_000,
        )
        for engine_raw in range(40, 61)
    )

    choice = stable_greedy_engine_setting(
        estimates,
        selection=StableGreedySelection(
            plateau_tolerance_seconds=1.0,
            boundary_softness_fraction=0.1,
        ),
    )

    assert choice in range(49, 52)


def test_ordered_tuner_recommends_lower_finish_time() -> None:
    context = EngineTuningContext(
        course_key="mute_city",
        vehicle_id="blue_falcon",
    )
    tuner = OrderedEngineTuner(
        settings=GaussianProcessEngineTunerSettings(
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


def test_mlp_ensemble_backend_records_model_state() -> None:
    context = EngineTuningContext(
        course_key="mute_city",
        vehicle_id="blue_falcon",
    )
    tuner = OrderedEngineTuner(
        settings=MlpEnsembleEngineTunerSettings(
            min_raw_value=40,
            max_raw_value=60,
            prior_finish_time_seconds=200.0,
            uniform_exploration=0.0,
        ),
    )

    state = tuner.record_many(
        (
            EngineTuningEpisodeOutcome(
                context=context,
                engine_setting_raw_value=40,
                completion_fraction=1.0,
                finished=True,
                race_time_ms=110_000,
            ),
            EngineTuningEpisodeOutcome(
                context=context,
                engine_setting_raw_value=60,
                completion_fraction=1.0,
                finished=True,
                race_time_ms=90_000,
            ),
        )
    )
    estimates = tuner.distribution(context, seed=123, draws=16)

    assert state.model_state is not None
    assert state.model_state.backend == "mlp_ensemble"
    assert state.candidates == ()
    assert state.model_state.contexts[0].context_key == context.key
    assert state.model_state.contexts[0].finish_count == 2
    assert len(estimates) == 21
    assert tuner.recommendation(context).engine_setting_raw_value in range(40, 61)


def test_bandit_backend_records_aggregate_candidates() -> None:
    context = EngineTuningContext(
        course_key="mute_city",
        vehicle_id="blue_falcon",
    )
    tuner = OrderedEngineTuner(
        settings=BanditEngineTunerSettings(
            min_raw_value=44,
            max_raw_value=84,
            slider_spacing=10,
            prior_finish_time_seconds=200.0,
            uniform_exploration=0.0,
        ),
    )

    state = tuner.record_many(
        (
            EngineTuningEpisodeOutcome(
                context=context,
                engine_setting_raw_value=44,
                completion_fraction=1.0,
                finished=True,
                race_time_ms=110_000,
            ),
            EngineTuningEpisodeOutcome(
                context=context,
                engine_setting_raw_value=84,
                completion_fraction=1.0,
                finished=True,
                race_time_ms=90_000,
            ),
        )
    )

    assert state.model_state is None
    assert set(state.candidate_map()) == {(context.key, 44), (context.key, 84)}
    assert tuner.recommendation(context).engine_setting_raw_value == 84


def test_bandit_recommendation_uses_best_observed_bucket() -> None:
    context = EngineTuningContext(
        course_key="mute_city",
        vehicle_id="blue_falcon",
    )
    tuner = OrderedEngineTuner(
        settings=BanditEngineTunerSettings(
            min_raw_value=0,
            max_raw_value=128,
            slider_spacing=13,
            prior_finish_time_seconds=200.0,
            uniform_exploration=0.0,
        ),
    )
    tuner.record_many(
        (
            EngineTuningEpisodeOutcome(
                context=context,
                engine_setting_raw_value=25,
                completion_fraction=1.0,
                finished=True,
                race_time_ms=80_000,
            ),
            EngineTuningEpisodeOutcome(
                context=context,
                engine_setting_raw_value=90,
                completion_fraction=1.0,
                finished=True,
                race_time_ms=80_500,
            ),
        )
    )

    choice = tuner.recommendation(context)

    assert choice.engine_setting_raw_value == 25
    assert choice.estimated_finish_time_ms == pytest.approx(80_000, abs=1)


def test_bandit_backend_explores_unobserved_buckets_before_resampling() -> None:
    context = EngineTuningContext(
        course_key="mute_city",
        vehicle_id="blue_falcon",
    )
    tuner = OrderedEngineTuner(
        settings=BanditEngineTunerSettings(
            min_raw_value=54,
            max_raw_value=74,
            slider_spacing=10,
            prior_finish_time_seconds=200.0,
            uniform_exploration=0.0,
        ),
    )
    tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=64,
            completion_fraction=1.0,
            finished=True,
            race_time_ms=95_000,
        )
    )

    probabilities = {
        estimate.engine_setting_raw_value: estimate.probability
        for estimate in tuner.distribution(context, seed=123, draws=128)
    }

    assert probabilities == {54: 0.5, 64: 0.0, 74: 0.5}


def test_bandit_backend_discards_off_grid_aggregate_candidates() -> None:
    context = EngineTuningContext(
        course_key="mute_city",
        vehicle_id="blue_falcon",
    )
    state = EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=2,
        candidates=(
            EngineTuningCandidateState(
                context_key=context.key,
                course_key=context.course_key,
                vehicle_id=context.vehicle_id,
                engine_setting_raw_value=44,
                finish_count=2,
                decayed_count=2.0,
                decayed_score_total=-200.0,
                score_total=-200.0,
                best_score=-90.0,
                best_time_ms=90_000,
            ),
            EngineTuningCandidateState(
                context_key=context.key,
                course_key=context.course_key,
                vehicle_id=context.vehicle_id,
                engine_setting_raw_value=50,
                finish_count=99,
                decayed_count=99.0,
                decayed_score_total=-7_920.0,
                score_total=-7_920.0,
                best_score=-70.0,
                best_time_ms=70_000,
            ),
            EngineTuningCandidateState(
                context_key=context.key,
                course_key=context.course_key,
                vehicle_id=context.vehicle_id,
                engine_setting_raw_value=67,
                finish_count=99,
                decayed_count=99.0,
                decayed_score_total=-7_920.0,
                score_total=-7_920.0,
                best_score=-70.0,
                best_time_ms=70_000,
            ),
            EngineTuningCandidateState(
                context_key=context.key,
                course_key=context.course_key,
                vehicle_id=context.vehicle_id,
                engine_setting_raw_value=84,
                finish_count=1,
                decayed_count=1.0,
                decayed_score_total=-80.0,
                score_total=-80.0,
                best_score=-80.0,
                best_time_ms=80_000,
            ),
        ),
    )
    tuner = OrderedEngineTuner(
        settings=BanditEngineTunerSettings(
            min_raw_value=44,
            max_raw_value=84,
            slider_spacing=10,
            prior_finish_time_seconds=200.0,
            uniform_exploration=0.0,
        ),
        state=state,
    )

    estimates = {
        estimate.engine_setting_raw_value: estimate
        for estimate in tuner.distribution(context, seed=123, draws=128)
    }

    assert estimates[44].finish_count == 2
    assert estimates[44].mean_score == pytest.approx(-100.0)
    assert estimates[64].finish_count == 0
    assert estimates[84].finish_count == 1
    assert tuner.recommendation(context).engine_setting_raw_value == 84
    assert [candidate.engine_setting_raw_value for candidate in tuner.state.candidates] == [44, 84]
    assert [candidate.finish_count for candidate in tuner.state.candidates] == [2, 1]


def test_mlp_ensemble_backend_does_not_warm_start_from_aggregate_candidates() -> None:
    context = EngineTuningContext(
        course_key="mute_city",
        vehicle_id="blue_falcon",
    )
    aggregate_state = EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=1,
        candidates=(
            EngineTuningCandidateState(
                context_key=context.key,
                course_key=context.course_key,
                vehicle_id=context.vehicle_id,
                engine_setting_raw_value=60,
                finish_count=1,
                decayed_count=1.0,
                decayed_score_total=-90.0,
                score_total=-90.0,
                best_score=-90.0,
                best_time_ms=90_000,
            ),
        ),
    )
    tuner = OrderedEngineTuner(
        settings=MlpEnsembleEngineTunerSettings(
            min_raw_value=40,
            max_raw_value=60,
            prior_finish_time_seconds=200.0,
            uniform_exploration=0.0,
        ),
        state=aggregate_state,
    )

    choice = tuner.recommendation(context)

    assert tuner.state.candidates == ()
    assert choice.finish_count == 0
    assert choice.mean_score == pytest.approx(-200.0)


def test_mlp_ensemble_backend_uses_configured_member_count() -> None:
    context = EngineTuningContext(
        course_key="mute_city",
        vehicle_id="blue_falcon",
    )
    tuner = OrderedEngineTuner(
        settings=MlpEnsembleEngineTunerSettings(
            min_raw_value=40,
            max_raw_value=60,
            prior_finish_time_seconds=200.0,
            ensemble_members=7,
        ),
    )

    state = tuner.record_many(
        (
            EngineTuningEpisodeOutcome(
                context=context,
                engine_setting_raw_value=40,
                completion_fraction=1.0,
                finished=True,
                race_time_ms=110_000,
            ),
            EngineTuningEpisodeOutcome(
                context=context,
                engine_setting_raw_value=60,
                completion_fraction=1.0,
                finished=True,
                race_time_ms=90_000,
            ),
        )
    )

    assert state.model_state is not None
    assert len(state.model_state.members) == 7


def test_mlp_ensemble_backend_keeps_model_state_separate_from_gp_aggregates() -> None:
    context = EngineTuningContext(
        course_key="mute_city",
        vehicle_id="blue_falcon",
    )
    tuner = OrderedEngineTuner(
        settings=MlpEnsembleEngineTunerSettings(
            min_raw_value=40,
            max_raw_value=60,
            prior_finish_time_seconds=200.0,
        ),
    )

    state = tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=40,
            completion_fraction=1.0,
            finished=True,
            race_time_ms=110_000,
        )
    )

    assert state.candidates == ()
    assert state.model_state is not None
    assert state.model_state.contexts[0].finish_count == 1


def test_mlp_ensemble_starts_near_uniform_after_one_success() -> None:
    context = EngineTuningContext(
        course_key="big_blue_2",
        vehicle_id="blue_falcon",
    )
    tuner = OrderedEngineTuner(
        settings=MlpEnsembleEngineTunerSettings(
            min_raw_value=0,
            max_raw_value=100,
            prior_finish_time_seconds=200.0,
            uniform_exploration=0.0,
        ),
    )

    tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=70,
            completion_fraction=1.0,
            finished=True,
            race_time_ms=75_000,
        )
    )

    probabilities = tuple(
        estimate.probability for estimate in tuner.distribution(context, seed=123, draws=512)
    )

    assert max(probabilities) < 0.011
    assert min(probabilities) > 0.009
    assert sum(probabilities) == pytest.approx(1.0)


def test_ordered_tuner_ignores_failed_engine_samples() -> None:
    context = EngineTuningContext(
        course_key="silence",
        vehicle_id="deep_claw",
    )
    tuner = OrderedEngineTuner(
        settings=GaussianProcessEngineTunerSettings(
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
        settings=GaussianProcessEngineTunerSettings(
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
        settings=GaussianProcessEngineTunerSettings(
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
        settings=GaussianProcessEngineTunerSettings(
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

    assert estimates[70].mean_score > estimates[50].mean_score
    assert estimates[80].mean_score > estimates[50].mean_score


def test_distribution_samples_correlated_posterior_functions() -> None:
    context = EngineTuningContext(
        course_key="big_blue_2",
        vehicle_id="blue_falcon",
    )
    tuner = OrderedEngineTuner(
        settings=GaussianProcessEngineTunerSettings(
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
