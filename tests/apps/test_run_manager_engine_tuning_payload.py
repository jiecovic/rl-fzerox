# tests/apps/test_run_manager_engine_tuning_payload.py
from __future__ import annotations

from rl_fzerox.apps.run_manager.api.payloads.engine_tuning import (
    engine_tuning_state_payload,
)
from rl_fzerox.core.engine_tuning import (
    ENGINE_TUNING_STATE_VERSION,
    BanditEngineTunerSettings,
    EngineTuningCandidateState,
    EngineTuningContext,
    EngineTuningRuntimeState,
    GaussianProcessEngineTunerSettings,
)

DEFAULT_BANDIT_BUCKETS = (0, 13, 26, 38, 51, 64, 77, 90, 102, 115, 128)
NARROW_BANDIT_BUCKETS = (44, 54, 64, 74, 84)


def test_engine_tuning_payload_reports_gp_backend_from_settings() -> None:
    context = EngineTuningContext(
        course_key="silence",
        vehicle_id="blue_falcon",
    )
    state = EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=1,
        candidates=(
            EngineTuningCandidateState(
                context_key=context.key,
                course_key=context.course_key,
                vehicle_id=context.vehicle_id,
                engine_setting_raw_value=40,
                finish_count=1,
                decayed_count=1.0,
                decayed_score_total=-90.0,
                score_total=-90.0,
                best_score=-90.0,
                best_time_ms=90_000,
            ),
        ),
    )

    payload = engine_tuning_state_payload(
        state,
        settings=GaussianProcessEngineTunerSettings(
            min_raw_value=40,
            max_raw_value=42,
            prior_finish_time_seconds=200.0,
        ),
    )

    contexts = payload["contexts"]
    assert payload["model_backend"] == "gaussian_process"
    assert isinstance(contexts, list)
    assert contexts[0]["observed_candidate_count"] == 1
    assert any(candidate["finish_count"] == 1 for candidate in contexts[0]["candidates"])


def test_engine_tuning_payload_reports_bandit_backend_from_settings() -> None:
    context = EngineTuningContext(
        course_key="silence",
        vehicle_id="blue_falcon",
    )
    state = EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=1,
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
                episode_count=2,
                return_count=2,
                completion_score_total=2.0,
                best_completion_score=1.0,
                finish_score_total=-200.0,
                best_finish_score=-90.0,
                return_score_total=30.0,
                best_return_score=20.0,
                best_time_ms=90_000,
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
                episode_count=1,
                return_count=1,
                completion_score_total=1.0,
                best_completion_score=1.0,
                finish_score_total=-80.0,
                best_finish_score=-80.0,
                return_score_total=10.0,
                best_return_score=10.0,
                best_time_ms=80_000,
            ),
        ),
    )

    payload = engine_tuning_state_payload(
        state,
        settings=BanditEngineTunerSettings(
            min_raw_value=44,
            max_raw_value=84,
            bucket_raw_values=NARROW_BANDIT_BUCKETS,
            prior_finish_time_seconds=200.0,
        ),
    )

    contexts = payload["contexts"]
    candidates = payload["candidates"]
    assert payload["model_backend"] == "bandit"
    assert isinstance(candidates, list)
    assert [candidate["engine_setting_raw_value"] for candidate in candidates] == [44, 84]
    assert isinstance(contexts, list)
    assert contexts[0]["observed_candidate_count"] == 2
    assert [candidate["engine_setting_raw_value"] for candidate in contexts[0]["candidates"]] == [
        44,
        54,
        64,
        74,
        84,
    ]
    bucket_44 = contexts[0]["candidates"][0]
    assert bucket_44["mean_finish_time_ms"] == 100_000
    assert bucket_44["mean_return_score"] == 15.0
    assert bucket_44["best_return_score"] == 20.0
    assert bucket_44["mean_completion_score"] == 1.0
    assert bucket_44["finish_rate"] == 1.0
    assert bucket_44["failure_rate"] == 0.0
    assert contexts[0]["recommended_engine_setting_raw_value"] == 84


def test_engine_tuning_payload_marks_inconsistent_candidate_rates_unknown() -> None:
    state = EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=1,
        candidates=(
            EngineTuningCandidateState(
                context_key="silence|blue_falcon",
                course_key="silence",
                vehicle_id="blue_falcon",
                engine_setting_raw_value=50,
                episode_count=1,
                finish_count=3,
                completion_score_total=4.0,
                best_completion_score=1.25,
            ),
        ),
    )

    payload = engine_tuning_state_payload(state)

    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    candidate = candidates[0]
    assert candidate["mean_completion_score"] is None
    assert candidate["best_completion_score"] == 1.0
    assert candidate["finish_rate"] is None
    assert candidate["failure_rate"] is None


def test_engine_tuning_payload_uses_bandit_bucket_recommendation() -> None:
    context = EngineTuningContext(
        course_key="silence",
        vehicle_id="blue_falcon",
    )
    state = EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=1,
        candidates=(
            EngineTuningCandidateState(
                context_key=context.key,
                course_key=context.course_key,
                vehicle_id=context.vehicle_id,
                engine_setting_raw_value=26,
                finish_count=1,
                decayed_count=1.0,
                decayed_score_total=-80.0,
                score_total=-80.0,
                best_score=-80.0,
                best_time_ms=80_000,
            ),
            EngineTuningCandidateState(
                context_key=context.key,
                course_key=context.course_key,
                vehicle_id=context.vehicle_id,
                engine_setting_raw_value=90,
                finish_count=1,
                decayed_count=1.0,
                decayed_score_total=-80.5,
                score_total=-80.5,
                best_score=-80.5,
                best_time_ms=80_500,
            ),
        ),
    )

    payload = engine_tuning_state_payload(
        state,
        settings=BanditEngineTunerSettings(
            min_raw_value=0,
            max_raw_value=128,
            bucket_raw_values=DEFAULT_BANDIT_BUCKETS,
            prior_finish_time_seconds=200.0,
        ),
    )

    contexts = payload["contexts"]
    assert isinstance(contexts, list)
    assert contexts[0]["recommended_engine_setting_raw_value"] == 26
