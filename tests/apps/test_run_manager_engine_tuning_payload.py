# tests/apps/test_run_manager_engine_tuning_payload.py
from __future__ import annotations

from rl_fzerox.apps.run_manager.api.payloads.engine_tuning import (
    engine_tuning_state_payload,
)
from rl_fzerox.core.engine_tuning import (
    ENGINE_TUNING_STATE_VERSION,
    EngineTuningCandidateState,
    EngineTuningContext,
    EngineTuningRuntimeState,
    GaussianProcessEngineTunerSettings,
)


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
