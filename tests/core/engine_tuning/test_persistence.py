# tests/core/engine_tuning/test_persistence.py
from __future__ import annotations

import json
from pathlib import Path

from rl_fzerox.core.engine_tuning import (
    EngineTuningContext,
    EngineTuningEpisodeOutcome,
    MlpEnsembleEngineTunerSettings,
    OrderedEngineTuner,
)
from rl_fzerox.core.engine_tuning.persistence import (
    load_engine_tuning_runtime_state,
    load_engine_tuning_runtime_state_json,
    save_engine_tuning_runtime_state,
)


def test_loader_rejects_unsupported_state_version() -> None:
    data = json.dumps(
        {
            "version": 4,
            "update_count": 1,
            "arms": [
                {
                    "context_key": "silence|blue_falcon",
                    "course_key": "silence",
                    "vehicle_id": "blue_falcon",
                    "engine_setting_raw_value": 40,
                    "attempts": 1,
                    "finished_attempts": 1,
                    "decayed_count": 1.0,
                    "decayed_score_total": 2.25,
                    "completion_total": 1.0,
                    "score_total": 2.25,
                    "best_score": 2.25,
                    "best_time_ms": 90_000,
                }
            ],
        }
    )

    assert load_engine_tuning_runtime_state_json(data) is None


def test_loader_rejects_legacy_state_versions_after_data_migration() -> None:
    data = json.dumps(
        {
            "version": 5,
            "update_count": 1,
            "candidates": [
                {
                    "context_key": "silence|blue_falcon",
                    "course_key": "silence",
                    "vehicle_id": "blue_falcon",
                    "engine_setting_raw_value": 50,
                    "finish_count": 1,
                }
            ],
        }
    )

    assert load_engine_tuning_runtime_state_json(data) is None


def test_mlp_model_weights_round_trip_through_pt_sidecar(tmp_path: Path) -> None:
    context = EngineTuningContext(
        course_key="big_blue_2",
        vehicle_id="blue_falcon",
    )
    tuner = OrderedEngineTuner(
        settings=MlpEnsembleEngineTunerSettings(
            min_raw_value=0,
            max_raw_value=128,
            prior_finish_time_seconds=200.0,
            uniform_exploration=0.0,
        ),
    )
    state = tuner.record(
        EngineTuningEpisodeOutcome(
            context=context,
            engine_setting_raw_value=70,
            completion_fraction=1.0,
            finished=True,
            race_time_ms=75_000,
        )
    )
    state_path = tmp_path / "engine_tuning_state.json"
    model_path = tmp_path / "engine_tuning_model.pt"

    save_engine_tuning_runtime_state(state_path, state, model_path=model_path)
    loaded = load_engine_tuning_runtime_state(state_path, model_path=model_path)

    assert model_path.is_file()
    assert loaded is not None
    assert loaded.model_state is not None
    assert loaded.model_state.backend == "mlp_ensemble"
    assert loaded.model_state.members
    assert loaded.model_state.contexts[0].finish_count == 1
