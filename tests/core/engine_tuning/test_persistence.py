# tests/core/engine_tuning/test_persistence.py
from __future__ import annotations

import json

from rl_fzerox.core.engine_tuning import ENGINE_TUNING_STATE_VERSION
from rl_fzerox.core.engine_tuning.persistence import load_engine_tuning_runtime_state_json


def test_loader_rejects_previous_completion_scored_state_version() -> None:
    data = json.dumps(
        {
            "version": ENGINE_TUNING_STATE_VERSION - 1,
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
