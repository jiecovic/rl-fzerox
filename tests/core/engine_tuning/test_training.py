# tests/core/engine_tuning/test_training.py
from __future__ import annotations

from typing import Literal

import pytest

from rl_fzerox.core.engine_tuning.training import (
    EngineTuningTrainingController,
    engine_tuning_outcome_from_episode,
)
from rl_fzerox.core.runtime_spec.schema import AdaptiveEngineTuningConfig


@pytest.mark.parametrize("backend", ["gaussian_process", "mlp_ensemble"])
def test_engine_tuning_controller_ignores_alt_baseline_episodes(
    backend: Literal["gaussian_process", "mlp_ensemble"],
) -> None:
    controller = EngineTuningTrainingController(
        AdaptiveEngineTuningConfig(
            enabled=True,
            backend=backend,
            min_raw_value=40,
            max_raw_value=60,
            ensemble_members=1,
            hidden_dim=4,
            training_steps=1,
            warmup_successes=1,
        )
    )

    assert not controller.record_episodes((_successful_engine_episode(alt_baseline_id="alt-a"),))
    assert not controller.record_rollout_episodes()
    assert controller.runtime_state.candidates == ()
    assert controller.runtime_state.model_state is None


def test_engine_tuning_outcome_builder_ignores_alt_baseline_episode() -> None:
    assert engine_tuning_outcome_from_episode(_successful_engine_episode()) is not None
    assert (
        engine_tuning_outcome_from_episode(_successful_engine_episode(alt_baseline_id="alt-a"))
        is None
    )


def _successful_engine_episode(*, alt_baseline_id: str | None = None) -> dict[str, object]:
    episode: dict[str, object] = {
        "engine_tuning_context_key": "mute_city|blue_falcon",
        "engine_tuning_course_key": "mute_city",
        "engine_tuning_vehicle_id": "blue_falcon",
        "episode_completion_fraction": 1.0,
        "position": 1,
        "race_time_ms": 90_000,
        "termination_reason": "finished",
        "total_racers": 30,
        "track_engine_setting_raw_value": 50,
    }
    if alt_baseline_id is not None:
        episode["track_alt_baseline_id"] = alt_baseline_id
    return episode
