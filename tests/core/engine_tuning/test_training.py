# tests/core/engine_tuning/test_training.py
from __future__ import annotations

from typing import Literal

import pytest

from rl_fzerox.core.engine_tuning import EngineTuningContext
from rl_fzerox.core.engine_tuning.training import (
    EngineTuningTrainingController,
    engine_tuning_outcome_from_episode,
)
from rl_fzerox.core.runtime_spec.schema import AdaptiveEngineTuningConfig


@pytest.mark.parametrize("backend", ["bandit", "gaussian_process", "mlp_ensemble"])
def test_engine_tuning_controller_ignores_alt_baseline_episodes(
    backend: Literal["bandit", "gaussian_process", "mlp_ensemble"],
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


def test_bandit_reset_sampler_uses_best_observed_bucket_as_greedy() -> None:
    context = EngineTuningContext(course_key="mute_city", vehicle_id="blue_falcon")
    controller = EngineTuningTrainingController(
        AdaptiveEngineTuningConfig(
            enabled=True,
            backend="bandit",
            min_raw_value=0,
            max_raw_value=128,
            slider_spacing=13,
            uniform_exploration=0.0,
        )
    )
    assert controller.record_episodes(
        (
            _successful_engine_episode(engine_raw=25, race_time_ms=80_000),
            _successful_engine_episode(engine_raw=90, race_time_ms=80_500),
        )
    )

    snapshot = controller.reset_sampler_snapshot((context,))

    assert snapshot.contexts[0].greedy_engine_setting_raw_value == 25


def _successful_engine_episode(
    *,
    alt_baseline_id: str | None = None,
    engine_raw: int = 50,
    race_time_ms: int = 90_000,
) -> dict[str, object]:
    episode: dict[str, object] = {
        "engine_tuning_context_key": "mute_city|blue_falcon",
        "engine_tuning_course_key": "mute_city",
        "engine_tuning_vehicle_id": "blue_falcon",
        "episode_completion_fraction": 1.0,
        "position": 1,
        "race_time_ms": race_time_ms,
        "termination_reason": "finished",
        "total_racers": 30,
        "track_engine_setting_raw_value": engine_raw,
    }
    if alt_baseline_id is not None:
        episode["track_alt_baseline_id"] = alt_baseline_id
    return episode
