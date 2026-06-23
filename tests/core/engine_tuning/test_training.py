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

DEFAULT_BANDIT_BUCKETS = (0, 13, 26, 38, 51, 64, 77, 90, 102, 115, 128)


def test_adaptive_engine_tuning_config_rejects_removed_objective_alias() -> None:
    with pytest.raises(ValueError, match="objective"):
        AdaptiveEngineTuningConfig.model_validate({"objective": "completion"})


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
            bucket_raw_values=DEFAULT_BANDIT_BUCKETS,
            uniform_exploration=0.0,
        )
    )
    assert controller.record_episodes(
        (
            _successful_engine_episode(engine_raw=26, race_time_ms=80_000),
            _successful_engine_episode(engine_raw=90, race_time_ms=80_500),
        )
    )

    snapshot = controller.reset_sampler_snapshot((context,))

    assert snapshot.contexts[0].greedy_engine_setting_raw_value == 26


def test_bandit_records_failed_episode_returns_as_diagnostics() -> None:
    controller = EngineTuningTrainingController(
        AdaptiveEngineTuningConfig(
            enabled=True,
            backend="bandit",
            objective="finish_time",
            min_raw_value=0,
            max_raw_value=128,
            bucket_raw_values=DEFAULT_BANDIT_BUCKETS,
            uniform_exploration=0.0,
        )
    )

    assert controller.record_episodes(
        (
            _engine_episode(
                engine_raw=26,
                episode_return=12.5,
                termination_reason="retired",
            ),
        )
    )

    candidate = controller.runtime_state.candidates[0]
    assert candidate.score_count == 0
    assert candidate.episode_count == 1
    assert candidate.return_count == 1
    assert candidate.finish_count == 0
    assert candidate.mean_return_score == 12.5
    assert candidate.mean_score is None


def test_bandit_finish_time_records_failed_episode_metrics_without_scoring_them() -> None:
    context = EngineTuningContext(course_key="mute_city", vehicle_id="blue_falcon")
    controller = EngineTuningTrainingController(
        AdaptiveEngineTuningConfig(
            enabled=True,
            backend="bandit",
            objective="finish_time",
            min_raw_value=0,
            max_raw_value=128,
            bucket_raw_values=DEFAULT_BANDIT_BUCKETS,
            uniform_exploration=0.0,
        )
    )

    assert controller.record_episodes(
        (
            _engine_episode(
                engine_raw=26,
                episode_return=-5.0,
                termination_reason="retired",
            ),
            _successful_engine_episode(engine_raw=90, race_time_ms=80_000, episode_return=8.0),
        )
    )

    candidates = controller.runtime_state.candidate_map()
    failed_candidate = candidates[(context.key, 26)]
    assert failed_candidate.active_score_count == 0
    assert failed_candidate.episode_count == 1
    assert failed_candidate.finish_count == 0
    assert failed_candidate.mean_completion_score == 0.5
    assert failed_candidate.finish_rate_score == 0.0
    assert (
        controller.reset_sampler_snapshot((context,)).contexts[0].greedy_engine_setting_raw_value
        == 90
    )


def test_bandit_can_score_finish_rate_objective() -> None:
    context = EngineTuningContext(course_key="mute_city", vehicle_id="blue_falcon")
    controller = EngineTuningTrainingController(
        AdaptiveEngineTuningConfig(
            enabled=True,
            backend="bandit",
            objective="finish_rate",
            min_raw_value=0,
            max_raw_value=128,
            bucket_raw_values=DEFAULT_BANDIT_BUCKETS,
            uniform_exploration=0.0,
        )
    )

    assert controller.record_episodes(
        (
            _engine_episode(
                engine_raw=26,
                episode_completion_fraction=0.8,
                episode_return=-5.0,
                termination_reason="retired",
            ),
            _engine_episode(
                engine_raw=90,
                episode_completion_fraction=0.6,
                episode_return=5.0,
                race_time_ms=80_000,
                termination_reason="finished",
            ),
        )
    )

    assert (
        controller.reset_sampler_snapshot((context,)).contexts[0].greedy_engine_setting_raw_value
        == 90
    )


def test_bandit_safe_finish_time_requires_finish_rate_before_time() -> None:
    context = EngineTuningContext(course_key="mute_city", vehicle_id="blue_falcon")
    controller = EngineTuningTrainingController(
        AdaptiveEngineTuningConfig(
            enabled=True,
            backend="bandit",
            objective="safe_finish_time",
            safe_finish_rate_threshold=0.75,
            min_raw_value=0,
            max_raw_value=128,
            bucket_raw_values=DEFAULT_BANDIT_BUCKETS,
            uniform_exploration=0.0,
        )
    )
    assert controller.record_episodes(
        (
            _successful_engine_episode(engine_raw=26, race_time_ms=82_000),
            _successful_engine_episode(engine_raw=26, race_time_ms=83_000),
            _engine_episode(
                engine_raw=90,
                episode_completion_fraction=0.8,
                race_time_ms=None,
                termination_reason="retired",
            ),
            _successful_engine_episode(engine_raw=90, race_time_ms=78_000),
        )
    )

    assert (
        controller.reset_sampler_snapshot((context,)).contexts[0].greedy_engine_setting_raw_value
        == 26
    )


def test_bandit_can_switch_to_safe_finish_time_without_recollecting() -> None:
    context = EngineTuningContext(course_key="mute_city", vehicle_id="blue_falcon")
    finish_controller = EngineTuningTrainingController(
        AdaptiveEngineTuningConfig(
            enabled=True,
            backend="bandit",
            objective="finish_time",
            min_raw_value=0,
            max_raw_value=128,
            bucket_raw_values=DEFAULT_BANDIT_BUCKETS,
            uniform_exploration=0.0,
        )
    )
    assert finish_controller.record_episodes(
        (
            _successful_engine_episode(engine_raw=26, race_time_ms=80_000),
            _successful_engine_episode(engine_raw=90, race_time_ms=78_000),
        )
    )

    safe_controller = EngineTuningTrainingController(
        AdaptiveEngineTuningConfig(
            enabled=True,
            backend="bandit",
            objective="safe_finish_time",
            safe_finish_rate_threshold=0.9,
            min_raw_value=0,
            max_raw_value=128,
            bucket_raw_values=DEFAULT_BANDIT_BUCKETS,
            uniform_exploration=0.0,
        ),
        state=finish_controller.runtime_state,
    )

    assert (
        safe_controller.reset_sampler_snapshot((context,))
        .contexts[0]
        .greedy_engine_setting_raw_value
        == 90
    )


def test_engine_tuning_omits_tensorboard_logs() -> None:
    controller = EngineTuningTrainingController(
        AdaptiveEngineTuningConfig(
            enabled=True,
            backend="bandit",
            min_raw_value=0,
            max_raw_value=128,
            bucket_raw_values=DEFAULT_BANDIT_BUCKETS,
        )
    )
    assert controller.record_episodes(
        (
            _successful_engine_episode(engine_raw=26),
            _engine_episode(engine_raw=90, termination_reason="retired"),
        )
    )

    values = controller.log_values()

    assert values == {}


def _successful_engine_episode(
    *,
    alt_baseline_id: str | None = None,
    engine_raw: int = 50,
    race_time_ms: int = 90_000,
    episode_return: float = 1.0,
) -> dict[str, object]:
    return _engine_episode(
        alt_baseline_id=alt_baseline_id,
        engine_raw=engine_raw,
        episode_return=episode_return,
        race_time_ms=race_time_ms,
        termination_reason="finished",
    )


def _engine_episode(
    *,
    alt_baseline_id: str | None = None,
    engine_raw: int = 50,
    episode_completion_fraction: float | None = None,
    episode_return: float = 1.0,
    race_time_ms: int | None = 90_000,
    termination_reason: str = "finished",
) -> dict[str, object]:
    episode: dict[str, object] = {
        "engine_tuning_context_key": "mute_city|blue_falcon",
        "engine_tuning_course_key": "mute_city",
        "engine_tuning_vehicle_id": "blue_falcon",
        "episode_completion_fraction": (
            episode_completion_fraction
            if episode_completion_fraction is not None
            else (1.0 if termination_reason == "finished" else 0.5)
        ),
        "episode_return": episode_return,
        "position": 1,
        "race_time_ms": race_time_ms,
        "termination_reason": termination_reason,
        "total_racers": 30,
        "track_engine_setting_raw_value": engine_raw,
    }
    if alt_baseline_id is not None:
        episode["track_alt_baseline_id"] = alt_baseline_id
    return episode
