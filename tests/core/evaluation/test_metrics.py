# tests/core/evaluation/test_metrics.py
from __future__ import annotations

import pytest

from rl_fzerox.core.evaluation import (
    EvaluationAttemptResult,
    EvaluationCheckpointSnapshot,
    EvaluationCourseResult,
    EvaluationRunResult,
    EvaluationSpec,
    EvaluationTargetSpec,
    aggregate_evaluation_metrics,
)


def test_evaluation_metrics_aggregate_primary_and_detail_stats() -> None:
    result = _sample_result()

    metrics = aggregate_evaluation_metrics(result)

    assert metrics.overall.primary.attempt_count == 2
    assert metrics.overall.primary.success_count == 1
    assert metrics.overall.primary.success_rate == 0.5
    assert metrics.overall.primary.finish_count == 2
    assert metrics.overall.primary.finish_rate == pytest.approx(2.0 / 3.0)
    assert metrics.overall.primary.completion_rate == pytest.approx(0.8)
    assert metrics.overall.primary.mean_finish_time_ms == 95_000.0
    assert metrics.overall.primary.best_finish_time_ms == 90_000
    assert metrics.overall.primary.mean_total_race_time_ms == 190_000.0
    assert metrics.overall.primary.mean_position == 1.5
    assert metrics.overall.primary.mean_final_gp_position == 1.0
    assert metrics.overall.primary.best_gp_points == 100
    assert metrics.overall.primary.total_env_steps == 6_900
    assert metrics.overall.primary.mean_episode_length_steps == 3_450.0

    assert metrics.overall.detail.mean_episode_return == pytest.approx(347.0)
    assert metrics.overall.detail.best_episode_return == 700.0
    assert metrics.overall.detail.boost_active_count == 3
    assert metrics.overall.detail.boost_active_frames == 100
    assert metrics.overall.detail.boost_pad_entries == 4
    assert metrics.overall.detail.damage_event_count == 2
    assert metrics.overall.detail.minimum_height == -10.0


def test_evaluation_metrics_break_down_by_course_and_cup() -> None:
    result = _sample_result()

    metrics = aggregate_evaluation_metrics(result)
    cup = metrics.cups[0]
    mute_city = next(group for group in metrics.courses if group.key == "mute_city")

    assert cup.key == "jack"
    assert cup.primary.success_rate == 0.5
    assert cup.primary.finish_rate == pytest.approx(2.0 / 3.0)
    assert cup.primary.best_total_race_time_ms == 190_000

    assert mute_city.label == "Mute City"
    assert mute_city.primary.attempt_count == 2
    assert mute_city.primary.finish_rate == 0.5
    assert mute_city.primary.completion_rate == 0.7
    assert mute_city.primary.mean_finish_time_ms == 90_000.0
    assert mute_city.primary.total_env_steps == 3_700
    assert mute_city.detail.damage_event_count == 2
    assert mute_city.detail.minimum_height == -10.0


def _sample_result() -> EvaluationRunResult:
    return EvaluationRunResult(
        spec=EvaluationSpec(
            evaluation_id="eval-test",
            seed=123,
            target=EvaluationTargetSpec(
                mode="gp_cup",
                cup_ids=("jack",),
                vehicle_ids=("blue_falcon",),
                repeats_per_target=2,
            ),
            checkpoint=EvaluationCheckpointSnapshot(
                source_run_id="run-a",
                source_run_name="Run A",
                artifact="latest",
                source_policy_path="/runs/run-a/checkpoints/latest/policy.zip",
                copied_policy_path="/evals/eval-test/checkpoints/latest/policy.zip",
                local_num_timesteps=10_000,
            ),
            total_planned_attempts=2,
        ),
        status="completed",
        attempts=(
            EvaluationAttemptResult(
                attempt_id="attempt-1",
                target_id="jack-master",
                target_label="Jack Cup Master",
                status="succeeded",
                cup_id="jack",
                final_gp_position=1,
                gp_points=100,
                total_race_time_ms=190_000,
                env_steps=6_200,
                episode_length_steps=6_200,
                episode_return=700.0,
                course_results=(
                    EvaluationCourseResult(
                        course_id="mute_city",
                        course_name="Mute City",
                        cup_id="jack",
                        status="finished",
                        race_time_ms=90_000,
                        position=1,
                        env_steps=3_000,
                        episode_length_steps=3_000,
                        episode_return=500.0,
                        boost_active_count=2,
                        boost_active_frames=80,
                        boost_pad_entries=3,
                        damage_event_count=1,
                        minimum_height=-2.0,
                        average_speed=820.0,
                    ),
                    EvaluationCourseResult(
                        course_id="silence",
                        course_name="Silence",
                        cup_id="jack",
                        status="finished",
                        race_time_ms=100_000,
                        position=2,
                        env_steps=3_200,
                        episode_length_steps=3_200,
                        episode_return=450.0,
                        boost_active_count=1,
                        boost_active_frames=20,
                        boost_pad_entries=1,
                        damage_event_count=0,
                        minimum_height=0.0,
                        average_speed=780.0,
                    ),
                ),
            ),
            EvaluationAttemptResult(
                attempt_id="attempt-2",
                target_id="jack-master",
                target_label="Jack Cup Master",
                status="failed",
                cup_id="jack",
                env_steps=700,
                episode_length_steps=700,
                episode_return=-6.0,
                course_results=(
                    EvaluationCourseResult(
                        course_id="mute_city",
                        course_name="Mute City",
                        cup_id="jack",
                        status="crashed",
                        completion_ratio=0.4,
                        env_steps=700,
                        episode_length_steps=700,
                        episode_return=-6.0,
                        damage_event_count=1,
                        minimum_height=-10.0,
                        average_speed=510.0,
                    ),
                ),
            ),
        ),
    )
