from __future__ import annotations

from dataclasses import dataclass

import pytest

from rl_fzerox.core.runtime_spec.schema import PolicyConfig
from rl_fzerox.ui.watch.view.auxiliary_metrics import (
    AuxiliaryEpisodeMetric,
    AuxiliaryEpisodeMetricsSnapshot,
    AuxiliaryEpisodeMetricsTracker,
)
from rl_fzerox.ui.watch.view.panels.content.auxiliary import auxiliary_episode_sections


@dataclass(frozen=True)
class _Snapshot:
    episode: int
    policy_decision_frame: bool
    policy_auxiliary_state_predictions: dict[str, object] | None
    policy_auxiliary_state_targets: dict[str, object] | None


def test_auxiliary_episode_metrics_tracker_respects_grounded_only() -> None:
    tracker = AuxiliaryEpisodeMetricsTracker.from_policy_config(
        PolicyConfig.model_validate(
            {
                "auxiliary_state": {
                    "enabled": True,
                    "losses": [
                        {
                            "name": "track_position.edge_ratio",
                            "weight": 0.5,
                            "grounded_only": True,
                        },
                        {
                            "name": "course_context.builtin_course_id",
                            "weight": 1.0,
                        },
                    ],
                }
            }
        )
    )

    tracker.observe_snapshot(
        _Snapshot(
            episode=3,
            policy_decision_frame=True,
            policy_auxiliary_state_predictions={
                "track_position.edge_ratio": 0.30,
                "course_context.builtin_course_id": {
                    "index": 2,
                    "confidence": 0.8,
                    "probabilities": [0.1, 0.1, 0.8],
                },
            },
            policy_auxiliary_state_targets={
                "vehicle_state.airborne": 0.0,
                "track_position.edge_ratio": 0.10,
                "course_context.builtin_course_id": {"index": 2},
            },
        )
    )
    tracker.observe_snapshot(
        _Snapshot(
            episode=3,
            policy_decision_frame=True,
            policy_auxiliary_state_predictions={
                "track_position.edge_ratio": 0.90,
                "course_context.builtin_course_id": {
                    "index": 1,
                    "confidence": 0.7,
                    "probabilities": [0.2, 0.7, 0.1],
                },
            },
            policy_auxiliary_state_targets={
                "vehicle_state.airborne": 1.0,
                "track_position.edge_ratio": 0.20,
                "course_context.builtin_course_id": {"index": 2},
            },
        )
    )

    snapshot = tracker.snapshot()

    assert snapshot is not None
    edge_ratio = next(
        metric for metric in snapshot.metrics if metric.name == "track_position.edge_ratio"
    )
    course_id = next(
        metric for metric in snapshot.metrics if metric.name == "course_context.builtin_course_id"
    )

    assert edge_ratio.sample_count == 1
    assert edge_ratio.mean_loss == pytest.approx(0.02)
    assert edge_ratio.mean_error_percent == 10.0
    assert course_id.sample_count == 2
    assert course_id.accuracy == 0.5


def test_auxiliary_episode_sections_show_aggregated_metrics() -> None:
    sections = auxiliary_episode_sections(
        AuxiliaryEpisodeMetricsSnapshot(
            episode=7,
            metrics=(
                AuxiliaryEpisodeMetric(
                    name="track_position.edge_ratio",
                    sample_count=12,
                    mean_loss=0.024,
                    mean_error_percent=8.2,
                ),
                AuxiliaryEpisodeMetric(
                    name="course_context.builtin_course_id",
                    sample_count=12,
                    mean_loss=0.123,
                    accuracy=0.75,
                    mean_confidence=0.82,
                ),
            ),
        )
    )

    section = sections[0]
    values = {line.label: line.value for line in section.lines if line.label and not line.heading}

    assert values["Episode"] == "7"
    assert values[" "] == " loss | kind |  stat |  conf |   cnt"
    assert values["edge_ratio"] == "0.024 |  err |    8% |     - |    12"
    assert values["course id"] == "0.123 |  acc |   75% |   82% |    12"
