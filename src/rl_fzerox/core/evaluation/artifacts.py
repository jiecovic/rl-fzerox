# src/rl_fzerox/core/evaluation/artifacts.py
"""Local JSON and Markdown artifacts for evaluation results."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_fzerox.core.evaluation.metrics import (
    EvaluationMetricGroup,
    EvaluationMetrics,
    aggregate_evaluation_metrics,
)
from rl_fzerox.core.evaluation.models import EvaluationAttemptResult, EvaluationRunResult


@dataclass(frozen=True, slots=True)
class EvaluationArtifactPaths:
    """Files written for one evaluation result artifact set."""

    json_path: Path
    markdown_path: Path


def write_evaluation_result_files(
    result: EvaluationRunResult,
    *,
    directory: Path,
) -> EvaluationArtifactPaths:
    """Write the evaluation result as JSON plus a human-readable summary."""

    directory.mkdir(parents=True, exist_ok=True)
    metrics = aggregate_evaluation_metrics(result)
    json_path = directory / "evaluation.summary.json"
    markdown_path = directory / "evaluation.summary.md"
    json_path.write_text(
        json.dumps(_evaluation_payload(result, metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _evaluation_markdown(result, metrics),
        encoding="utf-8",
    )
    return EvaluationArtifactPaths(json_path=json_path, markdown_path=markdown_path)


def _evaluation_payload(
    result: EvaluationRunResult,
    metrics: EvaluationMetrics,
) -> dict[str, object]:
    return {
        "kind": "evaluation_summary",
        "schema_version": 1,
        "result": asdict(result),
        "metrics": asdict(metrics),
    }


def _evaluation_markdown(result: EvaluationRunResult, metrics: EvaluationMetrics) -> str:
    source_run = result.spec.checkpoint.source_run_name or result.spec.checkpoint.source_run_id
    lines = [
        f"# Evaluation {result.spec.evaluation_id}",
        "",
        f"- Status: {_text(result.status)}",
        f"- Policy mode: {_text(result.spec.policy_mode)}",
        f"- Seed: {_text(result.spec.seed)}",
        f"- Device: {_text(result.runtime.device)}",
        f"- Workers: {_text(result.runtime.worker_count)}",
        f"- Source run: {_text(source_run)}",
        f"- Artifact: {_text(result.spec.checkpoint.artifact)}",
        f"- Checkpoint copy: {_text(result.spec.checkpoint.copied_policy_path)}",
        f"- Started: {_text(result.started_at_utc)}",
        f"- Closed: {_text(result.closed_at_utc)}",
        "",
    ]
    lines.extend(_primary_section(metrics))
    lines.extend(_detail_section(metrics))
    lines.extend(_attempt_section(result))
    return "\n".join(lines)


def _primary_section(metrics: EvaluationMetrics) -> list[str]:
    lines = [
        "## Primary metrics",
        "",
        ("| Scope | Runs | Finish | Completion | Mean pos | Worst pos |"),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in _metric_groups(metrics):
        primary = group.primary
        lines.append(
            "| "
            + " | ".join(
                (
                    _text(group.label),
                    _text(primary.attempt_count),
                    _percent(primary.finish_rate, count=primary.finish_count),
                    _percent(primary.completion_rate),
                    _number(primary.mean_position),
                    _text(primary.worst_position),
                )
            )
            + " |"
        )
    lines.append("")
    return lines


def _detail_section(metrics: EvaluationMetrics) -> list[str]:
    lines = [
        "## Diagnostic metrics",
        "",
        (
            "| Scope | Success | Mean finish | Best finish | Env steps | Mean episode steps | "
            "Mean return | Best return | Boost active | Boost frames | Boost pads | Damage | "
            "Min height | Avg speed |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: |"
        ),
    ]
    for group in _metric_groups(metrics):
        primary = group.primary
        detail = group.detail
        lines.append(
            "| "
            + " | ".join(
                (
                    _text(group.label),
                    _percent(primary.success_rate, count=primary.success_count),
                    _time(primary.mean_finish_time_ms),
                    _time(primary.best_finish_time_ms),
                    _text(primary.total_env_steps),
                    _number(primary.mean_episode_length_steps),
                    _number(detail.mean_episode_return),
                    _number(detail.best_episode_return),
                    _text(detail.boost_active_count),
                    _text(detail.boost_active_frames),
                    _text(detail.boost_pad_entries),
                    _text(detail.damage_event_count),
                    _number(detail.minimum_height),
                    _number(detail.average_speed),
                )
            )
            + " |"
        )
    lines.append("")
    return lines


def _attempt_section(result: EvaluationRunResult) -> list[str]:
    lines = [
        "## Attempts",
        "",
    ]
    if not result.attempts:
        return lines + ["No attempts recorded.", ""]
    lines.extend(
        (
            ("| Attempt | Target | Status | Seed | Time | Position | Env steps | Return |"),
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for attempt in result.attempts:
        lines.append(
            "| "
            + " | ".join(
                (
                    _text(attempt.attempt_id),
                    _text(attempt.target_label or attempt.target_id),
                    _text(attempt.status),
                    _text(attempt.seed),
                    _time(attempt.total_race_time_ms),
                    _text(_attempt_position(attempt)),
                    _text(attempt.env_steps),
                    _number(attempt.episode_return),
                )
            )
            + " |"
        )
    lines.append("")
    return lines


def _metric_groups(metrics: EvaluationMetrics) -> Iterable[EvaluationMetricGroup]:
    yield metrics.overall
    yield from metrics.cups
    yield from metrics.courses


def _attempt_position(attempt: EvaluationAttemptResult) -> int | None:
    course_results = attempt.course_results
    if len(course_results) != 1:
        return None
    return course_results[0].position


def _text(value: object) -> str:
    if value is None:
        return "-"
    return str(value).replace("|", "\\|")


def _percent(value: float | None, *, count: int | None = None) -> str:
    if value is None:
        return "-"
    text = f"{100.0 * value:.1f}%"
    if count is not None:
        return f"{text} ({count})"
    return text


def _time(value: int | float | None) -> str:
    if value is None:
        return "-"
    milliseconds = max(0, int(round(value)))
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def _number(value: int | float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}"
