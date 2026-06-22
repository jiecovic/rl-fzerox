# tests/core/evaluation/test_artifacts.py
from __future__ import annotations

import json
from pathlib import Path

from rl_fzerox.core.evaluation import (
    EvaluationAttemptResult,
    EvaluationCheckpointSnapshot,
    EvaluationCourseResult,
    EvaluationRunResult,
    EvaluationSpec,
    EvaluationTargetSpec,
    write_evaluation_result_files,
)


def test_evaluation_writer_persists_json_and_markdown_summary(tmp_path: Path) -> None:
    result = EvaluationRunResult(
        spec=EvaluationSpec(
            evaluation_id="eval-artifact",
            seed=42,
            target=EvaluationTargetSpec(mode="time_attack", course_ids=("mute_city",)),
            checkpoint=EvaluationCheckpointSnapshot(
                source_run_id="run-b",
                source_run_name="Run B",
                artifact="best",
                source_policy_path="/runs/run-b/checkpoints/best/policy.zip",
                copied_policy_path=str(tmp_path / "checkpoints" / "best" / "policy.zip"),
            ),
            total_planned_attempts=1,
        ),
        status="completed",
        attempts=(
            EvaluationAttemptResult(
                attempt_id="attempt-1",
                target_id="mute_city",
                status="succeeded",
                target_label="Mute City",
                env_steps=3_000,
                episode_length_steps=3_000,
                course_results=(
                    EvaluationCourseResult(
                        course_id="mute_city",
                        course_name="Mute City",
                        status="finished",
                        race_time_ms=86_123,
                        position=1,
                        env_steps=3_000,
                        episode_length_steps=3_000,
                        boost_pad_entries=2,
                    ),
                ),
            ),
        ),
    )

    paths = write_evaluation_result_files(result, directory=tmp_path)

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    markdown = paths.markdown_path.read_text(encoding="utf-8")

    assert payload["kind"] == "evaluation_summary"
    assert payload["result"]["spec"]["checkpoint"]["copied_policy_path"] == str(
        tmp_path / "checkpoints" / "best" / "policy.zip"
    )
    assert payload["metrics"]["overall"]["primary"]["finish_count"] == 1
    assert "# Evaluation eval-artifact" in markdown
    assert "Checkpoint copy" in markdown
    assert "Mute City" in markdown
    assert "1:26.123" in markdown
