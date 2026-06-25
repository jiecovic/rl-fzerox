# tests/core/runtime_spec/test_training_schema.py
"""Validation coverage for runtime training schema execution controls."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rl_fzerox.core.runtime_spec.schema.training import TrainConfig


def test_full_model_resume_requires_checkpoint_run_dir() -> None:
    with pytest.raises(ValidationError, match="resume_mode=full_model"):
        TrainConfig(resume_mode="full_model")


def test_in_place_continue_requires_matching_resume_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    other_run_dir = tmp_path / "other-run"

    with pytest.raises(ValidationError, match="continue_run_dir requires train.resume_run_dir"):
        TrainConfig(continue_run_dir=run_dir)

    with pytest.raises(ValidationError, match="must match train.resume_run_dir"):
        TrainConfig(
            continue_run_dir=run_dir,
            resume_run_dir=other_run_dir,
            resume_mode="full_model",
        )


def test_in_place_continue_requires_full_model_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    with pytest.raises(ValidationError, match="continue_run_dir requires train.resume_mode"):
        TrainConfig(
            continue_run_dir=run_dir,
            resume_run_dir=run_dir,
            resume_mode="weights_only",
        )


def test_explicit_run_dir_must_match_in_place_continue_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    other_run_dir = tmp_path / "other-run"

    with pytest.raises(ValidationError, match="explicit_run_dir must match"):
        TrainConfig(
            explicit_run_dir=other_run_dir,
            continue_run_dir=run_dir,
            resume_run_dir=run_dir,
            resume_mode="full_model",
        )


def test_required_resume_source_metadata_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="resume_source_algorithm"):
        TrainConfig(
            resume_run_dir=tmp_path / "source-run",
            resume_source_metadata_required=True,
        )
