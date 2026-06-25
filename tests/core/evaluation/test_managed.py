# tests/core/evaluation/test_managed.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from rl_fzerox.core.engine_tuning import EngineTuningResetSampler
from rl_fzerox.core.evaluation.managed.engine_tuning import (
    configure_evaluation_engine_tuning,
)
from rl_fzerox.core.evaluation.managed.runner import (
    EvaluationBaselineSuite,
    _evaluation_baseline_suite,
    _materialize_baseline_suite,
    run_managed_evaluation,
)
from rl_fzerox.core.evaluation.models import (
    EvaluationCheckpointSnapshot,
    EvaluationTargetSpec,
)
from rl_fzerox.core.manager import ManagedEvaluation, default_managed_run_config
from rl_fzerox.core.manager.training import build_managed_train_app_config
from rl_fzerox.core.runtime_spec.schema import (
    AdaptiveEngineTuningConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
)
from rl_fzerox.core.training.runs import (
    RunPaths,
    ensure_run_dirs,
    explicit_run_paths,
    save_train_run_config,
)


def test_baseline_suite_uses_preset_version_key(tmp_path: Path) -> None:
    evaluation = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-001",
        preset_id="gp-master-blue-falcon",
        preset_version=3,
    )

    assert _suite_dir(evaluation) == (
        tmp_path / "evaluations" / "_baseline_suites" / "gp-master-blue-falcon-v3"
    )


def test_baseline_suite_is_shared_by_same_preset_version(tmp_path: Path) -> None:
    first = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-001",
        target=EvaluationTargetSpec(mode="gp_course", repeats_per_target=1),
    )
    second = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-002",
        target=EvaluationTargetSpec(mode="gp_course", repeats_per_target=10),
    )

    assert _suite_dir(second) == _suite_dir(first)


def test_baseline_suite_changes_with_preset_version(tmp_path: Path) -> None:
    base = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-001",
        preset_id="gp-master-blue-falcon",
        preset_version=1,
    )
    different_version = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-002",
        preset_id="gp-master-blue-falcon",
        preset_version=2,
    )

    assert _suite_dir(different_version) != _suite_dir(base)


def test_baseline_suite_uses_projected_config_instead_of_existing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_paths = explicit_run_paths(tmp_path / "evaluations" / "_baseline_suites" / "suite-v1")
    ensure_run_dirs(run_paths)
    config = build_managed_train_app_config(
        default_managed_run_config(),
        run_id="eval-001",
        run_dir=tmp_path / "evaluations" / "eval-001" / "runtime_projection",
    )
    config_path = save_train_run_config(config=config, run_dir=run_paths.run_dir)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "\nremoved_top_level_field:\n  enabled: false\n",
        encoding="utf-8",
    )
    rematerialized_config = config.model_copy(update={"seed": 456})
    materialized_run_dirs: list[Path] = []

    def fake_materialize_train_run_config(
        input_config: TrainAppConfig,
        *,
        run_paths: RunPaths,
    ) -> TrainAppConfig:
        assert input_config == config
        materialized_run_dirs.append(run_paths.run_dir)
        return rematerialized_config

    monkeypatch.setattr(
        "rl_fzerox.core.evaluation.managed.runner.materialize_train_run_config",
        fake_materialize_train_run_config,
    )

    loaded = _materialize_baseline_suite(
        config,
        suite=EvaluationBaselineSuite(run_paths=run_paths, manifest_path=config_path),
    )

    assert loaded == rematerialized_config
    assert materialized_run_dirs == [run_paths.run_dir]
    saved_manifest = config_path.read_text(encoding="utf-8")
    assert "removed_top_level_field" not in saved_manifest
    assert "seed: 456" in saved_manifest


def test_evaluation_engine_tuning_uses_greedy_checkpoint_sampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_paths: list[Path] = []
    monkeypatch.setattr(
        "rl_fzerox.core.evaluation.managed.engine_tuning.load_engine_tuning_checkpoint_state",
        lambda path: loaded_paths.append(path) or None,
    )
    config = build_managed_train_app_config(
        default_managed_run_config(),
        run_id="eval-001",
        run_dir=tmp_path / "eval-001",
    )
    config = config.model_copy(
        update={
            "env": config.env.model_copy(
                update={
                    "track_sampling": TrackSamplingConfig(
                        enabled=True,
                        entries=(
                            TrackSamplingEntryConfig(
                                id="mute-city",
                                course_id="mute_city",
                                vehicle="blue_falcon",
                            ),
                        ),
                        engine_tuning=AdaptiveEngineTuningConfig(enabled=True),
                    )
                }
            )
        }
    )
    env = _EngineTuningEnv()
    wrapped_env = _TransparentWrapper(env)
    policy_path = tmp_path / "checkpoint_snapshot" / "checkpoints" / "latest" / "policy.zip"

    configure_evaluation_engine_tuning(wrapped_env, config, policy_path=policy_path)

    assert loaded_paths == [policy_path]
    assert env.selection == "greedy"
    assert env.sampler is not None
    assert [context.context.key for context in env.sampler.contexts] == ["mute_city|blue_falcon"]


def test_managed_evaluation_rejects_invalid_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="worker_count must be at least 1"):
        run_managed_evaluation(
            _managed_evaluation(tmp_path, evaluation_id="eval-001"), worker_count=0
        )


class _EngineTuningEnv:
    def __init__(self) -> None:
        self.sampler: EngineTuningResetSampler | None = None
        self.selection: Literal["sample", "greedy"] | None = None

    def set_engine_tuning_sampler(self, sampler: EngineTuningResetSampler) -> None:
        self.sampler = sampler

    def set_engine_tuning_selection(self, selection: Literal["sample", "greedy"]) -> None:
        self.selection = selection


class _TransparentWrapper:
    """Small Gymnasium-like wrapper that does not expose env controls directly."""

    def __init__(self, env: _EngineTuningEnv) -> None:
        self.env = env


def _suite_dir(evaluation: ManagedEvaluation) -> Path:
    return _evaluation_baseline_suite(evaluation).run_paths.run_dir


def _managed_evaluation(
    tmp_path: Path,
    *,
    evaluation_id: str,
    target: EvaluationTargetSpec | None = None,
    seed: int = 123,
    preset_id: str = "gp-master-blue-falcon",
    preset_version: int = 1,
) -> ManagedEvaluation:
    return ManagedEvaluation(
        id=evaluation_id,
        name="Eval",
        status="created",
        evaluation_dir=tmp_path / "evaluations" / evaluation_id,
        source_run_id="run-001",
        source_artifact="latest",
        preset_id=preset_id,
        preset_version=preset_version,
        policy_mode="deterministic",
        seed=seed,
        target=target or EvaluationTargetSpec(mode="gp_course"),
        config=default_managed_run_config(),
        checkpoint=EvaluationCheckpointSnapshot(
            source_run_id="run-001",
            source_run_name="Run 1",
            artifact="latest",
            source_policy_path="runs/run-001/checkpoints/latest/policy.zip",
            copied_policy_path="evaluations/eval/checkpoint_snapshot/policy.zip",
        ),
        created_at="2026-06-23T00:00:00+00:00",
        updated_at="2026-06-23T00:00:00+00:00",
    )
