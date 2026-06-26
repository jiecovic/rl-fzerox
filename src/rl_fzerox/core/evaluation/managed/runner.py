# src/rl_fzerox/core/evaluation/managed/runner.py
"""Project manager evaluation records into executable evaluation runs.

SQLite-owned evaluation records remain the source of truth. This runner builds
the temporary train-like runtime config, materializes/reuses baseline suites,
loads the frozen checkpoint copy, and chooses the single or parallel executor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Literal

from rl_fzerox.core.evaluation.execution.executor import FZeroXSingleCourseEpisodeExecutor
from rl_fzerox.core.evaluation.execution.runner import (
    build_evaluation_attempt_plan,
    run_course_evaluation,
)
from rl_fzerox.core.evaluation.managed.engine_tuning import (
    configure_evaluation_engine_tuning,
)
from rl_fzerox.core.evaluation.managed.parallel import run_parallel_managed_evaluation
from rl_fzerox.core.evaluation.models import (
    EvaluationRunResult,
    EvaluationRuntimeSpec,
    EvaluationSpec,
)
from rl_fzerox.core.evaluation.targets import single_course_targets_from_config
from rl_fzerox.core.manager.models import ManagedEvaluation
from rl_fzerox.core.manager.training import build_managed_train_app_config
from rl_fzerox.core.runtime_spec.inference import inference_train_app_config
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.training.inference import load_policy_runner
from rl_fzerox.core.training.runs import (
    RUN_LAYOUT,
    RunPaths,
    ensure_run_dirs,
    explicit_run_paths,
    materialize_train_run_config,
    save_train_run_config,
)
from rl_fzerox.core.training.session.env import build_single_training_env


@dataclass(frozen=True, slots=True)
class EvaluationBaselineSuite:
    """Shared materialized baseline suite for equivalent evaluation presets."""

    run_paths: RunPaths
    manifest_path: Path


def run_managed_evaluation(
    evaluation: ManagedEvaluation,
    *,
    device: Literal["cpu", "cuda"] = "cuda",
    worker_count: int = 1,
    should_cancel: Callable[[], bool] | None = None,
) -> EvaluationRunResult:
    """Execute one manager evaluation record."""

    if worker_count < 1:
        raise ValueError(f"worker_count must be at least 1, got {worker_count}")
    train_config, run_paths = _materialize_evaluation_train_config(evaluation)
    runtime_config = _runtime_device_config(train_config, device=device)
    targets = single_course_targets_from_config(runtime_config, evaluation.target)
    spec = EvaluationSpec(
        evaluation_id=evaluation.id,
        seed=evaluation.seed,
        target=evaluation.target,
        checkpoint=evaluation.checkpoint,
        policy_mode=evaluation.policy_mode,
    )
    plan = build_evaluation_attempt_plan(spec, targets)
    actual_worker_count = min(worker_count, len(plan.jobs))
    runtime = EvaluationRuntimeSpec(device=device, worker_count=actual_worker_count)
    max_env_steps = ceil(train_config.env.max_episode_steps / train_config.env.action_repeat)
    if actual_worker_count > 1:
        return run_parallel_managed_evaluation(
            evaluation,
            runtime_config=runtime_config,
            run_paths=run_paths,
            plan=plan,
            runtime=runtime,
            max_env_steps=max_env_steps,
            should_cancel=should_cancel,
        )

    policy_runner = load_policy_runner(
        evaluation.evaluation_dir / "checkpoint_snapshot",
        artifact=evaluation.checkpoint.artifact,
        device=runtime_config.train.device,
        algorithm=runtime_config.train.algorithm,
    )
    env = build_single_training_env(
        runtime_config,
        env_index=0,
        runtime_dir=run_paths.env_runtime_dir(0),
    )
    try:
        configure_evaluation_engine_tuning(
            env,
            runtime_config,
            policy_path=Path(evaluation.checkpoint.copied_policy_path),
        )
        executor = FZeroXSingleCourseEpisodeExecutor(
            env=env,
            policy_runner=policy_runner,
            max_env_steps=max_env_steps,
        )
        return run_course_evaluation(
            plan.spec,
            targets,
            executor,
            runtime=runtime,
            result_dir=evaluation.evaluation_dir,
            should_cancel=should_cancel,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _runtime_device_config(
    config: TrainAppConfig,
    *,
    device: Literal["cpu", "cuda"],
) -> TrainAppConfig:
    return config.model_copy(update={"train": config.train.model_copy(update={"device": device})})


def _materialize_evaluation_train_config(
    evaluation: ManagedEvaluation,
) -> tuple[TrainAppConfig, RunPaths]:
    """Project and materialize the train-like runtime config used by evaluation."""

    run_paths = _evaluation_runtime_paths(evaluation)
    ensure_run_dirs(run_paths)
    projected_config = build_managed_train_app_config(
        evaluation.config,
        run_id=evaluation.id,
        run_dir=run_paths.run_dir,
    )
    projected_config = inference_train_app_config(projected_config)
    materializer_input = _evaluation_materializer_input(
        projected_config,
        seed=evaluation.seed,
    )
    suite = _evaluation_baseline_suite(evaluation)
    ensure_run_dirs(suite.run_paths)
    materialized_config = _materialize_baseline_suite(
        _suite_materializer_input(materializer_input, suite=suite),
        suite=suite,
    )
    # Keep an eval-local manifest for debugging/repro while the baseline_state_path
    # values intentionally point at the shared suite.
    save_train_run_config(config=materialized_config, run_dir=run_paths.run_dir)
    return materialized_config, run_paths


def _evaluation_runtime_paths(evaluation: ManagedEvaluation) -> RunPaths:
    return explicit_run_paths(evaluation.evaluation_dir / "runtime_projection")


def _evaluation_baseline_suite(evaluation: ManagedEvaluation) -> EvaluationBaselineSuite:
    run_paths = explicit_run_paths(
        evaluation.evaluation_dir.parent
        / "_baseline_suites"
        / _evaluation_baseline_suite_id(evaluation)
    )
    return EvaluationBaselineSuite(
        run_paths=run_paths,
        manifest_path=run_paths.run_dir / RUN_LAYOUT.config_filename,
    )


def _evaluation_baseline_suite_id(evaluation: ManagedEvaluation) -> str:
    """Return the explicit preset-version baseline-suite key."""

    return f"{evaluation.preset_id}-v{evaluation.preset_version}"


def _suite_materializer_input(
    config: TrainAppConfig,
    *,
    suite: EvaluationBaselineSuite,
) -> TrainAppConfig:
    return config.model_copy(
        update={
            "train": config.train.model_copy(
                update={
                    "run_name": suite.run_paths.run_dir.name,
                    "output_root": suite.run_paths.run_dir.parent,
                    "explicit_run_dir": suite.run_paths.run_dir,
                }
            )
        }
    )


def _materialize_baseline_suite(
    config: TrainAppConfig,
    *,
    suite: EvaluationBaselineSuite,
) -> TrainAppConfig:
    # Evaluation suites reuse baseline files, not stale config snapshots. Always
    # project from the SQLite-owned evaluation config and then sync the manifest mirror.
    materialized_config = materialize_train_run_config(config, run_paths=suite.run_paths)
    save_train_run_config(config=materialized_config, run_dir=suite.run_paths.run_dir)
    return materialized_config


def _evaluation_materializer_input(config: TrainAppConfig, *, seed: int) -> TrainAppConfig:
    """Return a materializer input whose reset pool is owned by the eval target.

    Evaluation configs are SQLite-owned snapshots produced from the preset.
    Preserve their baseline-variant count so the preset-version baseline suite
    can be reused across evaluations with the same preset.
    """

    return config.model_copy(update={"seed": seed})
