# src/rl_fzerox/core/evaluation/managed.py
"""Run manager-owned evaluation records through the course evaluator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Literal

from rl_fzerox.core.engine_tuning.training import EngineTuningTrainingController
from rl_fzerox.core.envs.engine.reset.track_sampling import engine_tuning_context_for_entry
from rl_fzerox.core.evaluation.env_control import sync_checkpoint_curriculum_stage
from rl_fzerox.core.evaluation.executor import FZeroXSingleCourseEpisodeExecutor
from rl_fzerox.core.evaluation.models import EvaluationRunResult, EvaluationSpec
from rl_fzerox.core.evaluation.runner import run_course_evaluation
from rl_fzerox.core.evaluation.targets import single_course_targets_from_config
from rl_fzerox.core.manager.models import ManagedEvaluation
from rl_fzerox.core.manager.training import build_managed_train_app_config
from rl_fzerox.core.runtime_spec.inference import inference_train_app_config
from rl_fzerox.core.runtime_spec.schema import (
    CurriculumConfig,
    CurriculumStageConfig,
    TrainAppConfig,
)
from rl_fzerox.core.training.inference import load_policy_runner
from rl_fzerox.core.training.runs import (
    RUN_LAYOUT,
    RunPaths,
    ensure_run_dirs,
    explicit_run_paths,
    load_train_run_config,
    materialize_train_run_config,
    save_train_run_config,
)
from rl_fzerox.core.training.session.artifacts import load_engine_tuning_checkpoint_state
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
    should_cancel: Callable[[], bool] | None = None,
) -> EvaluationRunResult:
    """Execute one manager evaluation record."""

    train_config, run_paths = _materialize_evaluation_train_config(evaluation)
    runtime_config = _runtime_device_config(train_config, device=device)
    targets = single_course_targets_from_config(runtime_config, evaluation.target)
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
        sync_checkpoint_curriculum_stage(env, policy_runner.checkpoint_curriculum_stage_index)
        _configure_evaluation_engine_tuning(
            env,
            runtime_config,
            policy_path=Path(evaluation.checkpoint.copied_policy_path),
        )
        executor = FZeroXSingleCourseEpisodeExecutor(
            env=env,
            policy_runner=policy_runner,
            max_env_steps=ceil(train_config.env.max_episode_steps / train_config.env.action_repeat),
        )
        return run_course_evaluation(
            EvaluationSpec(
                evaluation_id=evaluation.id,
                seed=evaluation.seed,
                target=evaluation.target,
                checkpoint=evaluation.checkpoint,
                policy_mode=evaluation.policy_mode,
            ),
            targets,
            executor,
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


def _configure_evaluation_engine_tuning(
    env: object,
    config: TrainAppConfig,
    *,
    policy_path: Path,
) -> None:
    track_sampling = config.env.track_sampling
    if not track_sampling.enabled or not track_sampling.engine_tuning.enabled:
        return

    contexts = tuple(engine_tuning_context_for_entry(entry) for entry in track_sampling.entries)
    if not contexts:
        return

    state = load_engine_tuning_checkpoint_state(policy_path)
    controller = EngineTuningTrainingController(track_sampling.engine_tuning, state=state)
    sampler = controller.reset_sampler_snapshot(contexts)
    set_sampler = getattr(env, "set_engine_tuning_sampler", None)
    if not callable(set_sampler):
        raise TypeError("evaluation env does not support engine tuning sampler updates")
    set_sampler(sampler)

    set_selection = getattr(env, "set_engine_tuning_selection", None)
    if not callable(set_selection):
        raise TypeError("evaluation env does not support engine tuning selection updates")
    set_selection("greedy")


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
    materialized_config = _load_or_materialize_baseline_suite(
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


def _load_or_materialize_baseline_suite(
    config: TrainAppConfig,
    *,
    suite: EvaluationBaselineSuite,
) -> TrainAppConfig:
    if suite.manifest_path.is_file():
        materialized_config = load_train_run_config(suite.run_paths.run_dir)
        if _materialized_baselines_are_ready(materialized_config):
            return materialized_config

    materialized_config = materialize_train_run_config(config, run_paths=suite.run_paths)
    save_train_run_config(config=materialized_config, run_dir=suite.run_paths.run_dir)
    return materialized_config


def _materialized_baselines_are_ready(config: TrainAppConfig) -> bool:
    if config.env.track_sampling.enabled:
        return all(
            entry.baseline_state_path is not None
            and entry.baseline_state_path.expanduser().is_file()
            for entry in config.env.track_sampling.entries
        )
    if config.emulator.baseline_state_path is None:
        return True
    return config.emulator.baseline_state_path.expanduser().is_file()


def _evaluation_materializer_input(config: TrainAppConfig, *, seed: int) -> TrainAppConfig:
    """Return a materializer input whose reset pool is owned by the eval target.

    Training configs may carry multiple GP baseline variants and curriculum
    stages with their own track pools. Evaluation repeats are specified by the
    evaluation preset, so variants must not multiply the target count. Curriculum
    action-mask stages are still preserved for checkpoint-stage inference, but
    their track-sampling overrides are removed so the explicit eval target pool
    remains the single reset source.
    """

    track_sampling = config.env.track_sampling.model_copy(update={"baseline_variant_count": 1})
    return config.model_copy(
        update={
            "seed": seed,
            "env": config.env.model_copy(update={"track_sampling": track_sampling}),
            "curriculum": _curriculum_without_track_sampling(config.curriculum),
        }
    )


def _curriculum_without_track_sampling(config: CurriculumConfig) -> CurriculumConfig:
    if not config.enabled or not config.stages:
        return config
    stages: list[CurriculumStageConfig] = []
    changed = False
    for stage in config.stages:
        if stage.track_sampling is None:
            stages.append(stage)
            continue
        changed = True
        stages.append(stage.model_copy(update={"track_sampling": None}))
    if not changed:
        return config
    return config.model_copy(update={"stages": tuple(stages)})
