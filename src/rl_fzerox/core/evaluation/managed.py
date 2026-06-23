# src/rl_fzerox/core/evaluation/managed.py
"""Run manager-owned evaluation records through the course evaluator."""

from __future__ import annotations

from math import ceil

from rl_fzerox.core.evaluation.executor import FZeroXSingleCourseEpisodeExecutor
from rl_fzerox.core.evaluation.models import EvaluationRunResult, EvaluationSpec
from rl_fzerox.core.evaluation.runner import run_course_evaluation
from rl_fzerox.core.evaluation.targets import single_course_targets_from_config
from rl_fzerox.core.manager.models import ManagedEvaluation
from rl_fzerox.core.manager.training import build_managed_train_app_config
from rl_fzerox.core.training.inference import load_policy_runner
from rl_fzerox.core.training.session.env import build_single_training_env


def run_managed_evaluation(evaluation: ManagedEvaluation) -> EvaluationRunResult:
    """Execute one manager evaluation record."""

    train_config = build_managed_train_app_config(
        evaluation.config,
        run_id=evaluation.id,
        run_dir=evaluation.evaluation_dir / "runtime_projection",
    )
    targets = single_course_targets_from_config(train_config, evaluation.target)
    policy_runner = load_policy_runner(
        evaluation.evaluation_dir / "checkpoint_snapshot",
        artifact=evaluation.checkpoint.artifact,
        device="cpu",
        algorithm=train_config.train.algorithm,
    )
    env = build_single_training_env(
        train_config,
        env_index=0,
        runtime_dir=evaluation.evaluation_dir / "runtime" / "env_000",
    )
    try:
        sync_stage = getattr(env, "sync_checkpoint_curriculum_stage", None)
        if callable(sync_stage):
            sync_stage(policy_runner.checkpoint_curriculum_stage_index)
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
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
