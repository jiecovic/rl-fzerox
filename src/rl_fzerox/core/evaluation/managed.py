# src/rl_fzerox/core/evaluation/managed.py
"""Run manager-owned evaluation records through the course evaluator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

from rl_fzerox.core.evaluation.executor import FZeroXSingleCourseEpisodeExecutor
from rl_fzerox.core.evaluation.models import EvaluationRunResult, EvaluationSpec
from rl_fzerox.core.evaluation.runner import run_course_evaluation
from rl_fzerox.core.evaluation.targets import single_course_targets_from_config
from rl_fzerox.core.manager.models import ManagedEvaluation
from rl_fzerox.core.manager.training import build_managed_train_app_config
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
from rl_fzerox.core.training.session.env import build_single_training_env


@dataclass(frozen=True, slots=True)
class EvaluationBaselineSuite:
    """Shared materialized baseline suite for equivalent evaluation presets."""

    run_paths: RunPaths
    manifest_path: Path


def run_managed_evaluation(evaluation: ManagedEvaluation) -> EvaluationRunResult:
    """Execute one manager evaluation record."""

    train_config, run_paths = _materialize_evaluation_train_config(evaluation)
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
        runtime_dir=run_paths.env_runtime_dir(0),
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
    materializer_input = _evaluation_materializer_input(
        projected_config,
        seed=evaluation.seed,
    )
    suite = _evaluation_baseline_suite(evaluation, materializer_input)
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


def _evaluation_baseline_suite(
    evaluation: ManagedEvaluation,
    config: TrainAppConfig,
) -> EvaluationBaselineSuite:
    run_paths = explicit_run_paths(
        evaluation.evaluation_dir.parent
        / "_baseline_suites"
        / _evaluation_baseline_suite_id(evaluation, config)
    )
    return EvaluationBaselineSuite(
        run_paths=run_paths,
        manifest_path=run_paths.run_dir / RUN_LAYOUT.config_filename,
    )


def _evaluation_baseline_suite_id(
    evaluation: ManagedEvaluation,
    config: TrainAppConfig,
) -> str:
    """Return the stable shared-baseline key for this eval target.

    Repeats do not affect baseline materialization, but the preset seed does:
    it controls GP opponent grids and should be identical across compared
    checkpoints that use the same preset.
    """

    target_data = asdict(evaluation.target)
    target_data.pop("repeats_per_target", None)
    payload = {
        "schema": 1,
        "seed": evaluation.seed,
        "target": target_data,
        "baseline_config": _baseline_suite_config_data(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{evaluation.target.mode}-{digest}"


def _baseline_suite_config_data(config: TrainAppConfig) -> dict[str, object]:
    """Return only the config fields that affect baseline materialization."""

    return {
        "seed": config.seed,
        "emulator": {
            "core_path": str(config.emulator.core_path),
            "rom_path": str(config.emulator.rom_path),
            "renderer": config.emulator.renderer,
        },
        "track": _without_baseline_path(config.track.model_dump(mode="json")),
        "env": {
            "camera_setting": config.env.camera_setting,
            "race_intro_target_timer": config.env.race_intro_target_timer,
            "track_sampling": _track_sampling_baseline_key(config),
        },
    }


def _track_sampling_baseline_key(config: TrainAppConfig) -> dict[str, object]:
    data = config.env.track_sampling.model_dump(mode="json")
    entries = data.get("entries")
    if isinstance(entries, list):
        data["entries"] = [
            _without_baseline_path(entry) if isinstance(entry, dict) else entry for entry in entries
        ]
    return data


def _without_baseline_path(data: dict[str, object]) -> dict[str, object]:
    cleaned = dict(data)
    cleaned["baseline_state_path"] = None
    return cleaned


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
