# src/rl_fzerox/core/training/runs/baseline_materializer/__init__.py
"""Run-local training baseline materialization and cache reuse."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fzerox_emulator import Emulator
from rl_fzerox.core.config.schema import TrainAppConfig
from rl_fzerox.core.training.runs.baseline_race_start import (
    materialize_time_attack_race_start_from_boot,
)
from rl_fzerox.core.training.runs.paths import RunPaths

from .materialize import materialize_baseline_impl, materialize_run_baselines_impl
from .models import BaselineArtifact, BaselineMaterializerContext, BaselineRequest
from .settings import BASELINE_MATERIALIZER_SETTINGS, BaselineMaterializerSettings

_RaceStartMaterializer = Callable[..., None]


def materialize_run_baselines(
    config: TrainAppConfig,
    *,
    run_paths: RunPaths,
    cache_root: Path | None = None,
) -> TrainAppConfig:
    """Generate run-local baseline state artifacts for the current run."""

    return materialize_run_baselines_impl(
        config,
        run_paths=run_paths,
        cache_root=cache_root,
        emulator_type=Emulator,
        race_start_materializer=materialize_time_attack_race_start_from_boot,
    )


def materialize_baseline(
    request: BaselineRequest,
    *,
    run_paths: RunPaths,
    cache_root: Path,
    context: BaselineMaterializerContext,
) -> BaselineArtifact:
    """Ensure one generated baseline exists in cache and copy it into the run."""

    return materialize_baseline_impl(
        request,
        run_paths=run_paths,
        cache_root=cache_root,
        context=context,
        emulator_type=Emulator,
        race_start_materializer=materialize_time_attack_race_start_from_boot,
    )


__all__ = [
    "BASELINE_MATERIALIZER_SETTINGS",
    "BaselineArtifact",
    "BaselineMaterializerContext",
    "BaselineMaterializerSettings",
    "BaselineRequest",
    "Emulator",
    "materialize_baseline",
    "materialize_run_baselines",
    "materialize_time_attack_race_start_from_boot",
]
