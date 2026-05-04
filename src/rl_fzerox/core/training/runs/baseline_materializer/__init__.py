# src/rl_fzerox/core/training/runs/baseline_materializer/__init__.py
"""Run-local training baseline materialization and cache reuse."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fzerox_emulator import Emulator
from rl_fzerox.core.config.schema import TrainAppConfig
from rl_fzerox.core.training.runs.baseline_materializer.materialization import (
    materialize_baseline_impl,
    materialize_run_baselines_impl,
)
from rl_fzerox.core.training.runs.baseline_materializer.models import (
    BaselineArtifact,
    BaselineMaterializerContext,
    BaselineRequest,
)
from rl_fzerox.core.training.runs.baseline_materializer.settings import (
    BASELINE_MATERIALIZER_SETTINGS,
    BaselineMaterializerSettings,
)
from rl_fzerox.core.training.runs.paths import RunPaths
from rl_fzerox.core.training.runs.race_start import (
    materialize_generic_mode_seed,
    materialize_race_start_from_boot,
    materialize_race_start_from_menu_seed,
    materialize_race_start_state,
)


def materialize_run_baselines(
    config: TrainAppConfig,
    *,
    run_paths: RunPaths,
    cache_root: Path | None = None,
    startup_reporter: Callable[[str, str], None] | None = None,
) -> TrainAppConfig:
    """Generate run-local baseline state artifacts for the current run."""

    return materialize_run_baselines_impl(
        config,
        run_paths=run_paths,
        cache_root=cache_root,
        startup_reporter=startup_reporter,
        emulator_type=Emulator,
        generic_mode_seed_materializer=materialize_generic_mode_seed,
        menu_seed_race_start_materializer=materialize_race_start_from_menu_seed,
    )


def materialize_baseline(
    request: BaselineRequest,
    *,
    run_paths: RunPaths,
    cache_root: Path,
    context: BaselineMaterializerContext,
) -> BaselineArtifact:
    """Ensure one generated baseline exists in cache and link it into the run."""

    return materialize_baseline_impl(
        request,
        run_paths=run_paths,
        cache_root=cache_root,
        context=context,
        emulator_type=Emulator,
        generic_mode_seed_materializer=materialize_generic_mode_seed,
        menu_seed_race_start_materializer=materialize_race_start_from_menu_seed,
    )


__all__ = [
    "BASELINE_MATERIALIZER_SETTINGS",
    "BaselineArtifact",
    "BaselineMaterializerContext",
    "BaselineMaterializerSettings",
    "BaselineRequest",
    "Emulator",
    "materialize_baseline",
    "materialize_generic_mode_seed",
    "materialize_run_baselines",
    "materialize_race_start_from_boot",
    "materialize_race_start_from_menu_seed",
    "materialize_race_start_state",
]
