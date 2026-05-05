# src/rl_fzerox/core/training/runs/baseline_materializer/__init__.py
"""Lazy baseline-materializer facade for run-local reset artifacts.

This module deliberately avoids importing emulator-backed materialization code
until one of its runtime entry points is actually called. Tests still patch the
facade itself, so the public names remain patchable here instead of being
hidden behind deeper implementation imports.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fzerox_emulator import Emulator
    from rl_fzerox.core.config.schema import TrainAppConfig
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

_EXPORT_MODULES = {
    "BASELINE_MATERIALIZER_SETTINGS": (
        "rl_fzerox.core.training.runs.baseline_materializer.settings"
    ),
    "BaselineArtifact": "rl_fzerox.core.training.runs.baseline_materializer.models",
    "BaselineMaterializerContext": (
        "rl_fzerox.core.training.runs.baseline_materializer.models"
    ),
    "BaselineMaterializerSettings": (
        "rl_fzerox.core.training.runs.baseline_materializer.settings"
    ),
    "BaselineRequest": "rl_fzerox.core.training.runs.baseline_materializer.models",
    "Emulator": "fzerox_emulator",
    "materialize_generic_mode_seed": "rl_fzerox.core.training.runs.race_start",
    "materialize_race_start_from_boot": "rl_fzerox.core.training.runs.race_start",
    "materialize_race_start_from_menu_seed": "rl_fzerox.core.training.runs.race_start",
    "materialize_race_start_state": "rl_fzerox.core.training.runs.race_start",
}


def materialize_run_baselines(
    config: TrainAppConfig,
    *,
    run_paths: RunPaths,
    cache_root: Path | None = None,
    startup_reporter: Callable[[str, str], None] | None = None,
) -> TrainAppConfig:
    """Generate run-local baseline state artifacts for the current run."""

    from rl_fzerox.core.training.runs import baseline_materializer as facade
    from rl_fzerox.core.training.runs.baseline_materializer.materialization import (
        materialize_run_baselines_impl,
    )

    return materialize_run_baselines_impl(
        config,
        run_paths=run_paths,
        cache_root=cache_root,
        startup_reporter=startup_reporter,
        emulator_type=facade.Emulator,
        generic_mode_seed_materializer=facade.materialize_generic_mode_seed,
        menu_seed_race_start_materializer=facade.materialize_race_start_from_menu_seed,
    )


def materialize_baseline(
    request: BaselineRequest,
    *,
    run_paths: RunPaths,
    cache_root: Path,
    context: BaselineMaterializerContext,
) -> BaselineArtifact:
    """Ensure one generated baseline exists in cache and link it into the run."""

    from rl_fzerox.core.training.runs import baseline_materializer as facade
    from rl_fzerox.core.training.runs.baseline_materializer.materialization import (
        materialize_baseline_impl,
    )

    return materialize_baseline_impl(
        request,
        run_paths=run_paths,
        cache_root=cache_root,
        context=context,
        emulator_type=facade.Emulator,
        generic_mode_seed_materializer=facade.materialize_generic_mode_seed,
        menu_seed_race_start_materializer=facade.materialize_race_start_from_menu_seed,
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


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
