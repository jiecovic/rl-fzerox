# src/rl_fzerox/core/manager/projection/compat.py
"""Compatibility checks for forking from existing managed checkpoints."""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.manager.projection.launches import build_managed_train_app_config
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.training.session.model.compatibility import (
    resume_compatibility_change_labels,
    resume_compatibility_signature,
)


def assert_managed_fork_compatible(
    source_config: ManagedRunConfig,
    candidate_config: ManagedRunConfig,
) -> None:
    """Fail early when one forked child config is incompatible with its source checkpoint."""

    source_train = build_managed_train_app_config(
        source_config,
        run_id="source-compat",
        run_dir=Path("local/manager/source-compat"),
    )
    candidate_train = build_managed_train_app_config(
        candidate_config,
        run_id="candidate-compat",
        run_dir=Path("local/manager/candidate-compat"),
    )

    changed = resume_compatibility_change_labels(source_train, candidate_train)
    if not changed:
        return

    detail = ", ".join(changed) if changed else "checkpoint structure"
    raise ValueError(
        "Cannot fork from this checkpoint after incompatible edits: "
        f"{detail}. Change reward/training knobs only, or start a fresh run."
    )


def fork_compatibility_signature(train_config: TrainAppConfig) -> dict[str, object]:
    return resume_compatibility_signature(train_config)
