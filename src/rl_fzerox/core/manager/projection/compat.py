"""Compatibility checks for forking from existing managed checkpoints."""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.manager.projection.launches import build_managed_train_app_config
from rl_fzerox.core.manager.projection.observations import (
    fork_observation_signature,
)
from rl_fzerox.core.manager.projection.policy import fork_policy_signature
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig


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

    source_signature = fork_compatibility_signature(source_train)
    candidate_signature = fork_compatibility_signature(candidate_train)
    if source_signature == candidate_signature:
        return

    changed = [
        label
        for key, label in (
            ("algorithm", "training algorithm"),
            ("observation", "observation structure"),
            ("action", "action layout"),
            ("policy", "policy architecture"),
        )
        if source_signature[key] != candidate_signature[key]
    ]
    detail = ", ".join(changed) if changed else "checkpoint structure"
    raise ValueError(
        "Cannot fork from this checkpoint after incompatible edits: "
        f"{detail}. Change reward/training knobs only, or start a fresh run."
    )


def fork_compatibility_signature(train_config: TrainAppConfig) -> dict[str, object]:
    observation = fork_observation_signature(train_config)
    runtime_action = train_config.env.action.runtime()
    action = {
        "name": runtime_action.name,
        "steer_buckets": runtime_action.steer_buckets,
        "pitch_buckets": runtime_action.pitch_buckets,
        "independent_lean_buttons": runtime_action.independent_lean_buttons,
        "layout_continuous_axes": tuple(runtime_action.layout_continuous_axes),
        "layout_discrete_axes": tuple(runtime_action.layout_discrete_axes),
    }
    policy = fork_policy_signature(train_config)
    return {
        "algorithm": train_config.train.algorithm,
        "observation": observation,
        "action": action,
        "policy": policy,
    }
