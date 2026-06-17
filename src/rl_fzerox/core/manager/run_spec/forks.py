# src/rl_fzerox/core/manager/run_spec/forks.py
from __future__ import annotations

from rl_fzerox.core.manager.run_spec.run import ManagedRunConfig


def reset_fork_action_bias_deltas(config: ManagedRunConfig) -> ManagedRunConfig:
    """Return a fork baseline with one-shot action-logit deltas reset.

    Action-logit fields are launch-time deltas. A fork inherits the source
    checkpoint weights and their persisted cumulative bias metadata, so the
    editable child config starts at zero and only applies another nudge if the
    user explicitly enters one before launching training.
    """

    return config.model_copy(
        update={
            "policy": config.policy.model_copy(
                update={
                    "gas_on_logit": 0.0,
                    "air_brake_on_logit": 0.0,
                    "spin_idle_logit": 0.0,
                }
            )
        }
    )
