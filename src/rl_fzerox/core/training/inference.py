# src/rl_fzerox/core/training/inference.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl_fzerox.core.training.runs import resolve_policy_artifact_path


@dataclass(frozen=True)
class LoadedPolicy:
    """Resolved policy-only artifact metadata for watch mode."""

    run_dir: Path
    policy_path: Path
    artifact: str


class PolicyRunner:
    """Small deterministic inference wrapper around one saved policy artifact."""

    def __init__(self, loaded_policy: LoadedPolicy, policy) -> None:
        self._loaded_policy = loaded_policy
        self._policy = policy

    @property
    def label(self) -> str:
        """Return a short label for the currently loaded run."""

        return self._loaded_policy.run_dir.name

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict one deterministic action for the current observation."""

        action, _ = self._policy.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.int64)


def load_policy_runner(run_dir: Path, *, artifact: str = "latest") -> PolicyRunner:
    """Load one saved policy artifact from one run directory."""

    resolved_run_dir = run_dir.expanduser().resolve()
    policy_path = resolve_policy_artifact_path(resolved_run_dir, artifact=artifact)
    policy = _load_saved_policy(policy_path)
    return PolicyRunner(
        LoadedPolicy(
            run_dir=resolved_run_dir,
            policy_path=policy_path,
            artifact=artifact,
        ),
        policy=policy,
    )


def _load_saved_policy(policy_path: Path):
    """Load one saved policy-only SB3 artifact.

    This remains PPO-specific today because the current training path only
    writes PPO `CnnPolicy` artifacts.
    """

    from stable_baselines3.ppo import CnnPolicy

    from rl_fzerox.core.policy import FZeroXCnnExtractor as _  # noqa: F401

    return CnnPolicy.load(str(policy_path), device="auto")
