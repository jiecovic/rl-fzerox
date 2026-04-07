# src/rl_fzerox/core/training/inference.py
from __future__ import annotations

import time
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
        self._policy_mtime_ns = _policy_mtime_ns(loaded_policy.policy_path)
        self._last_reload_monotonic = time.monotonic()

    @property
    def label(self) -> str:
        """Return a short label for the currently loaded run."""

        return self._loaded_policy.run_dir.name

    @property
    def reload_age_seconds(self) -> float:
        """Return how long ago the current policy artifact was loaded."""

        return max(0.0, time.monotonic() - self._last_reload_monotonic)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict one deterministic action for the current observation."""

        self._maybe_reload()
        action, _ = self._policy.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.int64)

    def _maybe_reload(self) -> None:
        try:
            policy_path = resolve_policy_artifact_path(
                self._loaded_policy.run_dir,
                artifact=self._loaded_policy.artifact,
            )
        except FileNotFoundError:
            return

        policy_mtime_ns = _policy_mtime_ns(policy_path)
        if (
            policy_path == self._loaded_policy.policy_path
            and policy_mtime_ns == self._policy_mtime_ns
        ):
            return

        try:
            policy = _load_saved_policy(policy_path)
        except Exception:
            return

        self._loaded_policy = LoadedPolicy(
            run_dir=self._loaded_policy.run_dir,
            policy_path=policy_path,
            artifact=self._loaded_policy.artifact,
        )
        self._policy = policy
        self._policy_mtime_ns = policy_mtime_ns
        self._last_reload_monotonic = time.monotonic()


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


def _policy_mtime_ns(policy_path: Path) -> int:
    return policy_path.stat().st_mtime_ns
