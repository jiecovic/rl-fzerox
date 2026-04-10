# src/rl_fzerox/core/training/inference/__init__.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.training.inference.loader import (
    _load_saved_policy,
    _policy_mtime_ns,
    _policy_supports_action_masks,
    _predict_policy_action,
)
from rl_fzerox.core.training.inference.metadata import (
    _loaded_policy_metadata_fields,
    _policy_metadata_mtime_ns,
)
from rl_fzerox.core.training.runs import resolve_policy_artifact_path


@dataclass(frozen=True)
class LoadedPolicy:
    """Resolved policy-only artifact metadata for watch mode."""

    run_dir: Path
    policy_path: Path
    artifact: str
    curriculum_stage_index: int | None = None
    curriculum_stage_name: str | None = None


class PolicyRunner:
    """Small inference wrapper around one saved policy artifact."""

    def __init__(self, loaded_policy: LoadedPolicy, policy: object) -> None:
        self._loaded_policy = loaded_policy
        self._policy = policy
        self._supports_action_masks = _policy_supports_action_masks(policy)
        self._policy_mtime_ns = _policy_mtime_ns(loaded_policy.policy_path)
        self._policy_metadata_mtime_ns = _policy_metadata_mtime_ns(loaded_policy.policy_path)
        self._last_reload_monotonic = time.monotonic()
        self._reload_error: str | None = None
        self._last_reload_error: str | None = None

    @property
    def label(self) -> str:
        """Return a short label for the currently loaded run."""

        return self._loaded_policy.run_dir.name

    @property
    def reload_age_seconds(self) -> float:
        """Return how long ago the current policy artifact was loaded."""

        return max(0.0, time.monotonic() - self._last_reload_monotonic)

    @property
    def reload_error(self) -> str | None:
        """Return the latest hot-reload failure, if any."""

        return self._reload_error

    @property
    def last_reload_error(self) -> str | None:
        """Return the last hot-reload failure, even if a later reload succeeded."""

        return self._last_reload_error

    @property
    def checkpoint_curriculum_stage(self) -> str | None:
        """Return the curriculum stage saved with the current checkpoint, if any."""

        return self._loaded_policy.curriculum_stage_name

    @property
    def checkpoint_curriculum_stage_index(self) -> int | None:
        """Return the curriculum stage index saved with the current checkpoint, if any."""

        return self._loaded_policy.curriculum_stage_index

    def refresh(self) -> None:
        """Reload artifact metadata if the watched policy checkpoint changed on disk."""

        self._maybe_reload()

    def predict(
        self,
        observation: ObservationValue,
        *,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict one action for the current observation."""

        self._maybe_reload()
        action, _ = _predict_policy_action(
            self._policy,
            observation,
            deterministic=deterministic,
            action_masks=action_masks if self._supports_action_masks else None,
        )
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
        policy_changed = (
            policy_path != self._loaded_policy.policy_path
            or policy_mtime_ns != self._policy_mtime_ns
        )
        policy_metadata_mtime_ns = _policy_metadata_mtime_ns(policy_path)
        metadata_changed = (
            policy_path != self._loaded_policy.policy_path
            or policy_metadata_mtime_ns != self._policy_metadata_mtime_ns
        )
        if not policy_changed and not metadata_changed:
            return

        if policy_changed:
            try:
                policy = _load_saved_policy(policy_path, run_dir=self._loaded_policy.run_dir)
            except Exception as exc:
                self._reload_error = str(exc)
                self._last_reload_error = self._reload_error
                return
            self._policy = policy
            self._supports_action_masks = _policy_supports_action_masks(policy)
            self._policy_mtime_ns = policy_mtime_ns
            self._last_reload_monotonic = time.monotonic()
            self._reload_error = None

        self._loaded_policy = LoadedPolicy(
            run_dir=self._loaded_policy.run_dir,
            policy_path=policy_path,
            artifact=self._loaded_policy.artifact,
            **_loaded_policy_metadata_fields(policy_path),
        )
        self._policy_metadata_mtime_ns = policy_metadata_mtime_ns


def load_policy_runner(run_dir: Path, *, artifact: str = "latest") -> PolicyRunner:
    """Load one saved policy artifact from one run directory."""

    resolved_run_dir = run_dir.expanduser().resolve()
    policy_path = resolve_policy_artifact_path(resolved_run_dir, artifact=artifact)
    policy = _load_saved_policy(policy_path, run_dir=resolved_run_dir)
    return PolicyRunner(
        LoadedPolicy(
            run_dir=resolved_run_dir,
            policy_path=policy_path,
            artifact=artifact,
            **_loaded_policy_metadata_fields(policy_path),
        ),
        policy=policy,
    )


__all__ = [
    "LoadedPolicy",
    "PolicyRunner",
    "_load_saved_policy",
    "_policy_supports_action_masks",
    "_predict_policy_action",
    "load_policy_runner",
    "resolve_policy_artifact_path",
]
