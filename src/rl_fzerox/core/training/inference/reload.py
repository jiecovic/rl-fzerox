# src/rl_fzerox/core/training/inference/reload.py
"""Hot-reload helpers for long-running policy inference sessions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rl_fzerox.core.training.inference.loader import (
    _load_saved_policy,
    _policy_mtime_ns,
)
from rl_fzerox.core.training.inference.metadata import (
    _loaded_policy_metadata_fields,
    _policy_metadata_mtime_ns,
)
from rl_fzerox.core.training.inference.types import LoadedPolicy
from rl_fzerox.core.training.runs import resolve_policy_artifact_path


@dataclass(frozen=True)
class PolicyReloadResult:
    policy: object
    policy_changed: bool


class PolicyHotReloader:
    """Track checkpoint metadata and reload policy artifacts when they change."""

    def __init__(self, loaded_policy: LoadedPolicy) -> None:
        self._loaded_policy = loaded_policy
        self._policy_mtime_ns = _policy_mtime_ns(loaded_policy.policy_path)
        self._policy_metadata_mtime_ns = _policy_metadata_mtime_ns(loaded_policy.policy_path)
        self._last_reload_monotonic = time.monotonic()
        self._next_refresh_check_monotonic = 0.0
        self._reload_error: str | None = None
        self._last_reload_error: str | None = None

    @property
    def loaded_policy(self) -> LoadedPolicy:
        return self._loaded_policy

    @property
    def reload_age_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._last_reload_monotonic)

    @property
    def reload_error(self) -> str | None:
        return self._reload_error

    @property
    def last_reload_error(self) -> str | None:
        return self._last_reload_error

    @property
    def policy_mtime_ns(self) -> int:
        return self._policy_mtime_ns

    @property
    def policy_mtime_utc(self) -> str:
        return _mtime_utc(self._loaded_policy.policy_path)

    def refresh(self, policy: object) -> PolicyReloadResult:
        """Reload policy weights and metadata if the watched artifact moved or changed.

        Metadata may change without a policy file rewrite, for example when a
        manager updates checkpoint step counts after export. Keep metadata fresh
        even when the in-memory policy object can be reused.
        """

        if self._loaded_policy.reload_source == "artifact":
            try:
                policy_path = resolve_policy_artifact_path(
                    self._loaded_policy.run_dir,
                    artifact=self._loaded_policy.artifact,
                )
            except FileNotFoundError:
                return PolicyReloadResult(policy=policy, policy_changed=False)
        else:
            policy_path = self._loaded_policy.policy_path

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
            return PolicyReloadResult(policy=policy, policy_changed=False)

        next_policy = policy
        if policy_changed:
            try:
                next_policy = _load_saved_policy(
                    policy_path,
                    run_dir=self._loaded_policy.run_dir,
                    model_path=self._loaded_policy.model_path,
                    device=self._loaded_policy.device,
                    algorithm=self._loaded_policy.algorithm,
                )
            except Exception as exc:
                self._reload_error = str(exc)
                self._last_reload_error = self._reload_error
                return PolicyReloadResult(policy=policy, policy_changed=False)
            self._policy_mtime_ns = policy_mtime_ns
            self._last_reload_monotonic = time.monotonic()
            self._reload_error = None

        self._loaded_policy = LoadedPolicy(
            run_dir=self._loaded_policy.run_dir,
            policy_path=policy_path,
            artifact=self._loaded_policy.artifact,
            reload_source=self._loaded_policy.reload_source,
            model_path=self._loaded_policy.model_path,
            device=self._loaded_policy.device,
            algorithm=self._loaded_policy.algorithm,
            **_loaded_policy_metadata_fields(policy_path=policy_path),
        )
        self._policy_metadata_mtime_ns = policy_metadata_mtime_ns
        return PolicyReloadResult(policy=next_policy, policy_changed=policy_changed)

    def refresh_if_due(self, policy: object, *, interval_seconds: float) -> PolicyReloadResult:
        now = time.monotonic()
        if now < self._next_refresh_check_monotonic:
            return PolicyReloadResult(policy=policy, policy_changed=False)
        self._next_refresh_check_monotonic = now + max(0.0, float(interval_seconds))
        return self.refresh(policy)


def _mtime_utc(path: Path) -> str:
    return (
        datetime.fromtimestamp(path.stat().st_mtime, UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
