# src/rl_fzerox/core/training/inference/__init__.py
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from gymnasium import spaces

from fzerox_emulator.arrays import ActionMask, PolicyState
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.observations import ImageStateObservation, ObservationValue
from rl_fzerox.core.policy.auxiliary_state import (
    AuxiliaryStateTargetName,
    auxiliary_state_target_spec,
)
from rl_fzerox.core.policy.auxiliary_state.observations import (
    auxiliary_state_targets_field,
    mapping_has_auxiliary_state_targets,
)
from rl_fzerox.core.training.inference.activations import (
    PolicyCnnActivation,
    collect_policy_cnn_activations,
)
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
    device: str = "cpu"
    algorithm: str | None = None
    curriculum_stage_index: int | None = None
    curriculum_stage_name: str | None = None
    num_timesteps: int | None = None
    lineage_num_timesteps: int | None = None


class _AuxiliaryStatePredictor(Protocol):
    def predict_auxiliary_state(
        self,
        observation: ObservationValue,
        *,
        state: PolicyState = None,
        episode_start: np.ndarray | None = None,
        target_names: tuple[AuxiliaryStateTargetName, ...] | None = None,
    ) -> dict[str, object]: ...


class PolicyRunner:
    """Small inference wrapper around one saved policy artifact."""

    def __init__(self, loaded_policy: LoadedPolicy, policy: object) -> None:
        self._loaded_policy = loaded_policy
        self._policy = policy
        self._supports_action_masks = _policy_supports_action_masks(policy)
        self._policy_mtime_ns = _policy_mtime_ns(loaded_policy.policy_path)
        self._policy_metadata_mtime_ns = _policy_metadata_mtime_ns(loaded_policy.policy_path)
        self._last_reload_monotonic = time.monotonic()
        self._next_refresh_check_monotonic = 0.0
        self._reload_error: str | None = None
        self._last_reload_error: str | None = None
        self._predict_state: PolicyState = None
        self._episode_start = np.array([True], dtype=bool)

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

    @property
    def checkpoint_num_timesteps(self) -> int | None:
        """Return the lineage-aware experience saved with the current checkpoint, if any."""

        return self._loaded_policy.lineage_num_timesteps or self._loaded_policy.num_timesteps

    @property
    def checkpoint_local_num_timesteps(self) -> int | None:
        """Return checkpoint timesteps local to the loaded run."""

        return self._loaded_policy.num_timesteps

    @property
    def supports_action_masks(self) -> bool:
        """Return whether the loaded policy accepts action_masks during prediction."""

        return self._supports_action_masks

    def refresh(self) -> None:
        """Reload artifact metadata if the watched policy checkpoint changed on disk."""

        self._maybe_reload()

    def refresh_if_due(self, *, interval_seconds: float) -> None:
        """Refresh at most once per interval unless the interval is zero."""

        now = time.monotonic()
        if now < self._next_refresh_check_monotonic:
            return
        self._next_refresh_check_monotonic = now + max(0.0, float(interval_seconds))
        self.refresh()

    def reset(self) -> None:
        """Reset any recurrent inference state for a fresh episode."""

        self._predict_state = None
        self._episode_start = np.array([True], dtype=bool)

    def predict(
        self,
        observation: ObservationValue,
        *,
        deterministic: bool = True,
        action_masks: ActionMask | None = None,
        refresh: bool = True,
    ) -> ActionValue:
        """Predict one action for the current observation."""

        if refresh:
            self._maybe_reload()
        policy_observation = self._observation_for_policy(observation)
        action, next_state = _predict_policy_action(
            self._policy,
            policy_observation,
            state=self._predict_state,
            episode_start=self._episode_start,
            deterministic=deterministic,
            action_masks=action_masks if self._supports_action_masks else None,
        )
        self._predict_state = next_state
        self._episode_start = np.array([False], dtype=bool)
        return action

    def cnn_activations(
        self,
        observation: ObservationValue,
    ) -> tuple[PolicyCnnActivation, ...]:
        """Return watch/debug CNN activations for the current policy."""

        policy_observation = self._observation_for_policy(observation)
        return collect_policy_cnn_activations(self._policy, policy_observation)

    def auxiliary_state_predictions(
        self,
        observation: ObservationValue,
        *,
        target_names: tuple[AuxiliaryStateTargetName, ...] | None = None,
    ) -> dict[str, object] | None:
        """Return decoded auxiliary-state predictions for the current observation."""

        predictor = _policy_auxiliary_state_predictor(self._policy)
        if predictor is None:
            return None
        return predictor.predict_auxiliary_state(
            self._observation_for_policy(observation),
            state=self._predict_state,
            episode_start=self._episode_start,
            target_names=target_names,
        )

    def _observation_for_policy(self, observation: ObservationValue) -> ObservationValue:
        observation_space = _policy_observation_space(self._policy)
        if not isinstance(observation_space, spaces.Dict):
            return observation
        dict_space = observation_space
        field_name = auxiliary_state_targets_field()
        dict_fields = getattr(dict_space, "spaces", {})
        if not isinstance(dict_fields, Mapping) or field_name not in dict_fields:
            return observation
        if not isinstance(observation, dict):
            raise TypeError("Auxiliary-state policy expects dict observations")
        if mapping_has_auxiliary_state_targets(observation):
            return observation

        zeros = np.zeros(auxiliary_state_target_spec().count, dtype=np.float32)
        augmented_observation: ImageStateObservation = {
            "image": observation["image"],
            "state": observation["state"],
            "auxiliary_state_targets": zeros,
        }
        return augmented_observation

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
                policy = _load_saved_policy(
                    policy_path,
                    run_dir=self._loaded_policy.run_dir,
                    device=self._loaded_policy.device,
                    algorithm=self._loaded_policy.algorithm,
                )
            except Exception as exc:
                self._reload_error = str(exc)
                self._last_reload_error = self._reload_error
                return
            self._policy = policy
            self._supports_action_masks = _policy_supports_action_masks(policy)
            self._policy_mtime_ns = policy_mtime_ns
            self._last_reload_monotonic = time.monotonic()
            self._reload_error = None
            self.reset()

        self._loaded_policy = LoadedPolicy(
            run_dir=self._loaded_policy.run_dir,
            policy_path=policy_path,
            artifact=self._loaded_policy.artifact,
            device=self._loaded_policy.device,
            algorithm=self._loaded_policy.algorithm,
            **_loaded_policy_metadata_fields(policy_path=policy_path),
        )
        self._policy_metadata_mtime_ns = policy_metadata_mtime_ns


def load_policy_runner(
    run_dir: Path,
    *,
    artifact: str = "latest",
    device: str = "cpu",
    algorithm: str | None = None,
) -> PolicyRunner:
    """Load one saved policy artifact from one run directory."""

    resolved_run_dir = run_dir.expanduser().resolve()
    policy_path = resolve_policy_artifact_path(resolved_run_dir, artifact=artifact)
    policy = _load_saved_policy(
        policy_path,
        run_dir=resolved_run_dir,
        device=device,
        algorithm=algorithm,
    )
    return PolicyRunner(
        LoadedPolicy(
            run_dir=resolved_run_dir,
            policy_path=policy_path,
            artifact=artifact,
            device=device,
            algorithm=algorithm,
            **_loaded_policy_metadata_fields(policy_path=policy_path),
        ),
        policy=policy,
    )


__all__ = [
    "LoadedPolicy",
    "PolicyCnnActivation",
    "PolicyRunner",
    "load_policy_runner",
]


def _policy_observation_space(policy: object) -> spaces.Space | None:
    observation_space = getattr(policy, "observation_space", None)
    if isinstance(observation_space, spaces.Space):
        return observation_space
    inner_policy = getattr(policy, "policy", None)
    inner_observation_space = getattr(inner_policy, "observation_space", None)
    if isinstance(inner_observation_space, spaces.Space):
        return inner_observation_space
    return None


def _policy_auxiliary_state_predictor(policy: object) -> _AuxiliaryStatePredictor | None:
    predictor = getattr(policy, "predict_auxiliary_state", None)
    if callable(predictor):
        return policy  # pyright: ignore[reportReturnType]
    inner_policy = getattr(policy, "policy", None)
    inner_predictor = getattr(inner_policy, "predict_auxiliary_state", None)
    if callable(inner_predictor):
        return inner_policy  # pyright: ignore[reportReturnType]
    return None
