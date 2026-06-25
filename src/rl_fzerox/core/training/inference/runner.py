# src/rl_fzerox/core/training/inference/runner.py
"""Runtime wrapper that drives a loaded policy against prepared observations."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeGuard

from fzerox_emulator.arrays import ActionMask, BoolArray, PolicyState
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.policy.auxiliary_state import (
    AuxiliaryStateTargetName,
)
from rl_fzerox.core.training.inference.activations import (
    PolicyCnnActivation,
    collect_policy_cnn_activations,
)
from rl_fzerox.core.training.inference.loader import (
    _load_saved_policy,
    _policy_supports_action_masks,
    _predict_policy_action,
)
from rl_fzerox.core.training.inference.metadata import _loaded_policy_metadata_fields
from rl_fzerox.core.training.inference.observations import observation_for_policy
from rl_fzerox.core.training.inference.recurrent import RecurrentInferenceState
from rl_fzerox.core.training.inference.reload import PolicyHotReloader
from rl_fzerox.core.training.inference.types import LoadedPolicy
from rl_fzerox.core.training.runs import resolve_policy_artifact_path


class _AuxiliaryStatePredictor(Protocol):
    def predict_auxiliary_state(
        self,
        observation: ObservationValue,
        *,
        state: PolicyState = None,
        episode_start: BoolArray | None = None,
        target_names: tuple[AuxiliaryStateTargetName, ...] | None = None,
    ) -> dict[str, object]: ...


class PolicyRunner:
    """Small inference wrapper around one saved policy artifact."""

    def __init__(self, loaded_policy: LoadedPolicy, policy: object) -> None:
        self._reloader = PolicyHotReloader(loaded_policy)
        self._policy = policy
        self._supports_action_masks = _policy_supports_action_masks(policy)
        self._recurrent = RecurrentInferenceState()

    @property
    def label(self) -> str:
        """Return a short label for the currently loaded run."""

        return self._reloader.loaded_policy.run_dir.name

    @property
    def reload_age_seconds(self) -> float:
        """Return how long ago the current policy artifact was loaded."""

        return self._reloader.reload_age_seconds

    @property
    def reload_error(self) -> str | None:
        """Return the latest hot-reload failure, if any."""

        return self._reloader.reload_error

    @property
    def last_reload_error(self) -> str | None:
        """Return the last hot-reload failure, even if a later reload succeeded."""

        return self._reloader.last_reload_error

    @property
    def checkpoint_num_timesteps(self) -> int | None:
        """Return the lineage-aware experience saved with the current checkpoint, if any."""

        loaded_policy = self._reloader.loaded_policy
        return loaded_policy.lineage_num_timesteps or loaded_policy.num_timesteps

    @property
    def checkpoint_local_num_timesteps(self) -> int | None:
        """Return checkpoint timesteps local to the loaded run."""

        return self._reloader.loaded_policy.num_timesteps

    @property
    def checkpoint_policy_path(self) -> Path:
        """Return the concrete policy checkpoint path currently loaded."""

        return self._reloader.loaded_policy.policy_path

    @property
    def checkpoint_policy_mtime_ns(self) -> int:
        """Return the file mtime for the concrete policy checkpoint."""

        return self._reloader.policy_mtime_ns

    @property
    def checkpoint_policy_mtime_utc(self) -> str:
        """Return the checkpoint file mtime as an ISO-8601 UTC timestamp."""

        return self._reloader.policy_mtime_utc

    @property
    def supports_action_masks(self) -> bool:
        """Return whether the loaded policy accepts action_masks during prediction."""

        return self._supports_action_masks

    def refresh(self) -> None:
        """Reload artifact metadata if the watched policy checkpoint changed on disk."""

        self._refresh_policy()

    def refresh_if_due(self, *, interval_seconds: float) -> None:
        """Refresh at most once per interval unless the interval is zero."""

        result = self._reloader.refresh_if_due(self._policy, interval_seconds=interval_seconds)
        self._apply_reload_result(result.policy, policy_changed=result.policy_changed)

    def reset(self) -> None:
        """Reset any recurrent inference state for a fresh episode."""

        self._recurrent.reset()

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
            self._refresh_policy()
        policy_observation = observation_for_policy(self._policy, observation)
        action, next_state = _predict_policy_action(
            self._policy,
            policy_observation,
            state=self._recurrent.predict_state,
            episode_start=self._recurrent.episode_start,
            deterministic=deterministic,
            action_masks=action_masks if self._supports_action_masks else None,
        )
        self._recurrent.advance(next_state)
        return action

    def cnn_activations(
        self,
        observation: ObservationValue,
    ) -> tuple[PolicyCnnActivation, ...]:
        """Return watch/debug CNN activations for the current policy."""

        policy_observation = observation_for_policy(self._policy, observation)
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
            observation_for_policy(self._policy, observation),
            state=self._recurrent.predict_state,
            episode_start=self._recurrent.episode_start,
            target_names=target_names,
        )

    def _refresh_policy(self) -> None:
        result = self._reloader.refresh(self._policy)
        self._apply_reload_result(result.policy, policy_changed=result.policy_changed)

    def _apply_reload_result(self, policy: object, *, policy_changed: bool) -> None:
        if not policy_changed:
            return
        self._policy = policy
        self._supports_action_masks = _policy_supports_action_masks(policy)
        self.reset()


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


def _policy_auxiliary_state_predictor(policy: object) -> _AuxiliaryStatePredictor | None:
    if _has_auxiliary_state_predictor(policy):
        return policy
    inner_policy = getattr(policy, "policy", None)
    if _has_auxiliary_state_predictor(inner_policy):
        return inner_policy
    return None


def _has_auxiliary_state_predictor(policy: object) -> TypeGuard[_AuxiliaryStatePredictor]:
    return callable(getattr(policy, "predict_auxiliary_state", None))
