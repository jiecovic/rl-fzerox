# src/rl_fzerox/core/training/inference/loader.py
"""Policy/model checkpoint loading and typed inference runner creation."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol, TypeGuard

from fzerox_emulator.arrays import ActionMask, BoolArray, PolicyState
from rl_fzerox.core.domain.policy import TRAINING_ALGORITHMS
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.training.runs import RUN_LAYOUT, resolve_model_artifact_path


class _HasPredict(Protocol):
    def predict(
        self,
        observation: ObservationValue,
        state: PolicyState = None,
        episode_start: BoolArray | None = None,
        deterministic: bool = True,
    ) -> tuple[ActionValue, PolicyState]: ...


class _HasMaskablePredict(Protocol):
    def predict(
        self,
        observation: ObservationValue,
        state: PolicyState = None,
        episode_start: BoolArray | None = None,
        deterministic: bool = True,
        action_masks: ActionMask | None = None,
    ) -> tuple[ActionValue, PolicyState]: ...


def _load_saved_policy(
    policy_path: Path,
    *,
    run_dir: Path | None = None,
    model_path: Path | None = None,
    device: str = "cpu",
    algorithm: str | None = None,
) -> _HasPredict:
    """Load one saved policy-only SB3 artifact."""

    _ensure_policy_dependencies_loaded()

    algorithm = _load_saved_policy_algorithm(run_dir, explicit_algorithm=algorithm)
    if algorithm in TRAINING_ALGORITHMS.full_model_policy:
        resolved_model_path = model_path
        if resolved_model_path is None:
            if run_dir is None:
                raise RuntimeError(
                    f"{algorithm} policy loading requires a model path or source run directory"
                )
            resolved_model_path = resolve_model_artifact_path(
                run_dir,
                artifact=_artifact_kind_from_policy_path(policy_path),
            )
        algorithm_class = _full_model_class_for_algorithm(algorithm)
        loaded_model = algorithm_class.load(str(resolved_model_path), device=device)
        if not _has_predict(loaded_model):
            raise TypeError("Loaded model does not expose a compatible predict(...)")
        return loaded_model

    raise ValueError(f"Unsupported saved policy algorithm: {algorithm!r}")


def _policy_mtime_ns(policy_path: Path) -> int:
    return policy_path.stat().st_mtime_ns


def _ensure_policy_dependencies_loaded() -> None:
    """Import custom policy modules before SB3 deserializes saved artifacts."""

    import_module("rl_fzerox.core.policy.extractors")
    import_module("rl_fzerox.core.policy.auxiliary_state.policies")


def _policy_predict_fn(policy: object) -> Callable[..., tuple[ActionValue, PolicyState]]:
    if not _has_predict(policy):
        raise TypeError("Policy does not expose a compatible predict(...) method")
    return policy.predict


def _predict_policy_action(
    policy: object,
    observation: ObservationValue,
    *,
    state: PolicyState,
    episode_start: BoolArray | None,
    deterministic: bool,
    action_masks: ActionMask | None,
) -> tuple[ActionValue, PolicyState]:
    if action_masks is None:
        return _policy_predict_fn(policy)(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
    if not _has_maskable_predict(policy):
        raise TypeError("Policy does not support masked action prediction")
    return policy.predict(
        observation,
        state=state,
        episode_start=episode_start,
        deterministic=deterministic,
        action_masks=action_masks,
    )


def _policy_supports_action_masks(policy: object) -> bool:
    return "action_masks" in inspect.signature(_policy_predict_fn(policy)).parameters


def _has_predict(policy: object) -> TypeGuard[_HasPredict]:
    predict = getattr(policy, "predict", None)
    return callable(predict)


def _has_maskable_predict(policy: object) -> TypeGuard[_HasMaskablePredict]:
    if not _has_predict(policy):
        return False
    return "action_masks" in inspect.signature(policy.predict).parameters


def _full_model_class_for_algorithm(algorithm: str):
    try:
        if algorithm == TRAINING_ALGORITHMS.maskable_hybrid_action_ppo:
            from sb3x import MaskableHybridActionPPO

            return MaskableHybridActionPPO
        if algorithm == TRAINING_ALGORITHMS.maskable_hybrid_recurrent_ppo:
            from sb3x import MaskableHybridRecurrentPPO

            return MaskableHybridRecurrentPPO
    except ImportError as exc:
        raise RuntimeError(
            f"Loading {algorithm} checkpoints requires sb3x in the active environment."
        ) from exc
    raise ValueError(f"Unsupported full-model policy algorithm: {algorithm!r}")


def _load_saved_policy_algorithm(
    run_dir: Path | None,
    explicit_algorithm: str | None = None,
) -> str:
    if explicit_algorithm is not None:
        return explicit_algorithm
    _ = run_dir
    raise RuntimeError(
        "Saved policy loading requires explicit policy algorithm metadata. "
        "train_manifest.yaml is a mirror and is not used as a config source."
    )


def _artifact_kind_from_policy_path(policy_path: Path) -> str:
    if policy_path.name != "policy.zip":
        raise ValueError(f"Unsupported policy artifact filename: {policy_path.name}")
    if policy_path.parent.parent.name != RUN_LAYOUT.checkpoints_dirname:
        raise ValueError(f"Unsupported policy artifact path: {policy_path}")
    artifact_kind = policy_path.parent.name
    if artifact_kind in {"latest", "best", "final"}:
        return artifact_kind
    raise ValueError(f"Unsupported policy artifact path: {policy_path}")
