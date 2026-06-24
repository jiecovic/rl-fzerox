# src/rl_fzerox/core/training/session/model/preload.py
from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeGuard, TypeVar

from stable_baselines3.common.utils import FloatSchedule
from torch import nn

from rl_fzerox.core.domain.policy import TrainAlgorithmName
from rl_fzerox.core.runtime_spec.schema import PolicyConfig, TrainAppConfig, TrainConfig
from rl_fzerox.core.training.runs import (
    load_train_run_config,
    load_train_run_train_config,
    resolve_model_artifact_path,
)
from rl_fzerox.core.training.session.model.action_bias import (
    TrainingEnvActionDimensions,
    apply_weights_only_action_bias_delta,
    load_action_bias_offsets_from_archive,
    set_model_action_bias_offsets,
)
from rl_fzerox.core.training.session.model.algorithms import (
    resolve_effective_training_algorithm,
    resolve_training_algorithm_class,
)
from rl_fzerox.core.training.session.model.compatibility import (
    resume_compatibility_change_labels,
)

_ModelT = TypeVar("_ModelT")
_AuxiliaryStateSignature = tuple[int, ...] | None


class _FullModelLoader(Protocol):
    @classmethod
    def load(cls, path: str, *, env: object, device: str) -> object: ...


def maybe_resume_training_model(
    *,
    model: _ModelT,
    train_env: object,
    train_config: TrainConfig,
    current_run_config: TrainAppConfig | None = None,
    policy_config: PolicyConfig | None = None,
) -> _ModelT:
    """Resume or warm-start a training model from a saved run artifact.

    `weights_only` keeps the current config authoritative and copies learned
    parameters into the freshly built model. `full_model` loads optimizer and
    scheduler state too, then attaches the current env.
    """

    if train_config.resume_run_dir is None:
        return model

    resume_run_dir = train_config.resume_run_dir.resolve()
    if current_run_config is not None and not train_config.resume_source_metadata_required:
        _validate_non_managed_resume_compatibility(
            source_config=load_train_run_config(resume_run_dir),
            current_config=current_run_config,
        )
    source_auxiliary_state_signature = _source_auxiliary_state_signature(
        train_config=train_config,
        resume_run_dir=resume_run_dir,
    )
    source_algorithm = _source_algorithm(
        train_config=train_config,
        resume_run_dir=resume_run_dir,
    )
    current_algorithm = resolve_effective_training_algorithm(train_config=train_config)
    if source_algorithm != current_algorithm:
        raise RuntimeError(
            "Resume checkpoint algorithm mismatch: "
            f"source={source_algorithm}, current={current_algorithm}. "
            "Use a checkpoint produced by the same training algorithm."
        )
    if train_config.resume_mode == "full_model":
        _validate_full_model_resume_compatibility(
            model,
            source_auxiliary_state_signature=source_auxiliary_state_signature,
        )

    model_path = resolve_model_artifact_path(
        resume_run_dir,
        artifact=train_config.resume_artifact,
    )
    if train_config.resume_mode == "full_model":
        # The algorithm check above guarantees the artifact and current model
        # are from the same SB3 family; the class factory itself is dynamic.
        algorithm_class = resolve_training_algorithm_class(current_algorithm)
        if not _is_full_model_loader(algorithm_class):
            raise TypeError("Training algorithm class does not expose load(...)")
        loaded_model = algorithm_class.load(
            str(model_path),
            env=train_env,
            device=train_config.device,
        )
        if not _is_training_model_instance(loaded_model, model):
            raise TypeError("Loaded training model type does not match the current model")
        _sync_loaded_model_training_config(
            loaded_model,
            train_config=train_config,
        )
        return loaded_model

    _set_model_parameters(
        model,
        model_path,
        exact_match=_resume_parameter_exact_match(
            model,
            source_auxiliary_state_signature=source_auxiliary_state_signature,
        ),
    )
    if policy_config is not None:
        source_action_bias_offsets = load_action_bias_offsets_from_archive(model_path)
        set_model_action_bias_offsets(model, source_action_bias_offsets)
        apply_weights_only_action_bias_delta(
            model,
            train_env=_action_dimensions_env(train_env),
            policy_config=policy_config,
            source_offsets=source_action_bias_offsets,
        )
    return model


def _validate_non_managed_resume_compatibility(
    *,
    source_config: TrainAppConfig,
    current_config: TrainAppConfig,
) -> None:
    changed = resume_compatibility_change_labels(source_config, current_config)
    if not changed:
        return
    detail = ", ".join(changed)
    raise RuntimeError(
        "Resume checkpoint is incompatible with the current run config: "
        f"{detail}. Change reward/training knobs only, use weights_only only for "
        "compatible auxiliary-head changes, or start a fresh run."
    )


def _source_auxiliary_state_signature(
    *,
    train_config: TrainConfig,
    resume_run_dir: Path,
) -> _AuxiliaryStateSignature:
    """Resolve source aux-head shape from manager metadata or saved manifests."""

    if train_config.resume_source_auxiliary_state_enabled is not None:
        if not train_config.resume_source_auxiliary_state_enabled:
            return None
        return tuple(int(value) for value in train_config.resume_source_auxiliary_state_head_arch)
    _raise_if_managed_source_metadata_missing(train_config)
    source_run_config = load_train_run_config(resume_run_dir)
    return _auxiliary_state_signature_from_config(source_run_config.policy)


def _source_algorithm(
    *,
    train_config: TrainConfig,
    resume_run_dir: Path,
) -> TrainAlgorithmName:
    """Resolve source algorithm before choosing SB3 load/set-parameter behavior."""

    if train_config.resume_source_algorithm is not None:
        return train_config.resume_source_algorithm
    _raise_if_managed_source_metadata_missing(train_config)
    source_train_config = load_train_run_train_config(resume_run_dir)
    return resolve_effective_training_algorithm(train_config=source_train_config)


def _raise_if_managed_source_metadata_missing(train_config: TrainConfig) -> None:
    if not train_config.resume_source_metadata_required:
        return
    raise RuntimeError(
        "Managed resume requires source checkpoint metadata from the manager DB. "
        "Saved YAML manifests are not used as a config source for managed runs."
    )


def _is_full_model_loader(value: object) -> TypeGuard[_FullModelLoader]:
    return callable(getattr(value, "load", None))


def _is_action_dimensions_env(value: object) -> TypeGuard[TrainingEnvActionDimensions]:
    return callable(getattr(value, "get_attr", None))


def _action_dimensions_env(value: object) -> TrainingEnvActionDimensions:
    if not _is_action_dimensions_env(value):
        raise TypeError("Training env does not expose get_attr(...)")
    return value


def _is_training_model_instance(value: object, model: _ModelT) -> TypeGuard[_ModelT]:
    return isinstance(value, type(model))


def _set_model_parameters(
    model: object,
    model_path: Path,
    *,
    exact_match: bool,
) -> None:
    set_parameters = getattr(model, "set_parameters", None)
    if not callable(set_parameters):
        raise TypeError("Training model does not expose set_parameters(...)")
    device = getattr(model, "device", "auto")
    set_parameters(str(model_path), exact_match=exact_match, device=device)


def _resume_parameter_exact_match(
    model: object,
    *,
    source_auxiliary_state_signature: _AuxiliaryStateSignature,
) -> bool:
    current_auxiliary_state_signature = _current_model_auxiliary_state_signature(model)
    return current_auxiliary_state_signature == source_auxiliary_state_signature


def _validate_full_model_resume_compatibility(
    model: object,
    *,
    source_auxiliary_state_signature: _AuxiliaryStateSignature,
) -> None:
    current_auxiliary_state_signature = _current_model_auxiliary_state_signature(model)
    if current_auxiliary_state_signature == source_auxiliary_state_signature:
        return
    raise RuntimeError(
        "Full-model resume requires the same auxiliary-state head-bank "
        "architecture as the source checkpoint. Use train.resume_mode=weights_only "
        "for aux-bank upgrades, downgrades, or trunk-width changes."
    )


def _auxiliary_state_signature_from_config(
    policy: object,
) -> _AuxiliaryStateSignature:
    auxiliary_state = getattr(policy, "auxiliary_state", None)
    if auxiliary_state is None:
        enabled = bool(getattr(policy, "auxiliary_state_enabled", False))
        head_arch = tuple(getattr(policy, "auxiliary_state_head_arch", ()))
    else:
        enabled = bool(getattr(auxiliary_state, "enabled", False))
        head_arch = tuple(getattr(auxiliary_state, "head_arch", ()))
    if not enabled:
        return None
    return tuple(int(width) for width in head_arch)


def _current_model_auxiliary_state_signature(model: object) -> _AuxiliaryStateSignature:
    """Read the current aux-head shape from the built torch module.

    We inspect the module instead of trusting config because full-model resume
    restores serialized policy internals, and weights-only resume may target a
    freshly built policy with a compatible but not identical config source.
    """

    policy = getattr(model, "policy", None)
    heads = getattr(policy, "_auxiliary_state_heads", None)
    if heads is None:
        return None
    trunk = getattr(heads, "trunk", None)
    if isinstance(trunk, nn.Identity) or trunk is None:
        return ()
    if not isinstance(trunk, nn.Sequential):
        raise TypeError("Auxiliary-state trunk must be Identity or Sequential")
    widths: list[int] = []
    for layer in trunk:
        if isinstance(layer, nn.Linear):
            widths.append(int(layer.out_features))
    return tuple(widths)


def _sync_loaded_model_training_config(
    model: object,
    *,
    train_config: TrainConfig,
) -> None:
    """Reapply the current run config after full-model resume.

    SB3 full-model load restores the serialized algorithm object exactly,
    including saved hyperparameters. That is correct for optimizer/replay state,
    but it makes YAML changes to future-training knobs such as `gamma` or
    `ent_coef` ineffective. After loading, keep the checkpoint state while
    making the current run config authoritative for the next optimization steps.
    """

    _sync_shared_training_config(model, train_config=train_config)
    _sync_loaded_ppo_training_config(model, train_config=train_config)


def _sync_shared_training_config(model: object, *, train_config: TrainConfig) -> None:
    _set_attr_if_present(model, "learning_rate", train_config.learning_rate)
    setup_lr_schedule = getattr(model, "_setup_lr_schedule", None)
    if callable(setup_lr_schedule):
        setup_lr_schedule()
    _set_attr_if_present(model, "verbose", train_config.verbose)


def _sync_loaded_ppo_training_config(model: object, *, train_config: TrainConfig) -> None:
    _set_attr_if_present(model, "batch_size", train_config.batch_size)
    _set_attr_if_present(model, "n_epochs", train_config.n_epochs)
    _set_attr_if_present(model, "gamma", train_config.gamma)
    _set_attr_if_present(model, "gae_lambda", train_config.gae_lambda)
    _set_attr_if_present(model, "ent_coef", float(train_config.ent_coef))
    _set_attr_if_present(model, "entropy_group_weights", dict(train_config.entropy_group_weights))
    _set_attr_if_present(model, "vf_coef", train_config.vf_coef)
    _set_attr_if_present(model, "max_grad_norm", train_config.max_grad_norm)
    _set_attr_if_present(model, "clip_range", FloatSchedule(train_config.clip_range))
    clip_range_vf = (
        None if train_config.clip_range_vf is None else FloatSchedule(train_config.clip_range_vf)
    )
    _set_attr_if_present(model, "clip_range_vf", clip_range_vf)
    _set_attr_if_present(model, "normalize_advantage", train_config.normalize_advantage)
    _set_attr_if_present(model, "target_kl", train_config.target_kl)
    _set_attr_if_present(model, "stats_window_size", train_config.stats_window_size)
    rollout_buffer = getattr(model, "rollout_buffer", None)
    if rollout_buffer is not None:
        _set_attr_if_present(rollout_buffer, "gamma", train_config.gamma)
        _set_attr_if_present(rollout_buffer, "gae_lambda", train_config.gae_lambda)


def _set_attr_if_present(model: object, attr: str, value: object) -> None:
    if hasattr(model, attr):
        _set_attr(model, attr, value)


def _set_attr(model: object, attr: str, value: object) -> None:
    setattr(model, attr, value)
