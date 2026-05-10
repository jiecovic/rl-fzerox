# src/rl_fzerox/ui/watch/runtime/visualization.py
from __future__ import annotations

from typing import TYPE_CHECKING

from rl_fzerox.ui.watch.runtime.cnn import (
    CnnActivationNormalizationMode,
    CnnActivationSampler,
    CnnActivationSnapshot,
)
from rl_fzerox.ui.watch.runtime.snapshots import (
    _policy_auxiliary_state_predictions,
    _policy_auxiliary_state_targets,
)

if TYPE_CHECKING:
    from fzerox_emulator import FZeroXTelemetry
    from rl_fzerox.core.envs.observations import ObservationValue
    from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName
    from rl_fzerox.core.training.inference import PolicyRunner
    from rl_fzerox.ui.watch.runtime.cnn import _CnnActivationRunner


def refresh_paused_cnn_activations(
    *,
    current_activations: CnnActivationSnapshot | None,
    cnn_sampler: CnnActivationSampler,
    cnn_visualization_enabled: bool,
    previous_cnn_visualization_enabled: bool,
    cnn_normalization: CnnActivationNormalizationMode,
    previous_cnn_normalization: CnnActivationNormalizationMode,
    policy_runner: _CnnActivationRunner | None,
    observation: ObservationValue,
) -> tuple[CnnActivationSnapshot | None, bool]:
    next_activations = cnn_sampler.capture(
        enabled=cnn_visualization_enabled,
        policy_runner=policy_runner,
        observation=observation,
        normalization=cnn_normalization,
        force_refresh=(
            cnn_visualization_enabled != previous_cnn_visualization_enabled
            or cnn_normalization != previous_cnn_normalization
        ),
    )
    return next_activations, next_activations is not current_activations


def current_auxiliary_predictions(
    *,
    policy_runner: PolicyRunner | None,
    enabled: bool,
    observation: ObservationValue,
    target_names: tuple[AuxiliaryStateTargetName, ...],
) -> dict[str, object] | None:
    if not enabled:
        return None
    return _policy_auxiliary_state_predictions(
        policy_runner=policy_runner,
        observation=observation,
        target_names=target_names,
    )


def current_auxiliary_targets(
    *,
    telemetry: FZeroXTelemetry | None,
    enabled: bool,
    target_names: tuple[AuxiliaryStateTargetName, ...],
) -> dict[str, object] | None:
    if not enabled:
        return None
    if telemetry is None:
        return {}
    return _policy_auxiliary_state_targets(telemetry, target_names=target_names)
