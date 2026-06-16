# src/rl_fzerox/ui/watch/runtime/snapshots/observation.py
from __future__ import annotations

from typing import TYPE_CHECKING

from fzerox_emulator import FZeroXTelemetry
from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.envs import observations as observation_access
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.policy.auxiliary_state import (
    AuxiliaryStateTargetName,
    auxiliary_state_target_values,
)
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.runtime.ipc import PolicyObservationSnapshot

if TYPE_CHECKING:
    from rl_fzerox.core.training.inference import PolicyRunner


def _policy_observation_snapshot(
    *,
    config: WatchAppConfig,
    observation: ObservationValue | None,
    telemetry: FZeroXTelemetry | None,
    info: dict[str, object],
) -> PolicyObservationSnapshot | None:
    if observation is None:
        return None
    return PolicyObservationSnapshot(
        image=observation_access.observation_image(observation),
        state=observation_access.observation_state(observation),
        state_reference=_reference_observation_state(
            config=config,
            observation=observation,
            telemetry=telemetry,
            info=info,
        ),
    )


def _policy_observation_shape(
    config: WatchAppConfig,
    observation: ObservationValue | None,
) -> tuple[int, ...] | None:
    del config
    if observation is not None:
        return tuple(
            int(value) for value in observation_access.observation_image(observation).shape
        )
    return None


def _reference_observation_state(
    *,
    config: WatchAppConfig,
    observation: ObservationValue,
    telemetry: FZeroXTelemetry | None,
    info: dict[str, object],
) -> StateVector | None:
    observed_state = observation_access.observation_state(observation)
    if config.env.observation.mode != "image_state":
        return observed_state
    state_components = config.env.observation.state_components_data()
    if state_components is None:
        return observed_state
    return observation_access.telemetry_state_vector(
        telemetry,
        state_components=state_components,
        action_history=_reference_action_history(
            observed_state=observed_state,
            info=info,
        ),
        split_lean_history=config.env.action.runtime().split_lean_history,
    )


def _reference_action_history(
    *,
    observed_state: StateVector | None,
    info: dict[str, object],
) -> dict[str, float]:
    if observed_state is None:
        return {}
    raw_feature_names = info.get("observation_state_features")
    if isinstance(raw_feature_names, list):
        feature_names = tuple(str(name) for name in raw_feature_names)
    elif isinstance(raw_feature_names, tuple):
        feature_names = tuple(str(name) for name in raw_feature_names)
    else:
        return {}
    flat_state = observed_state.reshape(-1)
    if len(feature_names) != int(flat_state.size):
        return {}
    return {
        name.removeprefix("control_history."): float(value)
        for name, value in zip(feature_names, flat_state, strict=True)
        if name.startswith("control_history.")
    }


def _policy_auxiliary_state_predictions(
    *,
    policy_runner: PolicyRunner | None,
    observation: ObservationValue,
    target_names: tuple[AuxiliaryStateTargetName, ...],
) -> dict[str, object] | None:
    if policy_runner is None:
        return None
    return policy_runner.auxiliary_state_predictions(
        observation,
        target_names=target_names,
    )


def _policy_auxiliary_state_targets(
    telemetry: FZeroXTelemetry | None,
    *,
    target_names: tuple[AuxiliaryStateTargetName, ...],
) -> dict[str, object]:
    if telemetry is None:
        return {}
    all_targets = auxiliary_state_target_values(telemetry)
    return {str(name): value for name, value in all_targets.items() if name in target_names}
