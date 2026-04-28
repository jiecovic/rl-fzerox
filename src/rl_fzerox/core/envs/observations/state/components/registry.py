# src/rl_fzerox/core/envs/observations/state/components/registry.py
from __future__ import annotations

from rl_fzerox.core.envs.observations.state.components.control_history import (
    control_history_component_features,
    control_history_component_values,
)
from rl_fzerox.core.envs.observations.state.components.core import StateComponentDefinition
from rl_fzerox.core.envs.observations.state.components.course import (
    course_context_component_features,
    course_context_component_values,
)
from rl_fzerox.core.envs.observations.state.components.machine import (
    machine_context_component_features,
    machine_context_component_values,
)
from rl_fzerox.core.envs.observations.state.components.surface import (
    surface_state_component_features,
    surface_state_component_values,
)
from rl_fzerox.core.envs.observations.state.components.track import (
    track_position_component_features,
    track_position_component_values,
)
from rl_fzerox.core.envs.observations.state.components.vehicle import (
    vehicle_component_features,
    vehicle_component_values,
)

STATE_COMPONENT_DEFINITIONS: dict[str, StateComponentDefinition] = {
    "vehicle_state": StateComponentDefinition(
        features=vehicle_component_features,
        values=vehicle_component_values,
    ),
    "machine_context": StateComponentDefinition(
        features=machine_context_component_features,
        values=machine_context_component_values,
    ),
    "track_position": StateComponentDefinition(
        features=track_position_component_features,
        values=track_position_component_values,
    ),
    "surface_state": StateComponentDefinition(
        features=surface_state_component_features,
        values=surface_state_component_values,
    ),
    "course_context": StateComponentDefinition(
        features=course_context_component_features,
        values=course_context_component_values,
    ),
    "control_history": StateComponentDefinition(
        features=control_history_component_features,
        values=control_history_component_values,
    ),
}
