from __future__ import annotations

from rl_fzerox.core.manager.run_spec import (
    ManagedRunConfig,
    ManagedStateComponentConfig,
)
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig


def build_observation_data(config: ManagedRunConfig) -> dict[str, object]:
    return {
        "mode": "image_state",
        "preset": config.observation.preset,
        "frame_stack": config.observation.frame_stack,
        "stack_mode": config.observation.stack_mode,
        "minimap_layer": config.observation.minimap_layer,
        "resize_filter": config.observation.resize_filter,
        "minimap_resize_filter": config.observation.minimap_resize_filter,
        "state_components": [
            _state_component_data(component) for component in config.observation.state_components
        ],
    }


def component_feature_names(
    component: ManagedStateComponentConfig,
    *,
    independent_lean_buttons: bool,
) -> tuple[str, ...]:
    from rl_fzerox.core.envs.observations.state.components import state_component_features

    settings = component.data()
    return tuple(
        feature.name
        for feature in state_component_features(
            settings,
            independent_lean_buttons=independent_lean_buttons,
        )
    )


def build_state_feature_dropout_groups(config: ManagedRunConfig) -> list[dict[str, object]]:
    feature_overrides = {
        feature.name: feature for feature in config.observation.state_feature_dropouts
    }
    groups: list[dict[str, object]] = []
    independent_lean_buttons = config.action.lean_output_mode == "independent_buttons"
    for component in config.observation.state_components:
        feature_names = component_feature_names(
            component,
            independent_lean_buttons=independent_lean_buttons,
        )
        if component.name == "course_context":
            feature_probs = [
                float(feature_override.dropout_prob)
                for feature_name in feature_names
                if (feature_override := feature_overrides.get(feature_name)) is not None
            ]
            dropout_prob = max(feature_probs, default=0.0)
            if dropout_prob > 0.0:
                groups.append(
                    {
                        "feature_names": feature_names,
                        "dropout_prob": dropout_prob,
                    }
                )
            continue
        for feature_name in feature_names:
            override = feature_overrides.get(feature_name)
            dropout_prob = 0.0 if override is None else float(override.dropout_prob)
            if dropout_prob <= 0.0:
                continue
            groups.append(
                {
                    "feature_names": (feature_name,),
                    "dropout_prob": dropout_prob,
                }
            )
    return groups


def fork_observation_signature(train_config: TrainAppConfig) -> dict[str, object]:
    observation = train_config.env.observation
    return {
        "mode": observation.mode,
        "preset": observation.preset,
        "frame_stack": observation.frame_stack,
        "stack_mode": observation.stack_mode,
        "minimap_layer": observation.minimap_layer,
        "state_components": tuple(
            component.model_dump(mode="python") for component in observation.state_components or ()
        ),
    }


def _state_component_data(component: ManagedStateComponentConfig) -> dict[str, object]:
    data: dict[str, object] = {"name": component.name}
    if component.encoding is not None:
        data["encoding"] = component.encoding
    if component.progress_source is not None:
        data["progress_source"] = component.progress_source
    if component.length is not None:
        data["length"] = component.length
    if component.controls is not None:
        data["controls"] = component.controls
    return data
