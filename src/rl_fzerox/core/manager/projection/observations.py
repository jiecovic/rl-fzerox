# src/rl_fzerox/core/manager/projection/observations.py
"""Managed-run to training observation-config projection."""

from __future__ import annotations

from rl_fzerox.core.manager.run_spec import (
    ManagedRunConfig,
    ManagedStateComponentConfig,
)
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.training.session.model.compatibility import resume_observation_signature


def build_observation_data(config: ManagedRunConfig) -> dict[str, object]:
    observation = config.observation
    return {
        "mode": "image_state",
        "resolution": observation.resolution.model_dump(mode="python"),
        "frame_stack": observation.frame_stack,
        "stack_mode": observation.stack_mode,
        "minimap_layer": observation.minimap_layer,
        "resize_filter": observation.resize_filter,
        "minimap_resize_filter": observation.minimap_resize_filter,
        "state_components": [
            _state_component_data(component) for component in observation.state_components
        ],
    }


def component_feature_names(
    component: ManagedStateComponentConfig,
    *,
    split_lean_history: bool,
) -> tuple[str, ...]:
    from rl_fzerox.core.envs.observations.state.components import state_component_features

    settings = component.data()
    return tuple(
        feature.name
        for feature in state_component_features(
            settings,
            split_lean_history=split_lean_history,
        )
    )


def build_state_feature_dropout_groups(config: ManagedRunConfig) -> list[dict[str, object]]:
    feature_overrides = {
        feature.name: feature for feature in config.observation.state_feature_dropouts
    }
    groups: list[dict[str, object]] = []
    split_lean_history = config.action.lean_output_mode != "three_way"
    for component in config.observation.state_components:
        feature_names = component_feature_names(
            component,
            split_lean_history=split_lean_history,
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
    return resume_observation_signature(train_config)


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
    if component.included_features is not None:
        data["included_features"] = component.included_features
    return data
