# src/rl_fzerox/core/manager/architecture/preview/state.py
from __future__ import annotations

from rl_fzerox.core.manager.architecture.metadata import component_features, preset_geometry
from rl_fzerox.core.manager.architecture.models import ShapePreview, StateFeaturePreview
from rl_fzerox.core.manager.run_spec import ManagedRunConfig

STACK_MODE_CHANNELS = {"rgb": 3, "gray": 1, "luma_chroma": 2}


def image_shape_preview(config: ManagedRunConfig) -> ShapePreview:
    height, width = preset_geometry(config.observation.preset)
    channels_per_frame = STACK_MODE_CHANNELS[config.observation.stack_mode]
    channels = (channels_per_frame * int(config.observation.frame_stack)) + (
        1 if config.observation.minimap_layer else 0
    )
    return ShapePreview(height=height, width=width, channels=channels)


def state_feature_previews(config: ManagedRunConfig) -> tuple[StateFeaturePreview, ...]:
    feature_dropouts = {
        feature.name: float(feature.dropout_prob)
        for feature in config.observation.state_feature_dropouts
    }
    independent_lean_buttons = config.action.lean_output_mode == "independent_buttons"
    features: list[StateFeaturePreview] = []
    for component in config.observation.state_components:
        for feature in component_features(
            component,
            independent_lean_buttons=independent_lean_buttons,
        ):
            features.append(
                StateFeaturePreview(
                    component=component.name,
                    name=feature.name,
                    dropout_prob=feature_dropouts.get(feature.name, 0.0),
                )
            )
    return tuple(features)
