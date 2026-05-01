# src/rl_fzerox/core/manager/architecture_metadata.py
from __future__ import annotations

from collections.abc import Iterable

from rl_fzerox.core.envs.observations.state.components import state_component_definition
from rl_fzerox.core.envs.observations.state.types import StateFeature
from rl_fzerox.core.manager.architecture_models import (
    ObservationPresetInfo,
    RunManagerConfigMetadata,
    SelectOption,
    StateComponentInfo,
    StateFeatureInfo,
)
from rl_fzerox.core.manager.config import (
    ManagedStateComponentConfig,
    ObservationPreset,
    default_state_components,
)


def run_manager_config_metadata() -> RunManagerConfigMetadata:
    """Return stable manager options derived from backend-supported config values."""

    return RunManagerConfigMetadata(
        observation_presets=tuple(
            ObservationPresetInfo(
                value=value,
                label=value.replace("crop_", "").replace("x", " x "),
                height=height,
                width=width,
            )
            for value, height, width in OBSERVATION_PRESET_GEOMETRIES
        ),
        stack_modes=_options(("rgb", "gray", "luma_chroma")),
        resize_filters=_options(("nearest", "bilinear")),
        progress_sources=_options(("lap_progress", "segment_progress", "none")),
        component_modes=_options(("include", "zero", "exclude")),
        action_history_controls=_options(
            ("steer", "thrust", "air_brake", "boost", "lean", "pitch")
        ),
        state_components=tuple(
            StateComponentInfo(
                name=component.name,
                label=component.name.replace("_", " "),
                default_mode=component.mode,
                features=tuple(
                    StateFeatureInfo(name=feature.name, low=feature.low, high=feature.high)
                    for feature in component_features(component)
                ),
            )
            for component in default_state_components()
        ),
        conv_profiles=_options(
            (
                "auto",
                "nature",
                "nature_32_64_128",
                "nature_wide",
                "nature_extra_k3",
                "compact_deep",
                "compact_bottleneck",
                "tiny_256",
            )
        ),
        activation_functions=_options(("relu", "gelu", "tanh")),
        net_arch_presets=(
            SelectOption(value="256,128", label="[256, 128]"),
            SelectOption(value="512,256", label="[512, 256]"),
            SelectOption(value="256", label="[256]"),
            SelectOption(value="128", label="[128]"),
        ),
    )


def component_features(component: ManagedStateComponentConfig) -> tuple[StateFeature, ...]:
    settings = component.data()
    return state_component_definition(settings).features(settings)


def preset_geometry(preset: ObservationPreset) -> tuple[int, int]:
    for value, height, width in OBSERVATION_PRESET_GEOMETRIES:
        if value == preset:
            return height, width
    raise ValueError(f"Unsupported observation preset: {preset!r}")


def _options(values: Iterable[str]) -> tuple[SelectOption, ...]:
    return tuple(SelectOption(value=value, label=value.replace("_", " ")) for value in values)


OBSERVATION_PRESET_GEOMETRIES: tuple[tuple[ObservationPreset, int, int], ...] = (
    ("crop_84x116", 84, 116),
    ("crop_92x124", 92, 124),
    ("crop_116x164", 116, 164),
    ("crop_98x130", 98, 130),
    ("crop_66x82", 66, 82),
    ("crop_60x76", 60, 76),
    ("crop_68x68", 68, 68),
    ("crop_84x84", 84, 84),
    ("crop_76x100", 76, 100),
    ("crop_64x64", 64, 64),
)
