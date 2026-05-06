# src/rl_fzerox/core/domain/observation_image.py
"""Shared observation-image geometry vocabulary.

Manager run-spec, derived runtime manifests, and the frontend all need the same
small set of preset names plus a bounded custom-resolution escape hatch. This
module owns that common language so shape math does not drift across layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

ObservationPresetName: TypeAlias = Literal[
    "crop_60x76",
    "crop_84x84",
]
ObservationResolutionMode: TypeAlias = Literal["preset", "custom"]
ObservationResizeFilter: TypeAlias = Literal["nearest", "bilinear"]
ObservationRendererName: TypeAlias = Literal["angrylion", "gliden64"]

MIN_CUSTOM_OBSERVATION_DIMENSION: Final[int] = 32
MAX_CUSTOM_OBSERVATION_HEIGHT: Final[int] = 208
MAX_CUSTOM_OBSERVATION_WIDTH: Final[int] = 592


@dataclass(frozen=True, slots=True)
class ObservationPresetGeometry:
    """Stable preset geometry exposed by the native image path."""

    name: ObservationPresetName
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class ObservationSourceGeometry:
    """Native cropped framebuffer size before the resize step."""

    renderer: ObservationRendererName
    height: int
    width: int


class ObservationCustomResolution(BaseModel):
    """Bounded custom image geometry accepted by the managed config surface."""

    model_config = ConfigDict(extra="forbid")

    height: int = Field(
        ge=MIN_CUSTOM_OBSERVATION_DIMENSION,
        le=MAX_CUSTOM_OBSERVATION_HEIGHT,
    )
    width: int = Field(
        ge=MIN_CUSTOM_OBSERVATION_DIMENSION,
        le=MAX_CUSTOM_OBSERVATION_WIDTH,
    )


OBSERVATION_PRESET_GEOMETRIES: Final[tuple[ObservationPresetGeometry, ...]] = (
    ObservationPresetGeometry("crop_60x76", 60, 76),
    ObservationPresetGeometry("crop_84x84", 84, 84),
)
OBSERVATION_PRESET_GEOMETRY_BY_NAME: Final[MappingProxyType[str, ObservationPresetGeometry]] = (
    MappingProxyType({geometry.name: geometry for geometry in OBSERVATION_PRESET_GEOMETRIES})
)
OBSERVATION_SOURCE_GEOMETRIES: Final[tuple[ObservationSourceGeometry, ...]] = (
    ObservationSourceGeometry("angrylion", 208, 592),
    ObservationSourceGeometry("gliden64", 208, 296),
)


def custom_resolution_label(height: int, width: int) -> str:
    """Return a stable human/debug label for one custom image geometry."""

    return f"custom_{height}x{width}"


def preset_geometry(preset: ObservationPresetName) -> tuple[int, int]:
    """Return `(height, width)` for one stable preset name."""

    geometry = OBSERVATION_PRESET_GEOMETRY_BY_NAME.get(preset)
    if geometry is None:
        raise ValueError(f"Unsupported observation preset: {preset!r}")
    return geometry.height, geometry.width


def resolve_observation_geometry(
    *,
    resolution_mode: ObservationResolutionMode,
    preset: ObservationPresetName,
    custom_resolution: ObservationCustomResolution | None,
) -> tuple[int, int]:
    """Return the active `(height, width)` pair for one observation config."""

    if resolution_mode == "preset":
        return preset_geometry(preset)
    if custom_resolution is None:
        raise ValueError("custom_resolution must be set when resolution_mode='custom'")
    return int(custom_resolution.height), int(custom_resolution.width)
