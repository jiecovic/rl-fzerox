# src/rl_fzerox/core/domain/observation_image.py
"""Shared observation-image geometry vocabulary.

Manager run-spec, derived runtime manifests, and the frontend all need the same
small set of preset names plus a bounded custom-resolution escape hatch. This
module owns that common language so shape math does not drift across layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

ObservationPresetName: TypeAlias = Literal[
    "crop_72x96",
    "crop_84x84",
]
ObservationResolutionMode: TypeAlias = Literal["preset", "custom", "source_crop"]
ObservationResizeFilter: TypeAlias = Literal["nearest", "bilinear"]
ObservationRendererName: TypeAlias = Literal["angrylion", "gliden64"]


@dataclass(frozen=True, slots=True)
class ObservationPresetGeometry:
    """Stable preset geometry exposed by the native image path."""

    name: ObservationPresetName
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class CustomResolutionBounds:
    """Accepted custom observation-resolution bounds."""

    min_dimension: int
    max_height: int
    max_width: int


@dataclass(frozen=True, slots=True)
class ObservationSourceGeometry:
    """Native cropped framebuffer size before the resize step."""

    renderer: ObservationRendererName
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class ObservationImageGeometry:
    """Supported image-observation geometry choices."""

    custom_bounds: CustomResolutionBounds
    presets: tuple[ObservationPresetGeometry, ...]
    default_preset: ObservationPresetName
    source_geometries: tuple[ObservationSourceGeometry, ...]


OBSERVATION_IMAGE_GEOMETRY: Final[ObservationImageGeometry] = ObservationImageGeometry(
    custom_bounds=CustomResolutionBounds(
        min_dimension=32,
        max_height=208,
        max_width=592,
    ),
    presets=(
        ObservationPresetGeometry("crop_72x96", 72, 96),
        ObservationPresetGeometry("crop_84x84", 84, 84),
    ),
    default_preset="crop_84x84",
    source_geometries=(
        ObservationSourceGeometry("angrylion", 208, 592),
        ObservationSourceGeometry("gliden64", 208, 296),
    ),
)
OBSERVATION_PRESET_GEOMETRY_BY_NAME: Final[MappingProxyType[str, ObservationPresetGeometry]] = (
    MappingProxyType({geometry.name: geometry for geometry in OBSERVATION_IMAGE_GEOMETRY.presets})
)


class PresetResolutionChoice(BaseModel):
    """Preset-backed observation resolution."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["preset"] = "preset"
    preset: ObservationPresetName = OBSERVATION_IMAGE_GEOMETRY.default_preset


class CustomResolutionChoice(BaseModel):
    """Custom fixed observation resolution."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["custom"] = "custom"
    height: int = Field(
        ge=OBSERVATION_IMAGE_GEOMETRY.custom_bounds.min_dimension,
        le=OBSERVATION_IMAGE_GEOMETRY.custom_bounds.max_height,
    )
    width: int = Field(
        ge=OBSERVATION_IMAGE_GEOMETRY.custom_bounds.min_dimension,
        le=OBSERVATION_IMAGE_GEOMETRY.custom_bounds.max_width,
    )


class SourceCropResolutionChoice(BaseModel):
    """Use the renderer-native crop size without target downsampling."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["source_crop"] = "source_crop"


ObservationResolutionConfig: TypeAlias = Annotated[
    PresetResolutionChoice
    | CustomResolutionChoice
    | SourceCropResolutionChoice,
    Field(discriminator="mode"),
]


def preset_geometry(preset: ObservationPresetName) -> tuple[int, int]:
    """Return `(height, width)` for one stable preset name."""

    geometry = OBSERVATION_PRESET_GEOMETRY_BY_NAME.get(preset)
    if geometry is None:
        raise ValueError(f"Unsupported observation preset: {preset!r}")
    return geometry.height, geometry.width


def source_crop_geometry(renderer: ObservationRendererName) -> tuple[int, int]:
    """Return the native cropped `(height, width)` before target downsampling."""

    for geometry in OBSERVATION_IMAGE_GEOMETRY.source_geometries:
        if geometry.renderer == renderer:
            return geometry.height, geometry.width
    raise ValueError(f"Unsupported observation renderer: {renderer!r}")


def resolve_observation_geometry(
    *,
    resolution: ObservationResolutionConfig,
    renderer: ObservationRendererName | None = None,
) -> tuple[int, int]:
    """Return the active `(height, width)` pair for one observation config."""

    if isinstance(resolution, PresetResolutionChoice):
        return preset_geometry(resolution.preset)
    if isinstance(resolution, CustomResolutionChoice):
        return int(resolution.height), int(resolution.width)
    if renderer is None:
        raise ValueError("renderer must be set for source-crop observation resolution")
    return source_crop_geometry(renderer)
