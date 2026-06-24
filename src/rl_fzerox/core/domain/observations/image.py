# src/rl_fzerox/core/domain/observations/image.py
"""Shared observation-image geometry vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

type ObservationPresetName = Literal[
    "crop_72x96",
    "crop_84x84",
]
type ObservationRendererName = Literal["angrylion", "gliden64"]


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
