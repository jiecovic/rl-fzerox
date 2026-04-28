from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ObservationStackMode
from fzerox_emulator.arrays import NumpyArray, UInt8Array

LazyReplayStackMode = Literal["rgb", "gray", "luma_chroma"]
SUPPORTED_LAZY_REPLAY_STACK_MODES: frozenset[LazyReplayStackMode] = frozenset(
    {"rgb", "gray", "luma_chroma"}
)


@dataclass(frozen=True, slots=True)
class LazyImageReplayLayout:
    """Static image-layout metadata needed to compress and rebuild stacks."""

    image_shape: tuple[int, int, int]
    state_shape: tuple[int, ...]
    frame_stack: int
    stack_mode: LazyReplayStackMode
    minimap_layer: bool
    current_slice_channels: int
    channels_first: bool

    @property
    def height(self) -> int:
        return self.image_shape[1] if self.channels_first else self.image_shape[0]

    @property
    def width(self) -> int:
        return self.image_shape[2] if self.channels_first else self.image_shape[1]

    @property
    def image_channels(self) -> int:
        return self.image_shape[0] if self.channels_first else self.image_shape[2]

    @property
    def minimap_channels(self) -> int:
        return 1 if self.minimap_layer else 0

    @property
    def stacked_frame_channels(self) -> int:
        return self.image_channels - self.minimap_channels

    def split_image_batch(
        self,
        image_batch: NumpyArray,
    ) -> tuple[UInt8Array, UInt8Array | None]:
        """Return the per-step image slice and optional minimap slice."""

        frame_end = self.stacked_frame_channels
        current_start = frame_end - self.current_slice_channels
        if self.channels_first:
            current_slice = as_uint8(image_batch[:, current_start:frame_end, :, :])
        else:
            current_slice = as_uint8(image_batch[..., current_start:frame_end])
        if not self.minimap_layer:
            return current_slice, None
        if self.channels_first:
            minimap_slice = as_uint8(image_batch[:, frame_end : frame_end + 1, :, :])
        else:
            minimap_slice = as_uint8(image_batch[..., frame_end : frame_end + 1])
        return current_slice, minimap_slice


def current_slice_channels(stack_mode: ObservationStackMode) -> int:
    if stack_mode == "rgb":
        return 3
    if stack_mode == "gray":
        return 1
    if stack_mode == "luma_chroma":
        return 2
    supported = ", ".join(sorted(SUPPORTED_LAZY_REPLAY_STACK_MODES))
    raise RuntimeError(
        f"lazy SAC replay does not support observation.stack_mode={stack_mode!r}; "
        f"use one of: {supported}"
    )


def action_storage_dtype(dtype: np.dtype | type | None) -> np.dtype | type | None:
    if dtype == np.float64:
        return np.float32
    return dtype


def is_channels_first(image_shape: tuple[int, int, int], expected_channels: int) -> bool:
    if image_shape[0] == expected_channels and image_shape[2] != expected_channels:
        return True
    if image_shape[2] == expected_channels and image_shape[0] != expected_channels:
        return False
    return image_shape[0] <= image_shape[2]


def image_shape(image_space: spaces.Box) -> tuple[int, int, int]:
    shape = image_space.shape
    if len(shape) != 3:
        raise RuntimeError("lazy SAC replay requires 3D image observations")
    return int(shape[0]), int(shape[1]), int(shape[2])


def as_uint8(array: NumpyArray) -> UInt8Array:
    return np.ascontiguousarray(array, dtype=np.uint8)
