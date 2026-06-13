# src/fzerox_emulator/arrays.py
"""Project-owned NumPy aliases for emulator frames, observations, and actions."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

NumpyArray: TypeAlias = NDArray[np.generic]
BoolArray: TypeAlias = NDArray[np.bool_]
UInt8Array: TypeAlias = NDArray[np.uint8]
Float32Array: TypeAlias = NDArray[np.float32]
Int16Array: TypeAlias = NDArray[np.int16]
Int64Array: TypeAlias = NDArray[np.int64]
UInt16Array: TypeAlias = NDArray[np.uint16]
UInt32Array: TypeAlias = NDArray[np.uint32]

RgbFrame: TypeAlias = UInt8Array
RgbFrameBatch: TypeAlias = UInt8Array
DisplayFrames: TypeAlias = RgbFrameBatch | tuple[RgbFrame, ...]
ControllerMaskBatch: TypeAlias = UInt16Array | tuple[int, ...]
Pcm16Samples: TypeAlias = Int16Array | tuple[int, ...]
AudioFrameCounts: TypeAlias = UInt32Array | tuple[int, ...]
ObservationFrame: TypeAlias = UInt8Array
StateVector: TypeAlias = Float32Array
ActionMask: TypeAlias = BoolArray
ContinuousAction: TypeAlias = Float32Array
DiscreteAction: TypeAlias = Int64Array
PolicyState: TypeAlias = tuple[NumpyArray, ...] | None
