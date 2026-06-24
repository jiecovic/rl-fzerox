# src/fzerox_emulator/arrays.py
"""Project-owned NumPy aliases for emulator frames, observations, and actions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type NumpyArray = NDArray[np.generic]
type BoolArray = NDArray[np.bool_]
type UInt8Array = NDArray[np.uint8]
type Float32Array = NDArray[np.float32]
type Int16Array = NDArray[np.int16]
type Int64Array = NDArray[np.int64]
type UInt16Array = NDArray[np.uint16]
type UInt32Array = NDArray[np.uint32]

type RgbFrame = UInt8Array
type RgbFrameBatch = UInt8Array
type DisplayFrames = RgbFrameBatch | tuple[RgbFrame, ...]
type ControllerMaskBatch = UInt16Array | tuple[int, ...]
type Pcm16Samples = Int16Array | tuple[int, ...]
type AudioFrameCounts = UInt32Array | tuple[int, ...]
type ObservationFrame = UInt8Array
type StateVector = Float32Array
type ActionMask = BoolArray
type ContinuousAction = Float32Array
type DiscreteAction = Int64Array
type PolicyState = tuple[NumpyArray, ...] | None
