# src/fzerox_emulator/arrays.py
from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

NumpyArray: TypeAlias = NDArray[np.generic]
BoolArray: TypeAlias = NDArray[np.bool_]
UInt8Array: TypeAlias = NDArray[np.uint8]
Float32Array: TypeAlias = NDArray[np.float32]
Int64Array: TypeAlias = NDArray[np.int64]

RgbFrame: TypeAlias = UInt8Array
ObservationFrame: TypeAlias = UInt8Array
StateVector: TypeAlias = Float32Array
ActionMask: TypeAlias = BoolArray
ContinuousAction: TypeAlias = Float32Array
DiscreteAction: TypeAlias = Int64Array
PolicyState: TypeAlias = tuple[NumpyArray, ...] | None
