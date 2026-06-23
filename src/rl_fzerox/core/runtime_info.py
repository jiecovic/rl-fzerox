# src/rl_fzerox/core/runtime_info.py
"""Typed accessors for emulator/runtime info mappings."""

from __future__ import annotations

import math
from collections.abc import Mapping


def bool_info(
    info: Mapping[object, object],
    key: str,
    *,
    numeric: bool = False,
) -> bool:
    """Return a boolean runtime flag.

    Runtime info sometimes stores flags as native booleans and telemetry payloads
    sometimes round-trip them as numeric values. Callers must opt into numeric
    coercion so state flags do not accidentally treat arbitrary objects as true.
    """

    value = info.get(key)
    if isinstance(value, bool):
        return value
    if numeric and isinstance(value, int | float):
        return bool(value)
    return False


def optional_int_info(
    info: Mapping[object, object],
    key: str,
    *,
    numeric_bool: bool = False,
    float_values: bool = True,
    minimum: int | None = None,
) -> int | None:
    value = info.get(key)
    if isinstance(value, bool):
        if not numeric_bool:
            return None
        number = int(value)
    elif isinstance(value, int):
        number = int(value)
    elif float_values and isinstance(value, float):
        number = int(value)
    else:
        return None
    return max(minimum, number) if minimum is not None else number


def int_info(
    info: Mapping[object, object],
    key: str,
    *,
    default: int = 0,
    numeric_bool: bool = False,
    float_values: bool = True,
    minimum: int | None = None,
) -> int:
    value = optional_int_info(
        info,
        key,
        numeric_bool=numeric_bool,
        float_values=float_values,
        minimum=minimum,
    )
    return default if value is None else value


def optional_float_info(
    info: Mapping[object, object],
    key: str,
    *,
    numeric_bool: bool = False,
    finite: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    value = info.get(key)
    if isinstance(value, bool):
        if not numeric_bool:
            return None
        number = float(value)
    elif isinstance(value, int | float):
        number = float(value)
    else:
        return None
    if finite and not math.isfinite(number):
        return None
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def float_info(
    info: Mapping[object, object],
    key: str,
    *,
    default: float = 0.0,
    numeric_bool: bool = False,
    finite: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = optional_float_info(
        info,
        key,
        numeric_bool=numeric_bool,
        finite=finite,
        minimum=minimum,
        maximum=maximum,
    )
    return default if value is None else value


def str_info(
    info: Mapping[object, object],
    key: str,
    *,
    default: str = "",
    strip: bool = False,
) -> str:
    value = optional_str_info(info, key, strip=strip)
    return default if value is None else value


def optional_str_info(
    info: Mapping[object, object],
    key: str,
    *,
    strip: bool = False,
    non_empty: bool = False,
) -> str | None:
    value = info.get(key)
    if not isinstance(value, str):
        return None
    if strip:
        value = value.strip()
    if non_empty and not value:
        return None
    return value
