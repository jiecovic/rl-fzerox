# src/rl_fzerox/core/engine_tuning/persistence/fields.py
"""Typed field decoders for engine-tuning persistence payloads."""

from __future__ import annotations

from collections.abc import Mapping

from rl_fzerox.core.engine_tuning.types import (
    EngineTunerBackend,
    EngineTunerObjective,
)


def decode_backend(raw: object) -> EngineTunerBackend | None:
    """Return a supported backend literal from persisted data."""

    if raw == "bandit":
        return "bandit"
    if raw == "gaussian_process":
        return "gaussian_process"
    if raw == "mlp_ensemble":
        return "mlp_ensemble"
    return None


def decode_objective(raw: object) -> EngineTunerObjective | None:
    """Return a supported objective literal from persisted data."""

    if raw == "finish_time":
        return "finish_time"
    if raw == "safe_finish_time":
        return "safe_finish_time"
    if raw == "finish_rate":
        return "finish_rate"
    return None


def objective_from_mapping(raw: Mapping[object, object]) -> EngineTunerObjective | None:
    """Decode the active objective, defaulting old payloads to finish time."""

    if "objective" not in raw:
        return "finish_time"
    return decode_objective(raw.get("objective"))


def string_tuple(raw: object) -> tuple[str, ...] | None:
    """Decode a JSON string list as a tuple."""

    if not isinstance(raw, list):
        return None
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            return None
        values.append(item)
    return tuple(values)


def mapping_str(raw: Mapping[object, object], key: str) -> str | None:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        return None
    return value


def mapping_optional_str(raw: Mapping[object, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        return None
    return value


def mapping_int(raw: Mapping[object, object], key: str) -> int | None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def mapping_float(raw: Mapping[object, object], key: str) -> float | None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mapping_optional_float(raw: Mapping[object, object], key: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    return mapping_float(raw, key)


def mapping_optional_int(raw: Mapping[object, object], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    return mapping_int(raw, key)
