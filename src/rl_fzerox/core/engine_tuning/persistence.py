# src/rl_fzerox/core/engine_tuning/persistence.py
"""JSON serialization helpers for adaptive engine-tuning checkpoints."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from rl_fzerox.core.engine_tuning.state import (
    EngineTuningArmState,
    EngineTuningRuntimeState,
)


def save_engine_tuning_runtime_state(path: Path, state: EngineTuningRuntimeState) -> None:
    """Persist one engine-tuning checkpoint atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        tmp_path.write_text(engine_tuning_runtime_state_json(state), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_engine_tuning_runtime_state(path: Path) -> EngineTuningRuntimeState | None:
    """Load one engine-tuning checkpoint, if present."""

    if not path.is_file():
        return None
    return load_engine_tuning_runtime_state_json(path.read_text(encoding="utf-8"))


def engine_tuning_runtime_state_json(state: EngineTuningRuntimeState) -> str:
    """Serialize one engine-tuning checkpoint."""

    data = {
        "version": state.version,
        "update_count": state.update_count,
        "arms": [
            {
                "context_key": arm.context_key,
                "course_key": arm.course_key,
                "vehicle_id": arm.vehicle_id,
                "engine_setting_raw_value": arm.engine_setting_raw_value,
                "attempts": arm.attempts,
                "finished_attempts": arm.finished_attempts,
                "decayed_count": arm.decayed_count,
                "decayed_score_total": arm.decayed_score_total,
                "completion_total": arm.completion_total,
                "score_total": arm.score_total,
                "best_score": arm.best_score,
            }
            for arm in state.arms
        ],
    }
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def load_engine_tuning_runtime_state_json(data: str) -> EngineTuningRuntimeState | None:
    """Load one engine-tuning checkpoint from JSON text."""

    loaded = json.loads(data)
    if not isinstance(loaded, Mapping):
        return None
    raw_arms = loaded.get("arms")
    if not isinstance(raw_arms, list):
        return None
    arms = tuple(_arm_from_mapping(raw_arm) for raw_arm in raw_arms)
    arms = tuple(arm for arm in arms if arm is not None)
    return EngineTuningRuntimeState(
        version=max(1, _mapping_int(loaded, "version") or 1),
        update_count=max(0, _mapping_int(loaded, "update_count") or 0),
        arms=arms,
    )


def _arm_from_mapping(raw: object) -> EngineTuningArmState | None:
    if not isinstance(raw, Mapping):
        return None
    context_key = _mapping_str(raw, "context_key")
    course_key = _mapping_str(raw, "course_key")
    vehicle_id = _mapping_str(raw, "vehicle_id")
    engine_setting_raw_value = _mapping_int(raw, "engine_setting_raw_value")
    if (
        context_key is None
        or course_key is None
        or vehicle_id is None
        or engine_setting_raw_value is None
    ):
        return None
    return EngineTuningArmState(
        context_key=context_key,
        course_key=course_key,
        vehicle_id=vehicle_id,
        engine_setting_raw_value=engine_setting_raw_value,
        attempts=max(0, _mapping_int(raw, "attempts") or 0),
        finished_attempts=max(0, _mapping_int(raw, "finished_attempts") or 0),
        decayed_count=max(0.0, _mapping_float(raw, "decayed_count") or 0.0),
        decayed_score_total=_mapping_float(raw, "decayed_score_total") or 0.0,
        completion_total=max(0.0, _mapping_float(raw, "completion_total") or 0.0),
        score_total=_mapping_float(raw, "score_total") or 0.0,
        best_score=_mapping_optional_float(raw, "best_score"),
    )


def _mapping_str(raw: Mapping[object, object], key: str) -> str | None:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        return None
    return value


def _mapping_int(raw: Mapping[object, object], key: str) -> int | None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _mapping_float(raw: Mapping[object, object], key: str) -> float | None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mapping_optional_float(raw: Mapping[object, object], key: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    return _mapping_float(raw, key)
