# src/rl_fzerox/core/engine_tuning/persistence.py
"""JSON serialization helpers for adaptive engine-tuning checkpoints."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import torch

from rl_fzerox.core.engine_tuning.state import (
    ENGINE_TUNING_STATE_VERSION,
    EngineTuningCandidateState,
    EngineTuningEnsembleMemberState,
    EngineTuningModelContextState,
    EngineTuningModelState,
    EngineTuningRuntimeState,
    EngineTuningTensorState,
)
from rl_fzerox.core.engine_tuning.types import EngineTunerBackend


def save_engine_tuning_runtime_state(
    path: Path,
    state: EngineTuningRuntimeState,
    *,
    model_path: Path | None = None,
) -> None:
    """Persist one engine-tuning checkpoint atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        tmp_path.write_text(engine_tuning_runtime_state_json(state), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    if model_path is not None:
        _save_engine_tuning_model_state(model_path, state.model_state)


def load_engine_tuning_runtime_state(
    path: Path,
    *,
    model_path: Path | None = None,
) -> EngineTuningRuntimeState | None:
    """Load one engine-tuning checkpoint, if present."""

    if not path.is_file():
        return None
    state = load_engine_tuning_runtime_state_json(path.read_text(encoding="utf-8"))
    if state is None or state.model_state is None:
        return state
    if state.model_state.backend != "mlp_ensemble":
        return state
    if model_path is None:
        return state.with_model_state(None)
    model_state = _load_engine_tuning_model_state(model_path, metadata=state.model_state)
    return state.with_model_state(model_state)


def engine_tuning_runtime_state_json(state: EngineTuningRuntimeState) -> str:
    """Serialize one engine-tuning checkpoint."""

    data = {
        "version": state.version,
        "update_count": state.update_count,
        "candidates": [
            {
                "context_key": candidate.context_key,
                "course_key": candidate.course_key,
                "vehicle_id": candidate.vehicle_id,
                "engine_setting_raw_value": candidate.engine_setting_raw_value,
                "finish_count": candidate.finish_count,
                "decayed_count": candidate.decayed_count,
                "decayed_score_total": candidate.decayed_score_total,
                "score_total": candidate.score_total,
                "best_score": candidate.best_score,
                "best_time_ms": candidate.best_time_ms,
            }
            for candidate in state.candidates
        ],
        "model_state": _model_state_payload(state.model_state),
    }
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def load_engine_tuning_runtime_state_json(data: str) -> EngineTuningRuntimeState | None:
    """Load one engine-tuning checkpoint from JSON text."""

    loaded = json.loads(data)
    if not isinstance(loaded, Mapping):
        return None
    version = _mapping_int(loaded, "version") or 1
    if version != ENGINE_TUNING_STATE_VERSION:
        return None
    raw_candidates = loaded.get("candidates")
    if not isinstance(raw_candidates, list):
        return None
    candidates = tuple(_candidate_from_mapping(raw_candidate) for raw_candidate in raw_candidates)
    candidates = tuple(candidate for candidate in candidates if candidate is not None)
    return EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=max(0, _mapping_int(loaded, "update_count") or 0),
        candidates=candidates,
        model_state=_model_state_from_mapping(loaded.get("model_state")),
    )


def _model_state_payload(state: EngineTuningModelState | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "backend": state.backend,
        "course_keys": list(state.course_keys),
        "vehicle_ids": list(state.vehicle_ids),
        "contexts": [
            {
                "context_key": context.context_key,
                "course_key": context.course_key,
                "vehicle_id": context.vehicle_id,
                "finish_count": context.finish_count,
            }
            for context in state.contexts
        ],
    }


def _candidate_from_mapping(raw: object) -> EngineTuningCandidateState | None:
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
    return EngineTuningCandidateState(
        context_key=context_key,
        course_key=course_key,
        vehicle_id=vehicle_id,
        engine_setting_raw_value=engine_setting_raw_value,
        finish_count=max(0, _mapping_int(raw, "finish_count") or 0),
        decayed_count=max(0.0, _mapping_float(raw, "decayed_count") or 0.0),
        decayed_score_total=_mapping_float(raw, "decayed_score_total") or 0.0,
        score_total=_mapping_float(raw, "score_total") or 0.0,
        best_score=_mapping_optional_float(raw, "best_score"),
        best_time_ms=_mapping_optional_int(raw, "best_time_ms"),
    )


def _model_state_from_mapping(raw: object) -> EngineTuningModelState | None:
    if not isinstance(raw, Mapping):
        return None
    backend = _backend(raw.get("backend"))
    if backend is None:
        return None
    course_keys = _string_tuple(raw.get("course_keys"))
    vehicle_ids = _string_tuple(raw.get("vehicle_ids"))
    contexts = _model_contexts_from_raw(raw.get("contexts"))
    if course_keys is None or vehicle_ids is None or contexts is None:
        return None
    return EngineTuningModelState(
        backend=backend,
        course_keys=course_keys,
        vehicle_ids=vehicle_ids,
        members=(),
        contexts=contexts,
    )


def _model_contexts_from_raw(raw: object) -> tuple[EngineTuningModelContextState, ...] | None:
    if not isinstance(raw, list):
        return None
    contexts: list[EngineTuningModelContextState] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        context_key = _mapping_str(item, "context_key")
        course_key = _mapping_str(item, "course_key")
        vehicle_id = _mapping_str(item, "vehicle_id")
        if context_key is None or course_key is None or vehicle_id is None:
            return None
        contexts.append(
            EngineTuningModelContextState(
                context_key=context_key,
                course_key=course_key,
                vehicle_id=vehicle_id,
                finish_count=max(0, _mapping_int(item, "finish_count") or 0),
            )
        )
    return tuple(contexts)


def _save_engine_tuning_model_state(
    path: Path,
    state: EngineTuningModelState | None,
) -> None:
    if state is None or state.backend != "mlp_ensemble" or not state.members:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": ENGINE_TUNING_STATE_VERSION,
        "backend": state.backend,
        "course_keys": list(state.course_keys),
        "vehicle_ids": list(state.vehicle_ids),
        "members": [
            {tensor.name: tensor.value.detach().cpu() for tensor in member.tensors}
            for member in state.members
        ],
    }
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_engine_tuning_model_state(
    path: Path,
    *,
    metadata: EngineTuningModelState,
) -> EngineTuningModelState | None:
    if not path.is_file():
        return None
    loaded = _torch_load(path)
    if not isinstance(loaded, Mapping):
        return None
    if _mapping_int(loaded, "version") != ENGINE_TUNING_STATE_VERSION:
        return None
    if loaded.get("backend") != metadata.backend:
        return None
    course_keys = _string_tuple(loaded.get("course_keys"))
    vehicle_ids = _string_tuple(loaded.get("vehicle_ids"))
    if course_keys != metadata.course_keys or vehicle_ids != metadata.vehicle_ids:
        return None
    members = _members_from_model_payload(loaded.get("members"))
    if members is None:
        return None
    return EngineTuningModelState(
        backend=metadata.backend,
        course_keys=metadata.course_keys,
        vehicle_ids=metadata.vehicle_ids,
        members=members,
        contexts=metadata.contexts,
    )


def _torch_load(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _members_from_model_payload(raw: object) -> tuple[EngineTuningEnsembleMemberState, ...] | None:
    if not isinstance(raw, list):
        return None
    members: list[EngineTuningEnsembleMemberState] = []
    for raw_member in raw:
        if not isinstance(raw_member, Mapping):
            return None
        tensors: list[EngineTuningTensorState] = []
        for name, value in raw_member.items():
            if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                return None
            tensors.append(EngineTuningTensorState(name=name, value=value.detach().cpu()))
        members.append(
            EngineTuningEnsembleMemberState(
                tensors=tuple(sorted(tensors, key=lambda tensor: tensor.name))
            )
        )
    return tuple(members)


def _backend(raw: object) -> EngineTunerBackend | None:
    if raw == "bandit":
        return "bandit"
    if raw == "gaussian_process":
        return "gaussian_process"
    if raw == "mlp_ensemble":
        return "mlp_ensemble"
    return None


def _string_tuple(raw: object) -> tuple[str, ...] | None:
    if not isinstance(raw, list):
        return None
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            return None
        values.append(item)
    return tuple(values)


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


def _mapping_optional_int(raw: Mapping[object, object], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    return _mapping_int(raw, key)
