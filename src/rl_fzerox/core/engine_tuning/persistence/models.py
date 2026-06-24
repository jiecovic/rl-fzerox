# src/rl_fzerox/core/engine_tuning/persistence/models.py
"""Model-state payload and sidecar handling for engine-tuning checkpoints."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import torch

from rl_fzerox.core.engine_tuning.persistence.fields import (
    decode_backend,
    mapping_int,
    mapping_str,
    string_tuple,
)
from rl_fzerox.core.engine_tuning.state import (
    ENGINE_TUNING_STATE_VERSION,
    EngineTuningEnsembleMemberState,
    EngineTuningModelContextState,
    EngineTuningModelState,
    EngineTuningTensorState,
)


def model_state_payload(state: EngineTuningModelState | None) -> dict[str, object] | None:
    """Return JSON metadata for a learned model sidecar."""

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


def model_state_from_mapping(raw: object) -> EngineTuningModelState | None:
    """Decode learned model metadata from the JSON checkpoint."""

    if not isinstance(raw, Mapping):
        return None
    backend = decode_backend(raw.get("backend"))
    if backend is None:
        return None
    course_keys = string_tuple(raw.get("course_keys"))
    vehicle_ids = string_tuple(raw.get("vehicle_ids"))
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


def save_engine_tuning_model_state(
    path: Path,
    state: EngineTuningModelState | None,
) -> None:
    """Persist learned model tensors, or remove the sidecar when none are needed."""

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


def load_engine_tuning_model_state(
    path: Path,
    *,
    metadata: EngineTuningModelState,
) -> EngineTuningModelState | None:
    """Load learned model tensors when they match the JSON metadata."""

    if not path.is_file():
        return None
    loaded = _torch_load(path)
    if not isinstance(loaded, Mapping):
        return None
    if mapping_int(loaded, "version") != ENGINE_TUNING_STATE_VERSION:
        return None
    if loaded.get("backend") != metadata.backend:
        return None
    course_keys = string_tuple(loaded.get("course_keys"))
    vehicle_ids = string_tuple(loaded.get("vehicle_ids"))
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


def _model_contexts_from_raw(raw: object) -> tuple[EngineTuningModelContextState, ...] | None:
    if not isinstance(raw, list):
        return None
    contexts: list[EngineTuningModelContextState] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        context_key = mapping_str(item, "context_key")
        course_key = mapping_str(item, "course_key")
        vehicle_id = mapping_str(item, "vehicle_id")
        if context_key is None or course_key is None or vehicle_id is None:
            return None
        contexts.append(
            EngineTuningModelContextState(
                context_key=context_key,
                course_key=course_key,
                vehicle_id=vehicle_id,
                finish_count=max(0, mapping_int(item, "finish_count") or 0),
            )
        )
    return tuple(contexts)


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
