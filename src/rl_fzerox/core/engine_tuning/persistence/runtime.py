# src/rl_fzerox/core/engine_tuning/persistence/runtime.py
"""Runtime checkpoint read/write API for adaptive engine tuning."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from rl_fzerox.core.engine_tuning.persistence.candidates import (
    candidate_from_mapping,
    candidate_payload,
)
from rl_fzerox.core.engine_tuning.persistence.fields import (
    mapping_int,
    mapping_optional_str,
    objective_from_mapping,
)
from rl_fzerox.core.engine_tuning.persistence.models import (
    load_engine_tuning_model_state,
    model_state_from_mapping,
    model_state_payload,
    save_engine_tuning_model_state,
)
from rl_fzerox.core.engine_tuning.state import (
    ENGINE_TUNING_STATE_VERSION,
    EngineTuningRuntimeState,
)


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
        save_engine_tuning_model_state(model_path, state.model_state)


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
    model_state = load_engine_tuning_model_state(model_path, metadata=state.model_state)
    return state.with_model_state(model_state)


def engine_tuning_runtime_state_json(state: EngineTuningRuntimeState) -> str:
    """Serialize one engine-tuning checkpoint."""

    data = {
        "version": state.version,
        "update_count": state.update_count,
        "objective": state.objective,
        "reward_fingerprint": state.reward_fingerprint,
        "candidates": [candidate_payload(candidate) for candidate in state.candidates],
        "model_state": model_state_payload(state.model_state),
    }
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def load_engine_tuning_runtime_state_json(data: str) -> EngineTuningRuntimeState | None:
    """Load one engine-tuning checkpoint from JSON text."""

    loaded = json.loads(data)
    if not isinstance(loaded, Mapping):
        return None
    version = mapping_int(loaded, "version") or 1
    if version != ENGINE_TUNING_STATE_VERSION:
        return None
    raw_candidates = loaded.get("candidates")
    if not isinstance(raw_candidates, list):
        return None
    candidates = tuple(candidate_from_mapping(raw_candidate) for raw_candidate in raw_candidates)
    candidates = tuple(candidate for candidate in candidates if candidate is not None)
    objective = objective_from_mapping(loaded)
    if objective is None:
        return None
    model_state = model_state_from_mapping(loaded.get("model_state"))
    return EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=max(0, mapping_int(loaded, "update_count") or 0),
        candidates=candidates,
        objective=objective,
        reward_fingerprint=mapping_optional_str(loaded, "reward_fingerprint"),
        model_state=model_state,
    )
