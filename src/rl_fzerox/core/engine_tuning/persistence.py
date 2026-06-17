# src/rl_fzerox/core/engine_tuning/persistence.py
"""JSON serialization helpers for adaptive engine-tuning checkpoints."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import torch

from rl_fzerox.core.domain.engine_setting import (
    centered_engine_slider_buckets,
    engine_percent_to_slider_step,
)
from rl_fzerox.core.engine_tuning.state import (
    ENGINE_TUNING_STATE_VERSION,
    EngineTuningCandidateState,
    EngineTuningEnsembleMemberState,
    EngineTuningModelContextState,
    EngineTuningModelState,
    EngineTuningRuntimeState,
    EngineTuningTensorState,
)
from rl_fzerox.core.engine_tuning.types import (
    ENGINE_TUNER_DEFAULTS,
    EngineTunerBackend,
    EngineTunerObjective,
)

_LEGACY_BANDIT_PERCENT_BUCKETS = tuple(range(0, 101, 10))
_DEFAULT_BANDIT_SLIDER_BUCKETS = centered_engine_slider_buckets(
    minimum=0,
    maximum=128,
    slider_spacing=ENGINE_TUNER_DEFAULTS.bandit_slider_spacing,
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
        "objective": state.objective,
        "reward_fingerprint": state.reward_fingerprint,
        "candidates": [
            {
                "context_key": candidate.context_key,
                "course_key": candidate.course_key,
                "vehicle_id": candidate.vehicle_id,
                "engine_setting_raw_value": candidate.engine_setting_raw_value,
                "score_count": candidate.score_count,
                "episode_count": candidate.episode_count,
                "finish_count": candidate.finish_count,
                "decayed_count": candidate.decayed_count,
                "decayed_score_total": candidate.decayed_score_total,
                "score_total": candidate.score_total,
                "best_score": candidate.best_score,
                "finish_score_total": candidate.finish_score_total,
                "best_finish_score": candidate.best_finish_score,
                "return_score_total": candidate.return_score_total,
                "best_return_score": candidate.best_return_score,
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
    if version not in {5, 6, ENGINE_TUNING_STATE_VERSION}:
        return None
    raw_candidates = loaded.get("candidates")
    if not isinstance(raw_candidates, list):
        return None
    candidates = tuple(_candidate_from_mapping(raw_candidate) for raw_candidate in raw_candidates)
    candidates = tuple(candidate for candidate in candidates if candidate is not None)
    model_state = _model_state_from_mapping(loaded.get("model_state"))
    if version == 5:
        candidates = _migrate_percent_candidates_to_slider_steps(candidates)
        if model_state is not None and model_state.backend == "mlp_ensemble":
            model_state = None
    return EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=max(0, _mapping_int(loaded, "update_count") or 0),
        candidates=candidates,
        objective=_objective(loaded.get("objective")) or "finish_time",
        reward_fingerprint=_mapping_optional_str(loaded, "reward_fingerprint"),
        model_state=model_state,
    )


def _migrate_percent_candidates_to_slider_steps(
    candidates: tuple[EngineTuningCandidateState, ...],
) -> tuple[EngineTuningCandidateState, ...]:
    merged: dict[tuple[str, int], EngineTuningCandidateState] = {}
    for candidate in candidates:
        slider_step = _legacy_percent_candidate_to_slider_step(candidate.engine_setting_raw_value)
        key = (candidate.context_key, slider_step)
        merged[key] = _merge_candidate(
            existing=merged.get(key),
            source=EngineTuningCandidateState(
                context_key=candidate.context_key,
                course_key=candidate.course_key,
                vehicle_id=candidate.vehicle_id,
                engine_setting_raw_value=slider_step,
                score_count=candidate.score_count,
                episode_count=candidate.episode_count,
                finish_count=candidate.finish_count,
                decayed_count=candidate.decayed_count,
                decayed_score_total=candidate.decayed_score_total,
                score_total=candidate.score_total,
                best_score=candidate.best_score,
                finish_score_total=candidate.finish_score_total,
                best_finish_score=candidate.best_finish_score,
                return_score_total=candidate.return_score_total,
                best_return_score=candidate.best_return_score,
                best_time_ms=candidate.best_time_ms,
            ),
        )
    return tuple(
        sorted(
            merged.values(),
            key=lambda candidate: (candidate.context_key, candidate.engine_setting_raw_value),
        )
    )


def _legacy_percent_candidate_to_slider_step(value: int) -> int:
    """Map old bandit percent buckets onto the centered 0..128 slider grid.

    Version 5 bandit checkpoints stored the old 0, 10, ..., 100 buckets as
    percent-like labels. Treat those as ordinal bucket identities instead of
    nearest displayed ENG percentages, so the default 11-bucket tuner keeps the
    same shape after moving to game-representable slider steps.
    """

    if value in _LEGACY_BANDIT_PERCENT_BUCKETS:
        return _DEFAULT_BANDIT_SLIDER_BUCKETS[_LEGACY_BANDIT_PERCENT_BUCKETS.index(value)]
    return engine_percent_to_slider_step(value)


def _merge_candidate(
    *,
    existing: EngineTuningCandidateState | None,
    source: EngineTuningCandidateState,
) -> EngineTuningCandidateState:
    if existing is None:
        return source
    return EngineTuningCandidateState(
        context_key=existing.context_key,
        course_key=existing.course_key,
        vehicle_id=existing.vehicle_id,
        engine_setting_raw_value=existing.engine_setting_raw_value,
        score_count=existing.score_count + source.score_count,
        episode_count=existing.episode_count + source.episode_count,
        finish_count=existing.finish_count + source.finish_count,
        decayed_count=existing.decayed_count + source.decayed_count,
        decayed_score_total=existing.decayed_score_total + source.decayed_score_total,
        score_total=existing.score_total + source.score_total,
        best_score=_max_optional_score(existing.best_score, source.best_score),
        finish_score_total=existing.finish_score_total + source.finish_score_total,
        best_finish_score=_max_optional_score(existing.best_finish_score, source.best_finish_score),
        return_score_total=existing.return_score_total + source.return_score_total,
        best_return_score=_max_optional_score(existing.best_return_score, source.best_return_score),
        best_time_ms=_min_optional_time(existing.best_time_ms, source.best_time_ms),
    )


def _max_optional_score(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _min_optional_time(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


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
    finish_count = max(0, _mapping_int(raw, "finish_count") or 0)
    score_count = _mapping_int(raw, "score_count")
    score_total = _mapping_float(raw, "score_total") or 0.0
    best_score = _mapping_optional_float(raw, "best_score")
    finish_score_total = _mapping_float(raw, "finish_score_total")
    best_finish_score = _mapping_optional_float(raw, "best_finish_score")
    return EngineTuningCandidateState(
        context_key=context_key,
        course_key=course_key,
        vehicle_id=vehicle_id,
        engine_setting_raw_value=engine_setting_raw_value,
        score_count=max(0, score_count if score_count is not None else finish_count),
        episode_count=max(0, _mapping_int(raw, "episode_count") or 0),
        finish_count=finish_count,
        decayed_count=max(0.0, _mapping_float(raw, "decayed_count") or 0.0),
        decayed_score_total=_mapping_float(raw, "decayed_score_total") or 0.0,
        score_total=score_total,
        best_score=best_score,
        finish_score_total=finish_score_total if finish_score_total is not None else score_total,
        best_finish_score=best_finish_score if best_finish_score is not None else best_score,
        return_score_total=_mapping_float(raw, "return_score_total") or 0.0,
        best_return_score=_mapping_optional_float(raw, "best_return_score"),
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


def _objective(raw: object) -> EngineTunerObjective | None:
    if raw == "finish_time":
        return "finish_time"
    if raw == "episode_return":
        return "episode_return"
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


def _mapping_optional_str(raw: Mapping[object, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
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
