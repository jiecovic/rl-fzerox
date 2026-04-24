# src/rl_fzerox/core/training/runs/baseline_materializer/cache.py
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from rl_fzerox.core.config.vehicle_catalog import vehicle_by_id
from rl_fzerox.core.training.runs.paths import RunPaths

from .models import BaselineMaterializerContext, BaselineRequest
from .settings import BASELINE_MATERIALIZER_SETTINGS


@contextmanager
def cache_write_lock(cache_state_path: Path) -> Iterator[None]:
    """Serialize generation for one cache key."""

    lock_path = cache_state_path.with_suffix(f"{cache_state_path.suffix}.lock")
    deadline = time.monotonic() + BASELINE_MATERIALIZER_SETTINGS.cache_lock_timeout_seconds
    while True:
        try:
            lock_path.mkdir(parents=True)
            break
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for baseline cache lock {lock_path}"
                ) from exc
            time.sleep(BASELINE_MATERIALIZER_SETTINGS.cache_lock_poll_seconds)
    try:
        yield
    finally:
        shutil.rmtree(lock_path, ignore_errors=True)


def cache_payload(
    request: BaselineRequest,
    *,
    materializer_mode: str,
    context: BaselineMaterializerContext,
) -> dict[str, object]:
    return {
        "schema_version": BASELINE_MATERIALIZER_SETTINGS.schema_version,
        "materializer_mode": materializer_mode,
        "target": target_variant_cache_data(request, context=context),
    }


def target_variant_cache_data(
    request: BaselineRequest,
    *,
    context: BaselineMaterializerContext,
) -> dict[str, object]:
    payload = {
        "camera_setting": request.camera_setting,
        "course_id": request.course_id,
        "course_index": request.course_index,
        "engine_setting": request.engine_setting,
        "engine_setting_raw_value": request.engine_setting_raw_value,
        "mode": request.mode,
        "race_intro_target_timer": context.race_intro_target_timer,
        "renderer": context.renderer,
        "vehicle": request.vehicle,
    }
    if request.vehicle is not None:
        vehicle = vehicle_by_id(request.vehicle)
        payload["vehicle_character_index"] = vehicle.character_index
        payload["vehicle_menu_slot"] = vehicle.machine_select_slot
    return payload


def cache_metadata(
    request: BaselineRequest,
    *,
    cache_payload: dict[str, object],
    cache_key: str,
    materializer_mode: str,
    materialized_state_sha256: str,
) -> dict[str, object]:
    return {
        **cache_payload,
        "cache_key": cache_key,
        "materializer_mode": materializer_mode,
        "materialized_state_sha256": materialized_state_sha256,
        "request": request_metadata(request),
    }


def request_metadata(request: BaselineRequest) -> dict[str, object]:
    request_data = asdict(request)
    if request.source_state_path is not None:
        request_data["source_state_path"] = str(request.source_state_path)
    return request_data


def run_state_path(run_paths: RunPaths, *, label: str, cache_key: str) -> Path:
    safe_label = safe_filename(label or "baseline")
    return run_paths.baselines_dir / f"{safe_label}__{cache_key[:12]}.state"


def safe_filename(value: str) -> str:
    stripped = BASELINE_MATERIALIZER_SETTINGS.safe_name_pattern.sub("_", value).strip("._")
    return stripped or "baseline"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(data: dict[str, object]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def atomic_write_json(target_path: Path, data: dict[str, object]) -> None:
    tmp_path = target_path.with_name(f".{target_path.stem}.tmp{target_path.suffix}")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, target_path)
