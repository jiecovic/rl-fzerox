# src/rl_fzerox/core/training/runs/baseline_materializer/cache.py
"""Filesystem cache primitives for expensive baseline save-state generation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from rl_fzerox.core.domain.courses import X_CUP_COURSE
from rl_fzerox.core.runtime_spec.vehicle_catalog import vehicle_by_id
from rl_fzerox.core.training.runs.paths import RunPaths

from .models import BaselineMaterializerContext
from .settings import BASELINE_MATERIALIZER_SETTINGS


@contextmanager
def cache_write_lock(cache_state_path: Path) -> Generator[None]:
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


def course_vehicle_cache_payload(
    *,
    mode: str,
    course_index: int,
    gp_difficulty: str | None,
    vehicle_id: str,
    camera_setting: str | None,
    race_intro_target_timer: int | None,
    context: BaselineMaterializerContext,
    baseline_variant_index: int | None = None,
    baseline_variant_count: int | None = None,
    baseline_variant_seed: int | None = None,
) -> dict[str, object]:
    """Return the content identity for a reusable course/vehicle baseline."""

    defaults = BASELINE_MATERIALIZER_SETTINGS.generic_mode_baseline
    vehicle = vehicle_by_id(vehicle_id)
    payload: dict[str, object] = {
        "schema_version": BASELINE_MATERIALIZER_SETTINGS.schema_version,
        "materializer_mode": f"course_vehicle_seed_{mode}",
        "mode": mode,
        "gp_difficulty": gp_difficulty,
        "renderer": context.renderer,
        **runtime_fingerprint_payload(context),
        "course_index": course_index,
        "vehicle": vehicle_id,
        "vehicle_character_index": vehicle.character_index,
        "vehicle_menu_slot": vehicle.machine_select_slot,
        "engine_setting_raw_value": defaults.engine_setting_raw_value,
        "camera_setting": camera_setting,
        "race_intro_target_timer": race_intro_target_timer,
    }
    if baseline_variant_index is not None:
        payload.update(
            {
                "baseline_variant_index": baseline_variant_index,
                "baseline_variant_count": baseline_variant_count,
                "baseline_variant_seed": baseline_variant_seed,
            }
        )
    return payload


def x_cup_cache_payload(
    *,
    seed: int,
    course_hash: str,
    gp_difficulty: str | None,
    vehicle_id: str,
    camera_setting: str | None,
    race_intro_target_timer: int | None,
    context: BaselineMaterializerContext,
) -> dict[str, object]:
    """Return the content identity for a generated X Cup baseline."""

    defaults = BASELINE_MATERIALIZER_SETTINGS.generic_mode_baseline
    vehicle = vehicle_by_id(vehicle_id)
    return {
        "schema_version": BASELINE_MATERIALIZER_SETTINGS.schema_version,
        "materializer_mode": X_CUP_COURSE.materializer_mode,
        "mode": X_CUP_COURSE.race_mode,
        "gp_difficulty": gp_difficulty,
        "renderer": context.renderer,
        **runtime_fingerprint_payload(context),
        "course_index": X_CUP_COURSE.course_index,
        "x_cup_seed": seed,
        "x_cup_course_hash": course_hash,
        "vehicle": vehicle_id,
        "vehicle_character_index": vehicle.character_index,
        "vehicle_menu_slot": vehicle.machine_select_slot,
        "engine_setting_raw_value": defaults.engine_setting_raw_value,
        "camera_setting": camera_setting,
        "race_intro_target_timer": race_intro_target_timer,
    }


def generic_mode_cache_payload(
    *,
    mode: str,
    context: BaselineMaterializerContext,
) -> dict[str, object]:
    """Return the content identity for a menu-state seed baseline."""

    return {
        "schema_version": BASELINE_MATERIALIZER_SETTINGS.schema_version,
        "materializer_mode": f"generic_mode_seed_{mode}",
        "mode": mode,
        "renderer": context.renderer,
        **runtime_fingerprint_payload(context),
    }


def runtime_fingerprint_payload(context: BaselineMaterializerContext) -> dict[str, str]:
    """Fingerprint runtime inputs that change save-state compatibility.

    Paths are intentionally excluded. Moving the same ROM/core should reuse the
    cache, while changing bytes at the same path must invalidate it.
    """

    return {
        "core_sha256": context.core_sha256,
        "rom_sha256": context.rom_sha256,
    }


def run_state_path(run_paths: RunPaths, *, label: str, cache_key: str) -> Path:
    safe_label = safe_filename(label or "baseline")
    return run_paths.baselines_dir / f"{safe_label}__{cache_key[:12]}.state"


def generic_mode_state_path(cache_root: Path, *, mode: str, cache_key: str) -> Path:
    safe_mode = safe_filename(mode or "generic")
    return cache_root / "generic" / f"{safe_mode}__{cache_key[:12]}.state"


def course_vehicle_state_path(cache_root: Path, *, label: str, cache_key: str) -> Path:
    safe_label = safe_filename(label or "course_vehicle")
    return cache_root / "course_vehicle" / f"{safe_label}__{cache_key[:12]}.state"


def x_cup_state_path(cache_root: Path, *, label: str, cache_key: str) -> Path:
    safe_label = safe_filename(label or X_CUP_COURSE.cache_dir)
    return cache_root / X_CUP_COURSE.cache_dir / f"{safe_label}__{cache_key[:12]}.state"


def link_or_copy_file(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.unlink(missing_ok=True)
    try:
        os.link(source_path, destination_path)
        return
    except OSError:
        pass

    tmp_path = destination_path.with_name(
        f".{destination_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{destination_path.suffix}"
    )
    try:
        shutil.copy2(source_path, tmp_path)
        os.replace(tmp_path, destination_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def safe_filename(value: str) -> str:
    stripped = BASELINE_MATERIALIZER_SETTINGS.safe_name_pattern.sub("_", value).strip("._")
    return stripped or "baseline"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(data: dict[str, object]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def atomic_write_json(target_path: Path, data: dict[str, object]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(
        f".{target_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{target_path.suffix}"
    )
    try:
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, target_path)
    finally:
        tmp_path.unlink(missing_ok=True)
