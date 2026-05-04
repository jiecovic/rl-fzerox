# src/rl_fzerox/core/training/runs/baseline_materializer/materialization/baselines.py
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fzerox_emulator.base import RaceStartMode
from rl_fzerox.core.config.vehicle_catalog import resolve_engine_setting
from rl_fzerox.core.training.runs.baseline_materializer.cache import (
    atomic_write_json,
    course_vehicle_cache_payload,
    link_or_copy_file,
    run_state_path,
    sha256_bytes,
    sha256_json,
)
from rl_fzerox.core.training.runs.baseline_materializer.materialization.course_vehicle import (
    ensure_course_vehicle_baseline,
    read_metadata,
)
from rl_fzerox.core.training.runs.baseline_materializer.models import (
    BaselineArtifact,
    BaselineMaterializerContext,
    BaselineRequest,
    GenericModeSeedMaterializer,
    RaceStartMaterializer,
    StateSavingEmulator,
)
from rl_fzerox.core.training.runs.baseline_materializer.settings import (
    BASELINE_MATERIALIZER_SETTINGS,
)
from rl_fzerox.core.training.runs.paths import RunPaths


def materialize_baseline_impl(
    request: BaselineRequest,
    *,
    run_paths: RunPaths,
    cache_root: Path,
    context: BaselineMaterializerContext,
    emulator_type: Callable[..., StateSavingEmulator],
    generic_mode_seed_materializer: GenericModeSeedMaterializer,
    menu_seed_race_start_materializer: RaceStartMaterializer,
) -> BaselineArtifact:
    reused_artifact = _reuse_existing_run_baseline(request, run_paths=run_paths)
    if reused_artifact is not None:
        return reused_artifact
    source_course_index = _required_request_course_index(request)
    source_vehicle_id = _request_vehicle_id(request)
    source_engine = _source_engine_setting()
    cache_state_path = ensure_course_vehicle_baseline(
        mode=_validated_request_mode(_required_request_mode(request)),
        label=request.label,
        course_index=source_course_index,
        vehicle_id=source_vehicle_id,
        camera_setting=request.camera_setting,
        cache_root=cache_root,
        context=context,
        emulator_type=emulator_type,
        generic_mode_seed_materializer=generic_mode_seed_materializer,
        menu_seed_race_start_materializer=menu_seed_race_start_materializer,
    )
    payload = course_vehicle_cache_payload(
        mode=_required_request_mode(request),
        course_index=source_course_index,
        vehicle_id=source_vehicle_id,
        camera_setting=request.camera_setting,
        race_intro_target_timer=context.race_intro_target_timer,
        context=context,
    )
    cache_key = sha256_json(payload)
    target_state_path = run_state_path(run_paths, label=request.label, cache_key=cache_key)
    target_metadata_path = target_state_path.with_suffix(".json")
    if not target_state_path.is_file():
        link_or_copy_file(cache_state_path, target_state_path)
    if not target_metadata_path.is_file():
        atomic_write_json(
            target_metadata_path,
            {
                **payload,
                "cache_key": cache_key,
                "cache_kind": "exact_run_baseline",
                "materialized_state_sha256": sha256_bytes(cache_state_path.read_bytes()),
                "source_course_index": source_course_index,
                "source_vehicle": source_vehicle_id,
                "source_engine_setting": source_engine.id,
                "source_engine_setting_raw_value": source_engine.raw_value,
            },
        )
    return BaselineArtifact(
        state_path=target_state_path,
        metadata_path=target_metadata_path,
        cache_key=cache_key,
        source_course_index=source_course_index,
        source_vehicle=source_vehicle_id,
        source_engine_setting=source_engine.id,
        source_engine_setting_raw_value=source_engine.raw_value,
    )


def _reuse_existing_run_baseline(
    request: BaselineRequest,
    *,
    run_paths: RunPaths,
) -> BaselineArtifact | None:
    if run_paths.fresh_run or request.source_state_path is None:
        return None

    source_state_path = request.source_state_path.expanduser().resolve()
    baselines_dir = run_paths.baselines_dir.expanduser().resolve()
    if not source_state_path.is_relative_to(baselines_dir):
        return None
    if not source_state_path.is_file():
        raise FileNotFoundError(
            f"In-place continuation baseline state not found: {source_state_path}"
        )

    metadata_path = source_state_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "In-place continuation baseline metadata not found: "
            f"{metadata_path}"
        )

    metadata = read_metadata(metadata_path)
    cache_kind = metadata.get("cache_kind")
    cache_key = metadata.get("cache_key")
    if not isinstance(cache_key, str) or not cache_key:
        raise ValueError(
            f"In-place continuation baseline metadata is missing cache_key: {metadata_path}"
        )
    schema_version = metadata.get("schema_version")
    if cache_kind != "exact_run_baseline":
        return None
    if schema_version != BASELINE_MATERIALIZER_SETTINGS.schema_version:
        return None

    return BaselineArtifact(
        state_path=source_state_path,
        metadata_path=metadata_path,
        cache_key=cache_key,
        source_course_index=_optional_metadata_int(metadata, "source_course_index"),
        source_vehicle=_optional_metadata_str(metadata, "source_vehicle"),
        source_engine_setting=_optional_metadata_str(metadata, "source_engine_setting"),
        source_engine_setting_raw_value=_optional_metadata_int(
            metadata,
            "source_engine_setting_raw_value",
        ),
    )


def _request_vehicle_id(request: BaselineRequest) -> str:
    vehicle_id = request.vehicle if request.vehicle is not None else request.source_vehicle
    if vehicle_id is None:
        raise ValueError("vehicle is required for race-start generation")
    return vehicle_id


def _required_request_mode(request: BaselineRequest) -> str:
    if request.mode is None:
        raise ValueError("mode is required for race-start generation")
    return request.mode


def _required_request_course_index(request: BaselineRequest) -> int:
    course_index = (
        request.course_index
        if request.course_index is not None
        else request.source_course_index
    )
    if course_index is None:
        raise ValueError("course_index is required for race-start generation")
    return course_index


def _source_engine_setting():
    return resolve_engine_setting(
        BASELINE_MATERIALIZER_SETTINGS.generic_mode_baseline.engine_setting_raw_value,
        context="baseline materializer source engine",
    )


def _validated_request_mode(mode: str) -> RaceStartMode:
    if mode in ("time_attack", "gp_race"):
        return mode
    raise ValueError(f"Unsupported race-start materializer mode {mode!r}")


def _optional_metadata_int(metadata: dict[str, object], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _optional_metadata_str(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value:
        return value
    return None
