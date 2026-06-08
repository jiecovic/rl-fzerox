# src/rl_fzerox/core/training/runs/baseline_materializer/materialization/baselines.py
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rl_fzerox.core.domain.race_difficulty import RaceDifficultyName, is_race_difficulty_name
from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.training.runs.baseline_materializer.cache import (
    atomic_write_json,
    course_vehicle_cache_payload,
    link_or_copy_file,
    run_state_path,
    sha256_json,
)
from rl_fzerox.core.training.runs.baseline_materializer.materialization.course_vehicle import (
    ensure_course_vehicle_baseline,
    ensure_x_cup_baseline,
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
from rl_fzerox.core.training.runs.race_start.models import RaceStartMode


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
    if request.generated_course_kind == X_CUP_COURSE.generated_kind:
        return _materialize_x_cup_baseline(
            request,
            run_paths=run_paths,
            cache_root=cache_root,
            context=context,
            emulator_type=emulator_type,
        )
    source_course_index = _required_request_course_index(request)
    source_vehicle_id = _request_vehicle_id(request)
    source_gp_difficulty = _request_gp_difficulty(request)
    source_engine_raw_value = _source_engine_setting_raw_value()
    cache_state_path = ensure_course_vehicle_baseline(
        mode=_validated_request_mode(_required_request_mode(request)),
        label=request.label,
        course_index=source_course_index,
        gp_difficulty=source_gp_difficulty,
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
        gp_difficulty=source_gp_difficulty,
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
        cache_metadata_path = cache_state_path.with_suffix(".json")
        atomic_write_json(
            target_metadata_path,
            {
                **payload,
                "cache_key": cache_key,
                "cache_kind": "exact_run_baseline",
                "materialized_state_sha256": _required_materialized_state_sha256(
                    cache_metadata_path
                ),
                "source_course_index": source_course_index,
                "source_vehicle": source_vehicle_id,
                "source_gp_difficulty": source_gp_difficulty,
                "source_engine_setting_raw_value": source_engine_raw_value,
            },
        )
    return BaselineArtifact(
        state_path=target_state_path,
        metadata_path=target_metadata_path,
        cache_key=cache_key,
        source_course_index=source_course_index,
        source_vehicle=source_vehicle_id,
        source_gp_difficulty=source_gp_difficulty,
        source_engine_setting_raw_value=source_engine_raw_value,
    )


def _materialize_x_cup_baseline(
    request: BaselineRequest,
    *,
    run_paths: RunPaths,
    cache_root: Path,
    context: BaselineMaterializerContext,
    emulator_type: Callable[..., StateSavingEmulator],
) -> BaselineArtifact:
    if request.mode != X_CUP_COURSE.race_mode:
        raise ValueError(f"X Cup baseline generation requires mode={X_CUP_COURSE.race_mode}")
    seed = _required_request_generated_seed(request)
    course_hash = _required_request_generated_hash(request)
    source_vehicle_id = _request_vehicle_id(request)
    source_gp_difficulty = _request_gp_difficulty(request)
    source_engine_raw_value = _source_engine_setting_raw_value()
    cache_state_path, course = ensure_x_cup_baseline(
        label=request.label,
        seed=seed,
        course_hash=course_hash,
        gp_difficulty=source_gp_difficulty,
        vehicle_id=source_vehicle_id,
        camera_setting=request.camera_setting,
        cache_root=cache_root,
        context=context,
        emulator_type=emulator_type,
    )
    payload = {
        "schema_version": BASELINE_MATERIALIZER_SETTINGS.schema_version,
        "cache_kind": "exact_run_baseline",
        "materializer_mode": X_CUP_COURSE.materializer_mode,
        "source_cache_path": str(cache_state_path),
        "x_cup_seed": seed,
        "x_cup_course_hash": course_hash,
        "x_cup_slot": request.generated_course_slot,
        "x_cup_generation": request.generated_course_generation,
        "source_course_index": X_CUP_COURSE.course_index,
        "source_vehicle": source_vehicle_id,
        "source_gp_difficulty": source_gp_difficulty,
        "source_engine_setting_raw_value": source_engine_raw_value,
        "generated_course_segment_count": course.segment_count,
        "generated_course_length": course.course_length,
    }
    cache_key = sha256_json(payload)
    target_state_path = run_state_path(run_paths, label=request.label, cache_key=cache_key)
    target_metadata_path = target_state_path.with_suffix(".json")
    if not target_state_path.is_file():
        link_or_copy_file(cache_state_path, target_state_path)
    if not target_metadata_path.is_file():
        cache_metadata_path = cache_state_path.with_suffix(".json")
        atomic_write_json(
            target_metadata_path,
            {
                **payload,
                "cache_key": cache_key,
                "materialized_state_sha256": _required_materialized_state_sha256(
                    cache_metadata_path
                ),
            },
        )
    return BaselineArtifact(
        state_path=target_state_path,
        metadata_path=target_metadata_path,
        cache_key=cache_key,
        source_course_index=X_CUP_COURSE.course_index,
        source_vehicle=source_vehicle_id,
        source_gp_difficulty=source_gp_difficulty,
        source_engine_setting_raw_value=source_engine_raw_value,
        generated_course_segment_count=course.segment_count,
        generated_course_length=course.course_length,
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
            f"In-place continuation baseline metadata not found: {metadata_path}"
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
        source_gp_difficulty=_optional_metadata_race_difficulty(metadata, "source_gp_difficulty"),
        source_engine_setting_raw_value=_optional_metadata_int(
            metadata,
            "source_engine_setting_raw_value",
        ),
        generated_course_segment_count=_optional_metadata_int(
            metadata,
            "generated_course_segment_count",
        ),
        generated_course_length=_optional_metadata_float(metadata, "generated_course_length"),
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
        request.course_index if request.course_index is not None else request.source_course_index
    )
    if course_index is None:
        raise ValueError("course_index is required for race-start generation")
    return course_index


def _required_request_generated_seed(request: BaselineRequest) -> int:
    seed = request.generated_course_seed
    if seed is None:
        raise ValueError("generated_course_seed is required for X Cup baseline generation")
    return seed


def _required_request_generated_hash(request: BaselineRequest) -> str:
    course_hash = request.generated_course_hash
    if course_hash is None:
        raise ValueError("generated_course_hash is required for X Cup baseline generation")
    return course_hash


def _request_gp_difficulty(request: BaselineRequest) -> RaceDifficultyName | None:
    return request.gp_difficulty if request.mode == "gp_race" else None


def _source_engine_setting_raw_value() -> int:
    return BASELINE_MATERIALIZER_SETTINGS.generic_mode_baseline.engine_setting_raw_value


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


def _optional_metadata_float(metadata: dict[str, object], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _optional_metadata_race_difficulty(
    metadata: dict[str, object],
    key: str,
) -> RaceDifficultyName | None:
    value = _optional_metadata_str(metadata, key)
    if value is None or not is_race_difficulty_name(value):
        return None
    return value


def _required_materialized_state_sha256(metadata_path: Path) -> str:
    metadata = read_metadata(metadata_path)
    value = metadata.get("materialized_state_sha256")
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"Baseline metadata is missing materialized_state_sha256: {metadata_path}")
