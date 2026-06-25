# src/rl_fzerox/core/training/runs/baseline_materializer/materialization/course_vehicle.py
"""Emulator-backed race-start materialization for course/vehicle baselines."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from rl_fzerox.core.domain.courses import X_CUP_COURSE
from rl_fzerox.core.domain.race import RaceDifficultyName
from rl_fzerox.core.envs.engine.info import read_live_telemetry
from rl_fzerox.core.envs.engine.reset.session import sync_reset_presentation
from rl_fzerox.core.runtime_spec.vehicle_catalog import vehicle_by_id
from rl_fzerox.core.training.runs.baseline_materializer.cache import (
    atomic_write_json,
    cache_write_lock,
    course_vehicle_cache_payload,
    course_vehicle_state_path,
    generic_mode_cache_payload,
    generic_mode_state_path,
    sha256_file,
    sha256_json,
    x_cup_cache_payload,
    x_cup_state_path,
)
from rl_fzerox.core.training.runs.baseline_materializer.models import (
    BaselineMaterializerContext,
    GenericModeSeedMaterializer,
    RaceStartMaterializer,
    StateSavingEmulator,
)
from rl_fzerox.core.training.runs.baseline_materializer.settings import (
    BASELINE_MATERIALIZER_SETTINGS,
)
from rl_fzerox.core.training.runs.race_start import RaceStartVariant
from rl_fzerox.core.training.runs.race_start.models import RaceStartMode
from rl_fzerox.core.training.runs.race_start.x_cup import (
    XCupMaterializedCourse,
    materialize_x_cup_race_start_from_boot,
)


def ensure_course_vehicle_baseline(
    *,
    mode: RaceStartMode,
    label: str,
    course_index: int,
    gp_difficulty: RaceDifficultyName | None,
    vehicle_id: str,
    camera_setting: str | None,
    cache_root: Path,
    context: BaselineMaterializerContext,
    emulator_type: Callable[..., StateSavingEmulator],
    generic_mode_seed_materializer: GenericModeSeedMaterializer,
    menu_seed_race_start_materializer: RaceStartMaterializer,
    baseline_variant_index: int | None = None,
    baseline_variant_count: int | None = None,
    baseline_variant_seed: int | None = None,
) -> Path:
    """Cache the exact race-start state before run-local projection.

    The key includes runtime fingerprints, renderer, course, vehicle and
    optional GP grid seed. The caller still owns linking/copying this cached
    state into the run's baseline directory.
    """

    payload = course_vehicle_cache_payload(
        mode=mode,
        course_index=course_index,
        gp_difficulty=gp_difficulty,
        vehicle_id=vehicle_id,
        camera_setting=camera_setting,
        race_intro_target_timer=context.race_intro_target_timer,
        baseline_variant_index=baseline_variant_index,
        baseline_variant_count=baseline_variant_count,
        baseline_variant_seed=baseline_variant_seed,
        context=context,
    )
    cache_key = sha256_json(payload)
    cache_state_path = course_vehicle_state_path(cache_root, label=label, cache_key=cache_key)

    def generate() -> dict[str, object]:
        materialized_sha256 = generate_course_vehicle_state(
            mode=mode,
            course_index=course_index,
            gp_difficulty=gp_difficulty,
            vehicle_id=vehicle_id,
            camera_setting=camera_setting,
            baseline_variant_seed=baseline_variant_seed,
            cache_state_path=cache_state_path,
            cache_root=cache_root,
            context=context,
            emulator_type=emulator_type,
            generic_mode_seed_materializer=generic_mode_seed_materializer,
            menu_seed_race_start_materializer=menu_seed_race_start_materializer,
        )
        return {"materialized_state_sha256": materialized_sha256}

    return _ensure_cached_state(
        cache_state_path=cache_state_path,
        payload=payload,
        cache_key=cache_key,
        cache_kind="course_vehicle_baseline",
        generate_metadata=generate,
    )


def ensure_generic_mode_baseline(
    *,
    mode: RaceStartMode,
    cache_root: Path,
    context: BaselineMaterializerContext,
    emulator_type: Callable[..., StateSavingEmulator],
    generic_mode_seed_materializer: GenericModeSeedMaterializer,
) -> Path:
    """Cache the reusable menu state that course-specific starts build from.

    This state is intentionally mode-only. Course, vehicle and GP grid choices
    happen in the child materialization step.
    """

    payload = generic_mode_cache_payload(mode=mode, context=context)
    cache_key = sha256_json(payload)
    cache_state_path = generic_mode_state_path(cache_root, mode=mode, cache_key=cache_key)

    def generate() -> dict[str, object]:
        materialized_sha256 = generate_generic_mode_state(
            mode=mode,
            cache_state_path=cache_state_path,
            cache_root=cache_root,
            context=context,
            emulator_type=emulator_type,
            generic_mode_seed_materializer=generic_mode_seed_materializer,
        )
        return {"materialized_state_sha256": materialized_sha256}

    return _ensure_cached_state(
        cache_state_path=cache_state_path,
        payload=payload,
        cache_key=cache_key,
        cache_kind="generic_mode_seed",
        generate_metadata=generate,
    )


def ensure_x_cup_baseline(
    *,
    label: str,
    seed: int,
    course_hash: str,
    gp_difficulty: RaceDifficultyName | None,
    vehicle_id: str,
    camera_setting: str | None,
    cache_root: Path,
    context: BaselineMaterializerContext,
    emulator_type: Callable[..., StateSavingEmulator],
) -> tuple[Path, XCupMaterializedCourse]:
    """Cache one generated-course state and return metadata needed for rotation."""

    payload = x_cup_cache_payload(
        seed=seed,
        course_hash=course_hash,
        gp_difficulty=gp_difficulty,
        vehicle_id=vehicle_id,
        camera_setting=camera_setting,
        race_intro_target_timer=context.race_intro_target_timer,
        context=context,
    )
    cache_key = sha256_json(payload)
    cache_state_path = x_cup_state_path(cache_root, label=label, cache_key=cache_key)
    cache_metadata_path = cache_state_path.with_suffix(".json")

    def generate() -> dict[str, object]:
        materialized_sha256, course = generate_x_cup_state(
            seed=seed,
            gp_difficulty=gp_difficulty,
            vehicle_id=vehicle_id,
            camera_setting=camera_setting,
            cache_state_path=cache_state_path,
            cache_root=cache_root,
            context=context,
            emulator_type=emulator_type,
        )
        return {
            "materialized_state_sha256": materialized_sha256,
            "generated_course_segment_count": course.segment_count,
            "generated_course_length": course.course_length,
        }

    _ensure_cached_state(
        cache_state_path=cache_state_path,
        payload=payload,
        cache_key=cache_key,
        cache_kind=X_CUP_COURSE.baseline_cache_kind,
        generate_metadata=generate,
    )
    return cache_state_path, _read_x_cup_metadata(cache_metadata_path)


def cache_entry_is_current(
    *,
    cache_state_path: Path,
    cache_metadata_path: Path,
    expected_kind: str,
    expected_cache_key: str,
) -> bool:
    if not cache_state_path.is_file() or not cache_metadata_path.is_file():
        return False
    metadata = read_metadata(cache_metadata_path)
    return bool(
        metadata.get("cache_kind") == expected_kind
        and metadata.get("cache_key") == expected_cache_key
        and metadata.get("schema_version") == BASELINE_MATERIALIZER_SETTINGS.schema_version
    )


def read_metadata(metadata_path: Path) -> dict[str, object]:
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw_metadata, dict):
        raise ValueError(f"Baseline metadata must be a JSON object: {metadata_path}")
    return raw_metadata


def _ensure_cached_state(
    *,
    cache_state_path: Path,
    payload: dict[str, object],
    cache_key: str,
    cache_kind: str,
    generate_metadata: Callable[[], dict[str, object]],
) -> Path:
    """Generate one cache state once and publish state+metadata together."""

    cache_metadata_path = cache_state_path.with_suffix(".json")
    if cache_entry_is_current(
        cache_state_path=cache_state_path,
        cache_metadata_path=cache_metadata_path,
        expected_kind=cache_kind,
        expected_cache_key=cache_key,
    ):
        return cache_state_path

    with cache_write_lock(cache_state_path):
        if not cache_entry_is_current(
            cache_state_path=cache_state_path,
            cache_metadata_path=cache_metadata_path,
            expected_kind=cache_kind,
            expected_cache_key=cache_key,
        ):
            cache_state_path.unlink(missing_ok=True)
            cache_metadata_path.unlink(missing_ok=True)
            cache_state_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_update = generate_metadata()
        else:
            metadata_update = {
                "materialized_state_sha256": _required_materialized_state_sha256(
                    cache_metadata_path,
                )
            }
        if not cache_metadata_path.is_file():
            atomic_write_json(
                cache_metadata_path,
                {
                    **payload,
                    "cache_key": cache_key,
                    "cache_kind": cache_kind,
                    **metadata_update,
                },
            )
    return cache_state_path


def generate_generic_mode_state(
    *,
    mode: RaceStartMode,
    cache_state_path: Path,
    cache_root: Path,
    context: BaselineMaterializerContext,
    emulator_type: Callable[..., StateSavingEmulator],
    generic_mode_seed_materializer: GenericModeSeedMaterializer,
) -> str:
    def materialize(emulator: StateSavingEmulator) -> None:
        generic_mode_seed_materializer(emulator=emulator, mode=mode)

    return _generate_state_file(
        cache_state_path=cache_state_path,
        runtime_dir=cache_root / "runtime" / "generic" / cache_state_path.stem,
        baseline_state_path=None,
        context=context,
        emulator_type=emulator_type,
        materialize=materialize,
    )


def generate_course_vehicle_state(
    *,
    mode: RaceStartMode,
    course_index: int,
    gp_difficulty: RaceDifficultyName | None,
    vehicle_id: str,
    camera_setting: str | None,
    baseline_variant_seed: int | None,
    cache_state_path: Path,
    cache_root: Path,
    context: BaselineMaterializerContext,
    emulator_type: Callable[..., StateSavingEmulator],
    generic_mode_seed_materializer: GenericModeSeedMaterializer,
    menu_seed_race_start_materializer: RaceStartMaterializer,
) -> str:
    """Generate an exact race-start state from the cheaper generic menu seed."""

    defaults = BASELINE_MATERIALIZER_SETTINGS.generic_mode_baseline
    vehicle = vehicle_by_id(vehicle_id)
    generic_state_path = ensure_generic_mode_baseline(
        mode=mode,
        cache_root=cache_root,
        context=context,
        emulator_type=emulator_type,
        generic_mode_seed_materializer=generic_mode_seed_materializer,
    )

    def materialize(emulator: StateSavingEmulator) -> None:
        menu_seed_race_start_materializer(
            emulator=emulator,
            variant=RaceStartVariant(
                course_index=course_index,
                mode=mode,
                gp_difficulty=gp_difficulty,
                character_index=vehicle.character_index,
                machine_select_slot=vehicle.machine_select_slot,
                rng_seed=baseline_variant_seed,
                engine_setting_raw_value=defaults.engine_setting_raw_value,
                race_intro_target_timer=None,
            ),
        )
        telemetry = read_live_telemetry(emulator)
        sync_reset_presentation(
            emulator,
            camera_setting=camera_setting,
            race_intro_target_timer=context.race_intro_target_timer,
            telemetry=telemetry,
            info={},
        )

    return _generate_state_file(
        cache_state_path=cache_state_path,
        runtime_dir=cache_root / "runtime" / "course_vehicle" / cache_state_path.stem,
        baseline_state_path=generic_state_path,
        context=context,
        emulator_type=emulator_type,
        materialize=materialize,
    )


def generate_x_cup_state(
    *,
    seed: int,
    gp_difficulty: RaceDifficultyName | None,
    vehicle_id: str,
    camera_setting: str | None,
    cache_state_path: Path,
    cache_root: Path,
    context: BaselineMaterializerContext,
    emulator_type: Callable[..., StateSavingEmulator],
) -> tuple[str, XCupMaterializedCourse]:
    """Generate an X Cup race-start state and return its parsed course metadata."""

    defaults = BASELINE_MATERIALIZER_SETTINGS.generic_mode_baseline
    vehicle = vehicle_by_id(vehicle_id)
    course: XCupMaterializedCourse | None = None

    def materialize(emulator: StateSavingEmulator) -> None:
        nonlocal course
        course = materialize_x_cup_race_start_from_boot(
            emulator=emulator,
            variant=RaceStartVariant(
                course_index=X_CUP_COURSE.course_index,
                mode=X_CUP_COURSE.race_mode,
                gp_difficulty=gp_difficulty,
                character_index=vehicle.character_index,
                machine_select_slot=vehicle.machine_select_slot,
                engine_setting_raw_value=defaults.engine_setting_raw_value,
                race_intro_target_timer=context.race_intro_target_timer,
            ),
            rng_seed=seed,
        )
        telemetry = read_live_telemetry(emulator)
        sync_reset_presentation(
            emulator,
            camera_setting=camera_setting,
            race_intro_target_timer=context.race_intro_target_timer,
            telemetry=telemetry,
            info={},
        )

    materialized_sha256 = _generate_state_file(
        cache_state_path=cache_state_path,
        runtime_dir=cache_root / "runtime" / X_CUP_COURSE.cache_dir / cache_state_path.stem,
        baseline_state_path=None,
        context=context,
        emulator_type=emulator_type,
        materialize=materialize,
    )
    if course is None:
        raise RuntimeError("X Cup materialization did not return generated course metadata")
    return materialized_sha256, course


def _generate_state_file(
    *,
    cache_state_path: Path,
    runtime_dir: Path,
    baseline_state_path: Path | None,
    context: BaselineMaterializerContext,
    emulator_type: Callable[..., StateSavingEmulator],
    materialize: Callable[[StateSavingEmulator], None],
) -> str:
    """Run emulator materialization in an isolated runtime dir and atomically publish."""

    runtime_dir.mkdir(parents=True, exist_ok=True)
    tmp_state_path = cache_state_path.with_name(
        f".{cache_state_path.stem}.{os.getpid()}.tmp{cache_state_path.suffix}"
    )
    emulator = emulator_type(
        core_path=context.core_path,
        rom_path=context.rom_path,
        runtime_dir=runtime_dir,
        baseline_state_path=baseline_state_path,
        renderer=context.renderer,
    )
    try:
        materialize(emulator)
        emulator.save_state(tmp_state_path)
    finally:
        emulator.close()
    os.replace(tmp_state_path, cache_state_path)
    return sha256_file(cache_state_path)


def _read_x_cup_metadata(metadata_path: Path) -> XCupMaterializedCourse:
    metadata = read_metadata(metadata_path)
    segment_count = metadata.get("generated_course_segment_count")
    course_length = metadata.get("generated_course_length")
    if (
        isinstance(segment_count, int)
        and not isinstance(segment_count, bool)
        and isinstance(course_length, int | float)
        and not isinstance(course_length, bool)
    ):
        return XCupMaterializedCourse(
            segment_count=segment_count,
            course_length=float(course_length),
        )
    raise ValueError(f"X Cup metadata is incomplete: {metadata_path}")


def _required_materialized_state_sha256(metadata_path: Path) -> str:
    metadata = read_metadata(metadata_path)
    value = metadata.get("materialized_state_sha256")
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"Baseline metadata is missing materialized_state_sha256: {metadata_path}")
