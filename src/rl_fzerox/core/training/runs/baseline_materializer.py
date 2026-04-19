# src/rl_fzerox/core/training/runs/baseline_materializer.py
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_fzerox.core.config.paths import project_root_dir
from rl_fzerox.core.config.schema import (
    CurriculumConfig,
    CurriculumStageConfig,
    TrackConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
)
from rl_fzerox.core.training.runs.paths import RunPaths

BASELINE_MATERIALIZER_SCHEMA_VERSION = 1
_CACHE_ROOT = project_root_dir() / "local" / "cache" / "baseline_materializer"
_SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class BaselineRequest:
    """Resolved input needed to materialize one run-local reset state."""

    label: str
    source_state_path: Path
    course_id: str | None = None
    course_name: str | None = None
    course_index: int | None = None
    mode: str | None = None
    vehicle: str | None = None
    vehicle_name: str | None = None
    engine_setting: str | None = None
    camera_setting: str | None = None


@dataclass(frozen=True, slots=True)
class BaselineArtifact:
    """Run-local materialized baseline artifact and metadata paths."""

    state_path: Path
    metadata_path: Path
    cache_key: str


def materialize_run_baselines(
    config: TrainAppConfig,
    *,
    run_paths: RunPaths,
    cache_root: Path | None = None,
) -> TrainAppConfig:
    """Materialize run-local baseline state artifacts from existing source states."""

    resolved_cache_root = (cache_root or _CACHE_ROOT).expanduser().resolve()
    materialized_track_path: Path | None = None
    track_config = config.track
    if track_config.baseline_state_path is not None:
        artifact = materialize_baseline(
            _request_from_track_config(
                track_config,
                camera_setting=config.env.camera_setting,
                fallback_label="track",
            ),
            run_paths=run_paths,
            cache_root=resolved_cache_root,
        )
        materialized_track_path = artifact.state_path
        track_config = track_config.model_copy(
            update={"baseline_state_path": artifact.state_path}
        )

    emulator_baseline_path = config.emulator.baseline_state_path
    if emulator_baseline_path is not None:
        if (
            config.track.baseline_state_path is not None
            and emulator_baseline_path.resolve() == config.track.baseline_state_path.resolve()
            and materialized_track_path is not None
        ):
            emulator_baseline_path = materialized_track_path
        else:
            artifact = materialize_baseline(
                _request_from_track_config(
                    config.track.model_copy(update={"baseline_state_path": emulator_baseline_path}),
                    camera_setting=config.env.camera_setting,
                    fallback_label="baseline",
                ),
                run_paths=run_paths,
                cache_root=resolved_cache_root,
            )
            emulator_baseline_path = artifact.state_path

    env_config = config.env.model_copy(
        update={
            "track_sampling": _materialize_track_sampling(
                config.env.track_sampling,
                run_paths=run_paths,
                cache_root=resolved_cache_root,
                camera_setting=config.env.camera_setting,
            )
        }
    )
    curriculum_config = _materialize_curriculum(
        config.curriculum,
        run_paths=run_paths,
        cache_root=resolved_cache_root,
        camera_setting=config.env.camera_setting,
    )

    return config.model_copy(
        update={
            "track": track_config,
            "emulator": config.emulator.model_copy(
                update={
                    "runtime_dir": run_paths.runtime_root,
                    "baseline_state_path": emulator_baseline_path,
                }
            ),
            "env": env_config,
            "curriculum": curriculum_config,
        }
    )


def materialize_baseline(
    request: BaselineRequest,
    *,
    run_paths: RunPaths,
    cache_root: Path,
) -> BaselineArtifact:
    """Ensure one materialized baseline exists in cache and copy it into the run."""

    source_path = request.source_state_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Baseline source state not found: {source_path}")

    source_bytes = source_path.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    cache_payload = _cache_payload(request, source_sha256=source_sha256)
    cache_key = _sha256_json(cache_payload)
    cache_state_path = cache_root / f"{cache_key}.state"
    cache_metadata_path = cache_root / f"{cache_key}.json"
    cache_metadata = {
        **cache_payload,
        "cache_key": cache_key,
        "materialized_state_sha256": source_sha256,
        "materializer_mode": "source_state_copy",
        "source_state_path": str(source_path),
    }
    if not cache_state_path.is_file():
        cache_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(cache_state_path, source_bytes)
        _atomic_write_json(cache_metadata_path, cache_metadata)
    elif not cache_metadata_path.is_file():
        cache_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(cache_metadata_path, cache_metadata)

    run_state_path = _run_state_path(run_paths, label=request.label, cache_key=cache_key)
    run_metadata_path = run_state_path.with_suffix(".json")
    run_state_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_state_path, run_state_path)
    shutil.copy2(cache_metadata_path, run_metadata_path)
    return BaselineArtifact(
        state_path=run_state_path,
        metadata_path=run_metadata_path,
        cache_key=cache_key,
    )


def _materialize_curriculum(
    config: CurriculumConfig,
    *,
    run_paths: RunPaths,
    cache_root: Path,
    camera_setting: str | None,
) -> CurriculumConfig:
    stages: list[CurriculumStageConfig] = []
    changed = False
    for stage in config.stages:
        if stage.track_sampling is None:
            stages.append(stage)
            continue
        changed = True
        stages.append(
            stage.model_copy(
                update={
                    "track_sampling": _materialize_track_sampling(
                        stage.track_sampling,
                        run_paths=run_paths,
                        cache_root=cache_root,
                        camera_setting=camera_setting,
                    )
                }
            )
        )
    if not changed:
        return config
    return config.model_copy(update={"stages": tuple(stages)})


def _materialize_track_sampling(
    config: TrackSamplingConfig,
    *,
    run_paths: RunPaths,
    cache_root: Path,
    camera_setting: str | None,
) -> TrackSamplingConfig:
    if not config.entries:
        return config
    entries = tuple(
        entry.model_copy(
            update={
                "baseline_state_path": materialize_baseline(
                    _request_from_track_entry(entry, camera_setting=camera_setting),
                    run_paths=run_paths,
                    cache_root=cache_root,
                ).state_path
            }
        )
        for entry in config.entries
    )
    return config.model_copy(update={"entries": entries})


def _request_from_track_config(
    track: TrackConfig,
    *,
    camera_setting: str | None,
    fallback_label: str,
) -> BaselineRequest:
    if track.baseline_state_path is None:
        raise ValueError("track baseline_state_path is required")
    return BaselineRequest(
        label=track.id or track.course_id or fallback_label,
        source_state_path=track.baseline_state_path,
        course_id=track.course_id or track.id,
        course_name=track.course_name or track.display_name,
        course_index=track.course_index,
        mode=track.mode,
        vehicle=track.vehicle,
        vehicle_name=track.vehicle_name,
        engine_setting=track.engine_setting,
        camera_setting=camera_setting,
    )


def _request_from_track_entry(
    entry: TrackSamplingEntryConfig,
    *,
    camera_setting: str | None,
) -> BaselineRequest:
    return BaselineRequest(
        label=entry.id,
        source_state_path=entry.baseline_state_path,
        course_id=entry.course_id,
        course_name=entry.course_name or entry.display_name,
        course_index=entry.course_index,
        mode=entry.mode,
        vehicle=entry.vehicle,
        vehicle_name=entry.vehicle_name,
        engine_setting=entry.engine_setting,
        camera_setting=camera_setting,
    )


def _cache_payload(
    request: BaselineRequest,
    *,
    source_sha256: str,
) -> dict[str, object]:
    request_data = asdict(request)
    request_data.pop("source_state_path", None)
    return {
        "schema_version": BASELINE_MATERIALIZER_SCHEMA_VERSION,
        "source_state_sha256": source_sha256,
        "request": request_data,
    }


def _run_state_path(run_paths: RunPaths, *, label: str, cache_key: str) -> Path:
    safe_label = _safe_filename(label or "baseline")
    return run_paths.baselines_dir / f"{safe_label}__{cache_key[:12]}.state"


def _safe_filename(value: str) -> str:
    stripped = _SAFE_NAME_PATTERN.sub("_", value).strip("._")
    return stripped or "baseline"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(data: dict[str, object]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _atomic_write_bytes(target_path: Path, data: bytes) -> None:
    tmp_path = target_path.with_name(f".{target_path.stem}.tmp{target_path.suffix}")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, target_path)


def _atomic_write_json(target_path: Path, data: dict[str, object]) -> None:
    tmp_path = target_path.with_name(f".{target_path.stem}.tmp{target_path.suffix}")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, target_path)
