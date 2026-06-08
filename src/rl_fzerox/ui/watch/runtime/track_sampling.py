# src/rl_fzerox/ui/watch/runtime/track_sampling.py
from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.projection.x_cup_runtime import (
    restore_generated_x_cup_track_sampling_from_state,
)
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, WatchAppConfig
from rl_fzerox.core.runtime_spec.schema.tracks import TrackSamplingEntryConfig
from rl_fzerox.core.training.runs import continue_run_paths

TrackSamplingSignature = tuple[tuple[object, ...], ...]
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ManagedTrackSamplingRefresh:
    """Refresh mutable manager-owned X Cup slots for a running watch worker."""

    store: ManagerStore
    run_id: str
    interval_seconds: float = 10.0
    _last_check_monotonic: float = field(default=0.0, init=False)

    @classmethod
    def from_config(cls, config: WatchAppConfig) -> ManagedTrackSamplingRefresh | None:
        db_path = config.watch.manager_db_path
        run_id = config.watch.managed_run_id
        if db_path is None or not run_id:
            return None
        return cls(
            store=ManagerStore(Path(db_path).expanduser().resolve()),
            run_id=run_id,
        )

    def refreshed_config(
        self,
        current_config: TrackSamplingConfig,
        *,
        force: bool = False,
    ) -> TrackSamplingConfig | None:
        if not force and not self._refresh_due():
            return None
        self._last_check_monotonic = time.monotonic()
        state = self.store.get_run_track_sampling_state(self.run_id)
        projected_config = restore_generated_x_cup_track_sampling_from_state(
            current_config,
            state=state,
        )
        projected_signature = _generated_x_cup_signature(projected_config)
        if projected_signature == _generated_x_cup_signature(current_config):
            return None
        refreshed_config = self._with_materialized_x_cup_artifacts(projected_config)
        if refreshed_config is None:
            LOGGER.warning(
                "skipping managed watch track-sampling refresh for run_id=%s: "
                "generated X Cup state changed but run-local baseline artifacts "
                "are not ready",
                self.run_id,
            )
            return None
        return refreshed_config

    def _refresh_due(self) -> bool:
        return time.monotonic() - self._last_check_monotonic >= self.interval_seconds

    def _with_materialized_x_cup_artifacts(
        self,
        config: TrackSamplingConfig,
    ) -> TrackSamplingConfig | None:
        run = self.store.get_run(self.run_id)
        if run is None:
            return None
        artifact_index = _x_cup_baseline_artifacts(continue_run_paths(run.run_dir).baselines_dir)
        next_entries: list[TrackSamplingEntryConfig] = []
        for entry in config.entries:
            if entry.generated_course_kind != X_CUP_COURSE.generated_kind:
                next_entries.append(entry)
                continue
            artifact = _artifact_for_entry(entry, artifact_index)
            if artifact is None:
                return None
            next_entries.append(_entry_with_artifact(entry, artifact))
        return config.model_copy(update={"entries": tuple(next_entries)})


def _generated_x_cup_signature(config: TrackSamplingConfig) -> TrackSamplingSignature:
    return tuple(
        (
            entry.generated_course_slot,
            entry.runtime_course_key,
            entry.id,
            entry.course_id,
            entry.generated_course_hash,
            entry.generated_course_seed,
            entry.generated_course_generation,
        )
        for entry in config.entries
        if entry.generated_course_kind == X_CUP_COURSE.generated_kind
    )


def _x_cup_baseline_artifacts(
    baselines_dir: Path,
) -> dict[tuple[object, ...], Mapping[str, object]]:
    artifacts: dict[tuple[object, ...], Mapping[str, object]] = {}
    for metadata_path in baselines_dir.glob("*.json"):
        metadata = _read_json_mapping(metadata_path)
        if metadata.get("materializer_mode") != X_CUP_COURSE.materializer_mode:
            continue
        state_path = metadata_path.with_suffix(".state").expanduser().resolve()
        if not state_path.is_file():
            continue
        key = _artifact_key(metadata)
        if key is None:
            continue
        artifacts[key] = {**metadata, "state_path": state_path}
    return artifacts


def _artifact_for_entry(
    entry: TrackSamplingEntryConfig,
    artifacts: Mapping[tuple[object, ...], Mapping[str, object]],
) -> Mapping[str, object] | None:
    return artifacts.get(
        (
            entry.generated_course_hash,
            entry.generated_course_seed,
            entry.generated_course_slot,
            entry.generated_course_generation,
            entry.gp_difficulty or entry.source_gp_difficulty,
            entry.vehicle or entry.source_vehicle,
        )
    )


def _entry_with_artifact(
    entry: TrackSamplingEntryConfig,
    artifact: Mapping[str, object],
) -> TrackSamplingEntryConfig:
    update: dict[str, object] = {"baseline_state_path": artifact["state_path"]}
    if (value := _optional_int(artifact.get("source_course_index"))) is not None:
        update["source_course_index"] = value
    if (value := _optional_str(artifact.get("source_vehicle"))) is not None:
        update["source_vehicle"] = value
    if (value := _optional_str(artifact.get("source_gp_difficulty"))) is not None:
        update["source_gp_difficulty"] = value
    if (value := _optional_int(artifact.get("source_engine_setting_raw_value"))) is not None:
        update["source_engine_setting_raw_value"] = value
    if (value := _optional_int(artifact.get("generated_course_segment_count"))) is not None:
        update["generated_course_segment_count"] = value
    if (value := _optional_float(artifact.get("generated_course_length"))) is not None:
        update["generated_course_length"] = value
    return entry.model_copy(update=update)


def _artifact_key(metadata: Mapping[str, object]) -> tuple[object, ...] | None:
    course_hash = _optional_str(metadata.get("x_cup_course_hash"))
    seed = _optional_int(metadata.get("x_cup_seed"))
    slot = _optional_int(metadata.get("x_cup_slot"))
    generation = _optional_int(metadata.get("x_cup_generation"))
    difficulty = _optional_str(metadata.get("source_gp_difficulty"))
    vehicle = _optional_str(metadata.get("source_vehicle"))
    if (
        course_hash is None
        or seed is None
        or slot is None
        or generation is None
        or difficulty is None
        or vehicle is None
    ):
        return None
    return (course_hash, seed, slot, generation, difficulty, vehicle)


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw_data if isinstance(raw_data, dict) else {}


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
