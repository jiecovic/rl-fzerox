# src/rl_fzerox/core/training/session/callbacks/track_sampling/x_cup_rotation.py
"""Runtime replacement of solved generated X Cup track-sampling slots."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Literal

from rl_fzerox.core.domain.x_cup import (
    X_CUP_COURSE,
    GeneratedXCupCourseIdentity,
    generated_x_cup_course_identity,
)
from rl_fzerox.core.runtime_spec.schema import (
    EnvConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    XCupRotationConfig,
)
from rl_fzerox.core.runtime_spec.track_sampling_identity import track_sampling_entry_id
from rl_fzerox.core.runtime_spec.x_cup_slots import (
    GeneratedXCupSlot,
    generated_x_cup_slots_from_track_sampling,
)
from rl_fzerox.core.training.runs import RunPaths, save_train_run_config
from rl_fzerox.core.training.runs.baseline_materializer import (
    BASELINE_MATERIALIZER_SETTINGS,
    BaselineMaterializerContext,
    materialize_baseline,
)
from rl_fzerox.core.training.runs.baseline_materializer.projection import (
    baseline_artifact_entry_update,
)
from rl_fzerox.core.training.runs.baseline_materializer.requests import (
    request_from_track_entry,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.artifacts import (
    TrackSamplingMaterializedArtifact,
    materialized_track_sampling_artifacts,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)

_LOG = logging.getLogger(__name__)

XCupRotationFailureKind = Literal["retryable", "blocked"]


@dataclass(frozen=True, slots=True)
class XCupRotationUpdate:
    """One materialized generated-course replacement ready for env workers."""

    env_config: EnvConfig
    train_config: TrainAppConfig
    replaced_course_key: str
    replacement_course_key: str
    replacement_label: str
    generated_course_slot: int | None
    generated_course_generation: int | None
    generated_course_id: str | None
    generated_course_name: str | None
    generated_course_hash: str | None
    generated_course_seed: int | None
    generated_course_segment_count: int | None
    generated_course_length: float | None
    materialized_artifacts: tuple[TrackSamplingMaterializedArtifact, ...]
    generated_x_cup_slots: tuple[GeneratedXCupSlot, ...]


@dataclass(frozen=True, slots=True)
class XCupRotationFailure:
    """Most recent materialization failure for one generated X Cup slot."""

    course_key: str
    kind: XCupRotationFailureKind
    message: str


class XCupRotationManager:
    """Materialize and commit replacement baselines for solved X Cup slots."""

    def __init__(
        self,
        *,
        config: TrainAppConfig,
        run_paths: RunPaths,
        cache_root: Path | None = None,
        persist_manifest_on_commit: bool = True,
        materialization_retry_delay_seconds: float = 60.0,
    ) -> None:
        self._config = config
        self._run_paths = run_paths
        self._persist_manifest_on_commit = persist_manifest_on_commit
        self._materialization_retry_delay_seconds = max(
            0.0,
            float(materialization_retry_delay_seconds),
        )
        self._materialization_retry_after_by_course_key: dict[str, float] = {}
        self._materialization_failures_by_course_key: dict[str, XCupRotationFailure] = {}
        self._cache_root = (
            BASELINE_MATERIALIZER_SETTINGS.cache_root
            if cache_root is None
            else cache_root.expanduser().resolve()
        )
        self._slots_by_index = _slots_by_index(
            generated_x_cup_slots_from_track_sampling(config.env.track_sampling)
        )

    def rotate_once(
        self,
        *,
        env_config: EnvConfig,
        state: TrackSamplingRuntimeState,
    ) -> XCupRotationUpdate | None:
        """Materialize one replacement update without changing durable ownership."""

        # Rotation is two-phase: rotate_once() prepares a replacement so the
        # callback can publish config and save runtime state first. commit() is
        # the only phase that writes the manifest and prunes inactive baselines.

        track_sampling = env_config.track_sampling
        rotation = track_sampling.x_cup_rotation
        if not rotation.enabled:
            return None
        groups = _generated_x_cup_entry_groups(track_sampling.entries)
        if not groups:
            return None
        eligible = _first_eligible_course(
            state.entries,
            groups=groups,
            rotation=rotation,
        )
        if eligible is None:
            return None
        if self._materialization_blocked(eligible.course_key):
            return None
        if self._materialization_retry_deferred(eligible.course_key):
            return None

        old_entries = groups[eligible.course_key]
        try:
            replacement_entries = self._materialized_replacement_entries(old_entries)
        except Exception as exc:
            self._record_materialization_failure(eligible.course_key, exc)
            return None
        self._materialization_failures_by_course_key.pop(eligible.course_key, None)
        self._materialization_retry_after_by_course_key.pop(eligible.course_key, None)
        entries = _replace_generated_x_cup_group_entries(
            track_sampling.entries,
            course_key=eligible.course_key,
            replacement_entries=replacement_entries,
        )
        replacement_course_key = replacement_entries[0].course_id
        if replacement_course_key is None:
            return None
        replacement_label = (
            replacement_entries[0].course_name
            or replacement_entries[0].display_name
            or replacement_course_key
        )
        next_track_sampling = track_sampling.model_copy(update={"entries": entries})
        next_env_config = env_config.model_copy(update={"track_sampling": next_track_sampling})
        next_train_config = self._config.model_copy(update={"env": next_env_config})
        generated_slots = generated_x_cup_slots_from_track_sampling(next_track_sampling)
        return XCupRotationUpdate(
            env_config=next_env_config,
            train_config=next_train_config,
            replaced_course_key=eligible.course_key,
            replacement_course_key=replacement_course_key,
            replacement_label=replacement_label,
            generated_course_slot=replacement_entries[0].generated_course_slot,
            generated_course_generation=replacement_entries[0].generated_course_generation,
            generated_course_id=replacement_entries[0].course_id,
            generated_course_name=(
                replacement_entries[0].course_name or replacement_entries[0].display_name
            ),
            generated_course_hash=replacement_entries[0].generated_course_hash,
            generated_course_seed=replacement_entries[0].generated_course_seed,
            generated_course_segment_count=replacement_entries[0].generated_course_segment_count,
            generated_course_length=replacement_entries[0].generated_course_length,
            materialized_artifacts=materialized_track_sampling_artifacts(next_track_sampling),
            generated_x_cup_slots=generated_slots,
        )

    def materialization_failure(self, course_key: str) -> XCupRotationFailure | None:
        """Return the latest materialization failure for a generated slot."""

        return self._materialization_failures_by_course_key.get(course_key)

    def _materialization_blocked(self, course_key: str) -> bool:
        failure = self._materialization_failures_by_course_key.get(course_key)
        return failure is not None and failure.kind == "blocked"

    def _materialization_retry_deferred(self, course_key: str) -> bool:
        retry_after = self._materialization_retry_after_by_course_key.get(course_key)
        if retry_after is None:
            return False
        if monotonic() < retry_after:
            return True
        self._materialization_retry_after_by_course_key.pop(course_key, None)
        return False

    def _defer_materialization_retry(self, course_key: str) -> None:
        self._materialization_retry_after_by_course_key[course_key] = (
            monotonic() + self._materialization_retry_delay_seconds
        )

    def _record_materialization_failure(self, course_key: str, exc: Exception) -> None:
        failure = _materialization_failure(course_key=course_key, exc=exc)
        self._materialization_failures_by_course_key[course_key] = failure
        if failure.kind == "retryable":
            self._defer_materialization_retry(course_key)
            _LOG.warning(
                "x cup rotation materialization failed for %s; "
                "keeping existing baselines and retrying later: %s",
                course_key,
                failure.message,
                exc_info=True,
            )
            return
        self._materialization_retry_after_by_course_key.pop(course_key, None)
        _LOG.error(
            "x cup rotation materialization is blocked for %s; "
            "keeping existing baselines until the config changes: %s",
            course_key,
            failure.message,
            exc_info=True,
        )

    def commit(self, update: XCupRotationUpdate) -> None:
        """Finalize one published env replacement and prune inactive states."""

        self._config = update.train_config
        self._slots_by_index = _slots_by_index(update.generated_x_cup_slots)
        if self._persist_manifest_on_commit:
            save_train_run_config(config=update.train_config, run_dir=self._run_paths.run_dir)
        self._prune_inactive_x_cup_baselines(
            update.env_config.track_sampling,
            protected_artifacts=update.materialized_artifacts,
        )

    def _materialized_replacement_entries(
        self,
        old_entries: Sequence[TrackSamplingEntryConfig],
    ) -> tuple[TrackSamplingEntryConfig, ...]:
        slot = _required_single_slot(old_entries)
        generation = self._next_generation(slot=slot, old_entries=old_entries)
        identity = generated_x_cup_course_identity(
            master_seed=self._config.seed,
            slot=slot,
            generation=generation,
        )
        return tuple(
            self._materialize_entry(_replacement_entry(old_entry, identity_course=identity))
            for old_entry in old_entries
        )

    def _next_generation(
        self,
        *,
        slot: int,
        old_entries: Sequence[TrackSamplingEntryConfig],
    ) -> int:
        current_slot = self._slots_by_index.get(slot)
        if current_slot is not None:
            return current_slot.generation + 1
        return _next_generation_from_entries(old_entries)

    def _materialize_entry(
        self,
        entry: TrackSamplingEntryConfig,
    ) -> TrackSamplingEntryConfig:
        context = BaselineMaterializerContext(
            core_path=self._config.emulator.core_path,
            rom_path=self._config.emulator.rom_path,
            renderer=self._config.emulator.renderer,
            race_intro_target_timer=self._config.env.race_intro_target_timer,
            run_seed=self._config.seed,
        )
        request = request_from_track_entry(
            entry,
            camera_setting=self._config.env.camera_setting,
        )
        artifact = materialize_baseline(
            request,
            run_paths=self._run_paths,
            cache_root=self._cache_root,
            context=context,
        )
        return entry.model_copy(update=baseline_artifact_entry_update(artifact=artifact))

    def _prune_inactive_x_cup_baselines(
        self,
        track_sampling: TrackSamplingConfig,
        *,
        protected_artifacts: Sequence[TrackSamplingMaterializedArtifact],
    ) -> None:
        active_paths = _active_x_cup_state_paths(track_sampling.entries)
        active_paths |= _active_x_cup_artifact_paths(protected_artifacts)
        active_groups = _active_x_cup_baseline_groups(track_sampling.entries)
        active_groups |= _active_x_cup_artifact_groups(protected_artifacts)
        inactive_groups = _inactive_x_cup_baseline_groups(
            self._run_paths.baselines_dir,
            active_groups=active_groups,
            active_paths=active_paths,
        )
        delete_count = max(
            0,
            len(inactive_groups) - X_CUP_COURSE.retention_policy.inactive_buffer_courses,
        )
        for _, artifacts in inactive_groups[:delete_count]:
            for _, state_path, metadata_path in artifacts:
                state_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)


def _generated_x_cup_entry_groups(
    entries: Sequence[TrackSamplingEntryConfig],
) -> dict[str, tuple[TrackSamplingEntryConfig, ...]]:
    grouped: dict[str, list[TrackSamplingEntryConfig]] = {}
    for entry in entries:
        if (
            entry.generated_course_kind != X_CUP_COURSE.generated_kind
            or entry.course_id is None
            or entry.generated_course_slot is None
        ):
            continue
        grouped.setdefault(_entry_runtime_course_key(entry), []).append(entry)
    return {
        course_key: tuple(group)
        for course_key, group in grouped.items()
        if _single_slot(group) is not None
    }


def _first_eligible_course(
    entries: Sequence[TrackSamplingRuntimeEntry],
    *,
    groups: Mapping[str, Sequence[TrackSamplingEntryConfig]],
    rotation: XCupRotationConfig,
) -> TrackSamplingRuntimeEntry | None:
    candidates = [
        entry
        for entry in entries
        if entry.course_key in groups and _rotation_eligible(entry, rotation=rotation)
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda entry: (entry.course_key, entry.generation_episode_count),
    )[0]


def _rotation_eligible(
    entry: TrackSamplingRuntimeEntry,
    *,
    rotation: XCupRotationConfig,
) -> bool:
    max_episodes = rotation.max_episodes
    if max_episodes is not None and entry.generation_episode_count >= max_episodes:
        return True
    return (
        entry.generation_episode_count >= rotation.min_episodes
        and entry.generation_ema_completion_fraction is not None
        and entry.generation_ema_completion_fraction >= rotation.completion_threshold
    )


def _entry_runtime_course_key(entry: TrackSamplingEntryConfig) -> str:
    return entry.runtime_course_key or entry.course_id or entry.id


def _replace_generated_x_cup_group_entries(
    entries: Sequence[TrackSamplingEntryConfig],
    *,
    course_key: str,
    replacement_entries: Sequence[TrackSamplingEntryConfig],
) -> tuple[TrackSamplingEntryConfig, ...]:
    replacements = iter(replacement_entries)
    replaced_count = 0
    next_entries: list[TrackSamplingEntryConfig] = []
    for entry in entries:
        if _entry_belongs_to_generated_x_cup_group(entry, course_key=course_key):
            next_entries.append(next(replacements))
            replaced_count += 1
        else:
            next_entries.append(entry)
    if replaced_count != len(replacement_entries):
        raise ValueError(
            "generated X Cup replacement count does not match grouped entries: "
            f"{replaced_count} replaced, {len(replacement_entries)} materialized",
        )
    return tuple(next_entries)


def _entry_belongs_to_generated_x_cup_group(
    entry: TrackSamplingEntryConfig,
    *,
    course_key: str,
) -> bool:
    return (
        entry.generated_course_kind == X_CUP_COURSE.generated_kind
        and entry.generated_course_slot is not None
        and _entry_runtime_course_key(entry) == course_key
    )


def _replacement_entry(
    entry: TrackSamplingEntryConfig,
    *,
    identity_course: GeneratedXCupCourseIdentity,
) -> TrackSamplingEntryConfig:
    update = {
        "id": track_sampling_entry_id(
            course_id=identity_course.course_id,
            runtime_course_key=entry.runtime_course_key,
            mode=entry.mode,
            gp_difficulty=entry.gp_difficulty,
            vehicle=entry.vehicle,
        ),
        "course_id": identity_course.course_id,
        "course_name": identity_course.display_name,
        "display_name": identity_course.display_name,
        "baseline_state_path": None,
        "course_index": X_CUP_COURSE.course_index,
        "source_course_index": X_CUP_COURSE.course_index,
        "generated_course_seed": identity_course.seed,
        "generated_course_hash": identity_course.course_hash,
        "generated_course_slot": identity_course.slot,
        "generated_course_generation": identity_course.generation,
        "generated_course_segment_count": None,
        "generated_course_length": None,
        "log_per_course": False,
    }
    return entry.model_copy(update=update)


def _materialization_failure(
    *,
    course_key: str,
    exc: Exception,
) -> XCupRotationFailure:
    kind: XCupRotationFailureKind = "blocked" if isinstance(exc, ValueError) else "retryable"
    message = str(exc) or type(exc).__name__
    return XCupRotationFailure(course_key=course_key, kind=kind, message=message)


def _required_single_slot(entries: Sequence[TrackSamplingEntryConfig]) -> int:
    slot = _single_slot(entries)
    if slot is None:
        raise ValueError("generated X Cup replacement requires one concrete slot")
    return slot


def _single_slot(entries: Sequence[TrackSamplingEntryConfig]) -> int | None:
    slots = {entry.generated_course_slot for entry in entries}
    if len(slots) != 1:
        return None
    slot = next(iter(slots))
    return None if slot is None else int(slot)


def _slots_by_index(slots: Sequence[GeneratedXCupSlot]) -> dict[int, GeneratedXCupSlot]:
    return {slot.slot: slot for slot in slots}


def _next_generation_from_entries(entries: Sequence[TrackSamplingEntryConfig]) -> int:
    generations = [
        int(entry.generated_course_generation)
        for entry in entries
        if entry.generated_course_generation is not None
    ]
    return (max(generations) if generations else 0) + 1


def _active_x_cup_state_paths(
    entries: Sequence[TrackSamplingEntryConfig],
) -> frozenset[Path]:
    return frozenset(
        entry.baseline_state_path.expanduser().resolve()
        for entry in entries
        if entry.generated_course_kind == X_CUP_COURSE.generated_kind
        and entry.baseline_state_path is not None
    )


def _active_x_cup_baseline_groups(
    entries: Sequence[TrackSamplingEntryConfig],
) -> frozenset[tuple[object, ...]]:
    return frozenset(
        key
        for entry in entries
        if entry.generated_course_kind == X_CUP_COURSE.generated_kind
        and (key := _x_cup_entry_group_key(entry)) is not None
    )


def _active_x_cup_artifact_paths(
    artifacts: Sequence[TrackSamplingMaterializedArtifact],
) -> frozenset[Path]:
    return frozenset(artifact.baseline_state_path.expanduser().resolve() for artifact in artifacts)


def _active_x_cup_artifact_groups(
    artifacts: Sequence[TrackSamplingMaterializedArtifact],
) -> frozenset[tuple[object, ...]]:
    return frozenset(
        key for artifact in artifacts if (key := _x_cup_artifact_group_key(artifact)) is not None
    )


def _inactive_x_cup_baseline_groups(
    baselines_dir: Path,
    *,
    active_groups: frozenset[tuple[object, ...]],
    active_paths: frozenset[Path],
) -> list[tuple[float, tuple[tuple[float, Path, Path], ...]]]:
    groups: dict[tuple[object, ...], list[tuple[float, Path, Path]]] = {}
    for metadata_path in baselines_dir.glob("*.json"):
        metadata = _read_json_mapping(metadata_path)
        if metadata.get("materializer_mode") != X_CUP_COURSE.materializer_mode:
            continue
        state_path = metadata_path.with_suffix(".state").expanduser().resolve()
        if not state_path.is_file():
            continue
        group_key = _x_cup_metadata_group_key(metadata)
        if group_key is None:
            group_key = ("state_path", state_path)
        if group_key in active_groups or state_path in active_paths:
            continue
        groups.setdefault(group_key, []).append(
            (state_path.stat().st_mtime, state_path, metadata_path),
        )
    candidates = [
        (min(item[0] for item in artifacts), tuple(sorted(artifacts, key=lambda item: item[0])))
        for artifacts in groups.values()
    ]
    return sorted(candidates, key=lambda candidate: candidate[0])


def _x_cup_artifact_group_key(
    artifact: TrackSamplingMaterializedArtifact,
) -> tuple[object, ...] | None:
    if (
        artifact.generated_course_hash is None
        or artifact.generated_course_seed is None
        or artifact.generated_course_slot is None
        or artifact.generated_course_generation is None
    ):
        return None
    return (
        artifact.generated_course_hash,
        int(artifact.generated_course_seed),
        int(artifact.generated_course_slot),
        int(artifact.generated_course_generation),
    )


def _x_cup_entry_group_key(entry: TrackSamplingEntryConfig) -> tuple[object, ...] | None:
    if (
        entry.generated_course_hash is None
        or entry.generated_course_seed is None
        or entry.generated_course_slot is None
        or entry.generated_course_generation is None
    ):
        return None
    return (
        entry.generated_course_hash,
        int(entry.generated_course_seed),
        int(entry.generated_course_slot),
        int(entry.generated_course_generation),
    )


def _x_cup_metadata_group_key(metadata: Mapping[str, object]) -> tuple[object, ...] | None:
    course_hash = _optional_str(metadata.get("x_cup_course_hash"))
    seed = _optional_int(metadata.get("x_cup_seed"))
    slot = _optional_int(metadata.get("x_cup_slot"))
    generation = _optional_int(metadata.get("x_cup_generation"))
    if course_hash is None or seed is None or slot is None or generation is None:
        return None
    return (course_hash, seed, slot, generation)


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
