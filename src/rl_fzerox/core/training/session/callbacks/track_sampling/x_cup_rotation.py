# src/rl_fzerox/core/training/session/callbacks/track_sampling/x_cup_rotation.py
"""Runtime replacement of solved generated X Cup track-sampling slots."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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
from rl_fzerox.core.training.runs import RunPaths, save_train_run_config
from rl_fzerox.core.training.runs.baseline_materializer import (
    BASELINE_MATERIALIZER_SETTINGS,
    BaselineArtifact,
    BaselineMaterializerContext,
    materialize_baseline,
)
from rl_fzerox.core.training.runs.baseline_materializer.requests import (
    request_from_track_entry,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)


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


class XCupRotationManager:
    """Materialize and commit replacement baselines for solved X Cup slots."""

    def __init__(
        self,
        *,
        config: TrainAppConfig,
        run_paths: RunPaths,
        cache_root: Path | None = None,
        persist_manifest_on_commit: bool = True,
    ) -> None:
        self._config = config
        self._run_paths = run_paths
        self._persist_manifest_on_commit = persist_manifest_on_commit
        self._cache_root = (
            BASELINE_MATERIALIZER_SETTINGS.cache_root
            if cache_root is None
            else cache_root.expanduser().resolve()
        )

    def rotate_once(
        self,
        *,
        env_config: EnvConfig,
        state: TrackSamplingRuntimeState,
    ) -> XCupRotationUpdate | None:
        """Return one replacement update if an X Cup slot is ready."""

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

        old_entries = groups[eligible.course_key]
        replacement_entries = self._materialized_replacement_entries(old_entries)
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
        )

    def commit(self, update: XCupRotationUpdate) -> None:
        """Persist one successful env replacement and prune inactive states."""

        self._config = update.train_config
        if self._persist_manifest_on_commit:
            save_train_run_config(config=update.train_config, run_dir=self._run_paths.run_dir)
        self._prune_inactive_x_cup_baselines(update.env_config.track_sampling)

    def _materialized_replacement_entries(
        self,
        old_entries: Sequence[TrackSamplingEntryConfig],
    ) -> tuple[TrackSamplingEntryConfig, ...]:
        slot = _required_single_slot(old_entries)
        generation = _next_generation(old_entries)
        identity = generated_x_cup_course_identity(
            master_seed=self._config.seed,
            slot=slot,
            generation=generation,
        )
        return tuple(
            self._materialize_entry(_replacement_entry(old_entry, identity_course=identity))
            for old_entry in old_entries
        )

    def _materialize_entry(
        self,
        entry: TrackSamplingEntryConfig,
    ) -> TrackSamplingEntryConfig:
        context = BaselineMaterializerContext(
            core_path=self._config.emulator.core_path,
            rom_path=self._config.emulator.rom_path,
            renderer=self._config.emulator.renderer,
            race_intro_target_timer=self._config.env.race_intro_target_timer,
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
        return entry.model_copy(update=_materialized_entry_update(artifact))

    def _prune_inactive_x_cup_baselines(self, track_sampling: TrackSamplingConfig) -> None:
        active_paths = _active_x_cup_state_paths(track_sampling.entries)
        active_groups = _active_x_cup_baseline_groups(track_sampling.entries)
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
        "id": identity_course.course_id,
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


def _materialized_entry_update(artifact: BaselineArtifact) -> dict[str, object]:
    update: dict[str, object] = {"baseline_state_path": artifact.state_path}
    if artifact.source_course_index is not None:
        update["source_course_index"] = artifact.source_course_index
    if artifact.source_vehicle is not None:
        update["source_vehicle"] = artifact.source_vehicle
    if artifact.source_gp_difficulty is not None:
        update["source_gp_difficulty"] = artifact.source_gp_difficulty
    if artifact.source_engine_setting_raw_value is not None:
        update["source_engine_setting_raw_value"] = artifact.source_engine_setting_raw_value
    if artifact.generated_course_segment_count is not None:
        update["generated_course_segment_count"] = artifact.generated_course_segment_count
    if artifact.generated_course_length is not None:
        update["generated_course_length"] = artifact.generated_course_length
    return update


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


def _next_generation(entries: Sequence[TrackSamplingEntryConfig]) -> int:
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
