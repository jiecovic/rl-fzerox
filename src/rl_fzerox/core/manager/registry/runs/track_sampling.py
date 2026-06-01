# src/rl_fzerox/core/manager/registry/runs/track_sampling.py
from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import TYPE_CHECKING

from rl_fzerox.core.manager.registry.common import utc_now
from rl_fzerox.core.training.session.callbacks.track_sampling.persistence import (
    load_track_sampling_runtime_state_json,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def clear_run_track_sampling_state(store: ManagerStore, run_id: str) -> None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        connection.execute("DELETE FROM run_track_sampling_entries WHERE run_id = ?", (run_id,))
        connection.execute("DELETE FROM run_track_sampling_runtime WHERE run_id = ?", (run_id,))


def get_run_track_sampling_state(
    store: ManagerStore,
    run_id: str,
) -> TrackSamplingRuntimeState | None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        runtime_row = connection.execute(
            """
            SELECT
                sampling_mode,
                action_repeat,
                update_episodes,
                ema_alpha,
                max_weight_scale,
                adaptive_completion_weight,
                adaptive_target_completion,
                adaptive_min_confidence_episodes,
                adaptive_confidence_scale,
                update_count,
                episodes_since_update
            FROM run_track_sampling_runtime
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        entry_rows = connection.execute(
            """
            SELECT
                track_id,
                course_key,
                label,
                base_weight,
                current_weight,
                completed_frames,
                episode_count,
                finished_episode_count,
                success_sample_count,
                ema_episode_frames,
                ema_completion_fraction,
                generation_episode_count,
                generation_finished_episode_count,
                generation_success_sample_count,
                generation_ema_completion_fraction,
                generated_course_slot,
                generated_course_generation,
                generated_entry_id,
                generated_course_id,
                generated_course_name,
                generated_course_hash,
                generated_course_seed,
                generated_baseline_state_path,
                generated_course_segment_count,
                generated_course_length
            FROM run_track_sampling_entries
            WHERE run_id = ?
            ORDER BY course_key
            """,
            (run_id,),
        ).fetchall()
    if runtime_row is None or not entry_rows:
        return None
    return TrackSamplingRuntimeState(
        sampling_mode=str(runtime_row["sampling_mode"]),
        action_repeat=int(runtime_row["action_repeat"]),
        update_episodes=int(runtime_row["update_episodes"]),
        ema_alpha=float(runtime_row["ema_alpha"]),
        max_weight_scale=float(runtime_row["max_weight_scale"]),
        adaptive_completion_weight=float(runtime_row["adaptive_completion_weight"]),
        adaptive_target_completion=float(runtime_row["adaptive_target_completion"]),
        adaptive_min_confidence_episodes=int(runtime_row["adaptive_min_confidence_episodes"]),
        adaptive_confidence_scale=float(runtime_row["adaptive_confidence_scale"]),
        update_count=int(runtime_row["update_count"]),
        episodes_since_update=int(runtime_row["episodes_since_update"]),
        entries=tuple(_track_sampling_entry_from_row(row) for row in entry_rows),
    )


def upsert_run_track_sampling_state(
    store: ManagerStore,
    *,
    run_id: str,
    state: TrackSamplingRuntimeState,
    updated_at: str | None = None,
) -> None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        _write_track_sampling_state(
            connection,
            run_id=run_id,
            state=state,
            updated_at=updated_at or utc_now(),
        )


def migrate_run_track_sampling_state_json(store: ManagerStore, run_id: str | None = None) -> int:
    """Move old JSON track-sampling rows into the typed runtime tables."""

    store._ensure_schema_initialized()
    with store._connect() as connection:
        if not _table_exists(connection, "run_track_sampling_state"):
            return 0
        if run_id is None:
            rows = connection.execute(
                """
                SELECT run_id, state_json, updated_at
                FROM run_track_sampling_state
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT run_id, state_json, updated_at
                FROM run_track_sampling_state
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()
        migrated_count = 0
        for row in rows:
            state = _track_sampling_state_from_state_json(str(row["state_json"]))
            if state is None:
                continue
            _write_track_sampling_state(
                connection,
                run_id=str(row["run_id"]),
                state=state,
                updated_at=str(row["updated_at"]),
            )
            migrated_count += 1
        if run_id is None and migrated_count == len(rows):
            connection.execute("DROP TABLE run_track_sampling_state")
        return migrated_count


def _write_track_sampling_state(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    state: TrackSamplingRuntimeState,
    updated_at: str,
) -> None:
    connection.execute(
        """
        INSERT INTO run_track_sampling_runtime(
            run_id,
            sampling_mode,
            action_repeat,
            update_episodes,
            ema_alpha,
            max_weight_scale,
            adaptive_completion_weight,
            adaptive_target_completion,
            adaptive_min_confidence_episodes,
            adaptive_confidence_scale,
            update_count,
            episodes_since_update,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            sampling_mode = excluded.sampling_mode,
            action_repeat = excluded.action_repeat,
            update_episodes = excluded.update_episodes,
            ema_alpha = excluded.ema_alpha,
            max_weight_scale = excluded.max_weight_scale,
            adaptive_completion_weight = excluded.adaptive_completion_weight,
            adaptive_target_completion = excluded.adaptive_target_completion,
            adaptive_min_confidence_episodes = excluded.adaptive_min_confidence_episodes,
            adaptive_confidence_scale = excluded.adaptive_confidence_scale,
            update_count = excluded.update_count,
            episodes_since_update = excluded.episodes_since_update,
            updated_at = excluded.updated_at
        """,
        (
            run_id,
            state.sampling_mode,
            state.action_repeat,
            state.update_episodes,
            state.ema_alpha,
            state.max_weight_scale,
            state.adaptive_completion_weight,
            state.adaptive_target_completion,
            state.adaptive_min_confidence_episodes,
            state.adaptive_confidence_scale,
            state.update_count,
            state.episodes_since_update,
            updated_at,
        ),
    )
    connection.execute("DELETE FROM run_track_sampling_entries WHERE run_id = ?", (run_id,))
    connection.executemany(
        """
        INSERT INTO run_track_sampling_entries(
            run_id,
            course_key,
            track_id,
            label,
            base_weight,
            current_weight,
            completed_frames,
            episode_count,
            finished_episode_count,
            success_sample_count,
            ema_episode_frames,
            ema_completion_fraction,
            generation_episode_count,
            generation_finished_episode_count,
            generation_success_sample_count,
            generation_ema_completion_fraction,
            generated_course_slot,
            generated_course_generation,
            generated_entry_id,
            generated_course_id,
            generated_course_name,
            generated_course_hash,
            generated_course_seed,
            generated_baseline_state_path,
            generated_course_segment_count,
            generated_course_length
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tuple(_track_sampling_entry_values(run_id, entry) for entry in state.entries),
    )


def _track_sampling_state_from_state_json(data: str) -> TrackSamplingRuntimeState | None:
    loaded = json.loads(data)
    if not isinstance(loaded, Mapping):
        return None
    raw_entries = loaded.get("entries")
    if not isinstance(raw_entries, list):
        return None
    migrated_entries: list[object] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            migrated_entries.append(raw_entry)
            continue
        migrated_entry = dict(raw_entry)
        migrated_entry.pop("generated_replacement_count", None)
        generation = migrated_entry.get("generated_course_generation")
        if generation is not None:
            migrated_entry["generated_course_generation"] = int(generation) + 1
        migrated_entries.append(migrated_entry)
    migrated = dict(loaded)
    migrated["entries"] = migrated_entries
    return load_track_sampling_runtime_state_json(json.dumps(migrated))


def _track_sampling_entry_from_row(row: sqlite3.Row) -> TrackSamplingRuntimeEntry:
    return TrackSamplingRuntimeEntry(
        track_id=str(row["track_id"]),
        course_key=str(row["course_key"]),
        label=str(row["label"]),
        base_weight=float(row["base_weight"]),
        current_weight=float(row["current_weight"]),
        completed_frames=int(row["completed_frames"]),
        episode_count=int(row["episode_count"]),
        finished_episode_count=int(row["finished_episode_count"]),
        success_sample_count=int(row["success_sample_count"]),
        ema_episode_frames=_optional_float(row["ema_episode_frames"]),
        ema_completion_fraction=_optional_float(row["ema_completion_fraction"]),
        generation_episode_count=int(row["generation_episode_count"]),
        generation_finished_episode_count=int(row["generation_finished_episode_count"]),
        generation_success_sample_count=int(row["generation_success_sample_count"]),
        generation_ema_completion_fraction=_optional_float(
            row["generation_ema_completion_fraction"]
        ),
        generated_course_slot=_optional_int(row["generated_course_slot"]),
        generated_course_generation=_optional_int(row["generated_course_generation"]),
        generated_entry_id=_optional_str(row["generated_entry_id"]),
        generated_course_id=_optional_str(row["generated_course_id"]),
        generated_course_name=_optional_str(row["generated_course_name"]),
        generated_course_hash=_optional_str(row["generated_course_hash"]),
        generated_course_seed=_optional_int(row["generated_course_seed"]),
        generated_baseline_state_path=_optional_str(row["generated_baseline_state_path"]),
        generated_course_segment_count=_optional_int(row["generated_course_segment_count"]),
        generated_course_length=_optional_float(row["generated_course_length"]),
    )


def _track_sampling_entry_values(
    run_id: str,
    entry: TrackSamplingRuntimeEntry,
) -> tuple[object, ...]:
    return (
        run_id,
        entry.course_key,
        entry.track_id,
        entry.label,
        entry.base_weight,
        entry.current_weight,
        entry.completed_frames,
        entry.episode_count,
        entry.finished_episode_count,
        entry.success_sample_count,
        entry.ema_episode_frames,
        entry.ema_completion_fraction,
        entry.generation_episode_count,
        entry.generation_finished_episode_count,
        entry.generation_success_sample_count,
        entry.generation_ema_completion_fraction,
        entry.generated_course_slot,
        entry.generated_course_generation,
        entry.generated_entry_id,
        entry.generated_course_id,
        entry.generated_course_name,
        entry.generated_course_hash,
        entry.generated_course_seed,
        entry.generated_baseline_state_path,
        entry.generated_course_segment_count,
        entry.generated_course_length,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("boolean SQLite value cannot be converted to optional float")
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"SQLite value cannot be converted to optional float: {value!r}")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("boolean SQLite value cannot be converted to optional int")
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"SQLite value cannot be converted to optional int: {value!r}")


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None
