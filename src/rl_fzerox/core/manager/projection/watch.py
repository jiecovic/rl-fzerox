# src/rl_fzerox/core/manager/projection/watch.py
"""Managed-run to watch-app config projection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from rl_fzerox.core.manager.models import ManagedRun
from rl_fzerox.core.manager.projection.launches import build_managed_train_app_config
from rl_fzerox.core.manager.projection.runtime import restore_managed_runtime_track_sampling
from rl_fzerox.core.manager.store import ManagerStore, default_manager_db_path
from rl_fzerox.core.runtime_spec.schema import (
    TrackConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.core.runtime_spec.track_sampling_variants import (
    entry_supports_baseline_variants,
)
from rl_fzerox.core.runtime_spec.watch_overrides import (
    apply_watch_config_delta,
    watch_config_delta_from_dotlist,
)
from rl_fzerox.core.training.runs import (
    apply_train_run_to_watch_config,
    continue_run_paths,
    materialize_train_run_config,
    materialize_watch_session_config,
    save_train_run_config,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    materialized_track_sampling_artifacts,
)


def resolve_watch_app_config(
    *,
    run_id: str,
    policy_artifact: Literal["latest", "best"] | None,
    manager_db_path: Path | None,
    session_name: str | None = None,
    overrides: Sequence[str],
    startup_reporter: Callable[[str, str], None] | None = None,
) -> WatchAppConfig:
    """Resolve watch config from the canonical SQLite-owned run surface."""

    resolved_manager_db_path = (
        manager_db_path.expanduser().resolve()
        if manager_db_path is not None
        else default_manager_db_path().resolve()
    )
    cli_override_delta = watch_config_delta_from_dotlist(overrides) if overrides else {}
    resolved_policy_artifact: Literal["latest", "best"] = policy_artifact or "latest"
    run, train_config, lineage_frame_offset = managed_watch_train_config(
        db_path=resolved_manager_db_path,
        run_id=run_id,
        startup_reporter=startup_reporter,
    )

    config = default_watch_config_from_train_run(
        train_config,
        run_dir=run.run_dir,
        artifact=resolved_policy_artifact,
    )
    if lineage_frame_offset is not None:
        config = config.model_copy(
            update={
                "watch": config.watch.model_copy(
                    update={"lineage_frame_offset": lineage_frame_offset}
                )
            }
        )

    config = apply_train_run_to_watch_config(
        config,
        run_dir=run.run_dir,
        train_config=train_config,
    )
    if cli_override_delta:
        config = apply_watch_config_delta(config, cli_override_delta)
    if policy_artifact is not None:
        config = config.model_copy(
            update={"watch": config.watch.model_copy(update={"policy_artifact": policy_artifact})}
        )
    config = config.model_copy(
        update={
            "watch": config.watch.model_copy(
                update={
                    "manager_db_path": resolved_manager_db_path,
                    "managed_run_id": run.id,
                }
            )
        }
    )

    return materialize_watch_session_config(
        config,
        run_dir=config.watch.policy_run_dir,
        session_name=session_name,
    )


def default_watch_config_from_train_run(
    train_config: TrainAppConfig,
    *,
    run_dir: Path,
    artifact: Literal["latest", "best"],
) -> WatchAppConfig:
    """Build one minimal watch config from a projected runtime train config."""

    return WatchAppConfig(
        seed=train_config.seed,
        emulator=train_config.emulator,
        env=train_config.env,
        reward=train_config.reward,
        policy=train_config.policy,
        train=train_config.train,
        watch=WatchConfig(
            policy_run_dir=run_dir,
            policy_artifact=artifact,
            policy_algorithm=train_config.train.algorithm,
        ),
    )


def managed_watch_train_config(
    *,
    db_path: Path,
    run_id: str,
    startup_reporter: Callable[[str, str], None] | None = None,
) -> tuple[ManagedRun, TrainAppConfig, int | None]:
    """Resolve one manager-owned run into a watch-ready training config mirror."""

    store = ManagerStore(db_path)
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"managed run not found: {run_id}")
    lineage_frame_offset = lineage_frame_offset_for_run(store, run)
    train_config = build_managed_train_app_config(
        run.config,
        run_id=run.id,
        run_dir=run.run_dir,
    )
    train_config = restore_managed_runtime_track_sampling(
        train_config,
        store=store,
        run_id=run.id,
        include_artifacts=True,
    )
    train_config = materialize_missing_watch_baselines(
        train_config,
        store=store,
        run=run,
        startup_reporter=startup_reporter,
    )
    # The manifest is a mirror of the SQLite-projected runtime config. Keep it
    # synchronized for debugging/export, but never read it as a watch fallback.
    save_train_run_config(config=train_config, run_dir=run.run_dir)
    return (run, train_config, lineage_frame_offset)


def materialize_missing_watch_baselines(
    config: TrainAppConfig,
    *,
    store: ManagerStore,
    run: ManagedRun,
    startup_reporter: Callable[[str, str], None] | None = None,
) -> TrainAppConfig:
    """Ensure watch has local reset states without trusting release bundles.

    Public checkpoints intentionally do not carry emulator save states. Watch
    resolves the manager-owned run spec from SQLite, repairs stale run-local
    baseline paths, and materializes the same derived reset artifacts that a
    training run would create. This covers both primary single-target baselines
    and track-sampling entries.
    """

    prepared = _drop_missing_baseline_paths(config)
    if not _watch_baselines_need_materialization(prepared):
        return prepared

    materialized = materialize_train_run_config(
        prepared,
        run_paths=continue_run_paths(run.run_dir),
        startup_reporter=startup_reporter,
    )
    if materialized.env.track_sampling.enabled:
        store.replace_run_track_sampling_artifacts(
            run_id=run.id,
            artifacts=materialized_track_sampling_artifacts(materialized.env.track_sampling),
        )
    return materialized


def _drop_missing_baseline_paths(config: TrainAppConfig) -> TrainAppConfig:
    track = _drop_missing_track_baseline_path(config.track)
    track_sampling = _drop_missing_track_sampling_paths(config.env.track_sampling)
    emulator = config.emulator
    if not _baseline_state_exists(emulator.baseline_state_path):
        emulator = emulator.model_copy(update={"baseline_state_path": None})
    if (
        track is config.track
        and track_sampling is config.env.track_sampling
        and emulator is config.emulator
    ):
        return config
    return config.model_copy(
        update={
            "track": track,
            "emulator": emulator,
            "env": config.env.model_copy(update={"track_sampling": track_sampling}),
        }
    )


def _drop_missing_track_baseline_path(track: TrackConfig) -> TrackConfig:
    if _baseline_state_exists(track.baseline_state_path):
        return track
    if track.baseline_state_path is None:
        return track
    return track.model_copy(update={"baseline_state_path": None})


def _drop_missing_track_sampling_paths(config: TrackSamplingConfig) -> TrackSamplingConfig:
    next_entries: list[TrackSamplingEntryConfig] = []
    changed = False
    for entry in config.entries:
        if _baseline_state_exists(entry.baseline_state_path):
            next_entries.append(entry)
            continue
        if entry.baseline_state_path is None:
            next_entries.append(entry)
            continue
        changed = True
        next_entries.append(entry.model_copy(update={"baseline_state_path": None}))
    if not changed:
        return config
    return config.model_copy(update={"entries": tuple(next_entries)})


def _watch_baselines_need_materialization(config: TrainAppConfig) -> bool:
    if config.env.track_sampling.enabled:
        return _track_sampling_needs_materialization(config.env.track_sampling)
    return _primary_baseline_needs_materialization(config)


def _track_sampling_needs_materialization(config: TrackSamplingConfig) -> bool:
    return any(
        entry.baseline_state_path is None
        or not _baseline_state_exists(entry.baseline_state_path)
        or _entry_needs_baseline_variant_expansion(config, entry)
        for entry in config.entries
    )


def _entry_needs_baseline_variant_expansion(
    config: TrackSamplingConfig,
    entry: TrackSamplingEntryConfig,
) -> bool:
    return config.baseline_variant_count > 1 and entry_supports_baseline_variants(entry)


def _primary_baseline_needs_materialization(config: TrainAppConfig) -> bool:
    if _baseline_state_exists(config.emulator.baseline_state_path):
        return False
    track = config.track
    return (
        track.course_index is not None
        and track.mode is not None
        and track.vehicle is not None
        and track.engine_setting_raw_value is not None
    )


def _baseline_state_exists(path: Path | None) -> bool:
    return path is not None and path.expanduser().is_file()


def lineage_frame_offset_for_run(store: ManagerStore, run: ManagedRun) -> int | None:
    """Return emulator frames completed before this run's local checkpoint timeline."""

    total_frames = 0
    current_run = run
    while current_run.parent_run_id is not None:
        parent_run = store.get_run(current_run.parent_run_id)
        if parent_run is None:
            return None
        source_steps = current_run.source_num_timesteps
        if source_steps is None:
            source_steps = current_run.lineage_step_offset - parent_run.lineage_step_offset
        if source_steps < 0:
            return None
        total_frames += source_steps * max(1, int(parent_run.config.action.action_repeat))
        current_run = parent_run
    return total_frames
