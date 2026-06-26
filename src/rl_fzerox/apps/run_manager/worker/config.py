# src/rl_fzerox/apps/run_manager/worker/config.py
from __future__ import annotations

from rl_fzerox.apps.run_manager.worker.clock import now_iso
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.artifacts.fork_source import load_fork_source_metadata
from rl_fzerox.core.manager.models import ManagedRun
from rl_fzerox.core.manager.projection.runtime import restore_managed_runtime_track_sampling
from rl_fzerox.core.manager.training import (
    apply_managed_resume_train_config,
    build_managed_fork_train_app_config,
    build_managed_fork_train_app_config_from_metadata,
    build_managed_train_app_config,
)
from rl_fzerox.core.training.runs import (
    RUN_LAYOUT,
    RunPaths,
    continue_run_paths,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingRuntimePersistence,
)


def _track_sampling_runtime_persistence(
    *,
    store: ManagerStore,
    run_id: str,
) -> TrackSamplingRuntimePersistence:
    return TrackSamplingRuntimePersistence(
        load=lambda: store.get_run_track_sampling_state(run_id),
        save=lambda state: store.upsert_run_track_sampling_state(
            run_id=run_id,
            state=state,
            updated_at=now_iso(),
        ),
        replace_materialized_artifacts=lambda artifacts: (
            store.replace_run_track_sampling_artifacts(
                run_id=run_id,
                artifacts=artifacts,
            )
        ),
        replace_generated_x_cup_slots=lambda slots: store.replace_run_generated_x_cup_slots(
            run_id=run_id,
            slots=slots,
            updated_at=now_iso(),
        ),
        load_alt_baselines=lambda: store.active_run_alt_baselines(run_id),
    )


def _resolved_train_config(*, store: ManagerStore, run: ManagedRun, resume: bool):
    if resume:
        config = _resume_train_config(run=run)
        return restore_managed_runtime_track_sampling(
            config,
            store=store,
            run_id=run.id,
            include_artifacts=True,
    )
    if run.source_snapshot_dir is not None and run.source_artifact is not None:
        if run.source_run_id is not None:
            source_run = store.get_run(run.source_run_id)
            if source_run is None:
                raise RuntimeError(f"source run not found for forked run: {run.source_run_id}")
            return build_managed_fork_train_app_config(
                run.config,
                run_id=run.id,
                run_dir=run.run_dir,
                source_run_dir=run.source_snapshot_dir,
                source_artifact=run.source_artifact,
                source_config=source_run.config,
                tensorboard_step_offset=run.lineage_step_offset,
            )
        source_metadata = load_fork_source_metadata(source_dir=run.source_snapshot_dir)
        return build_managed_fork_train_app_config_from_metadata(
            run.config,
            run_id=run.id,
            run_dir=run.run_dir,
            source_run_dir=run.source_snapshot_dir,
            source_artifact=run.source_artifact,
            source_algorithm=source_metadata.source_algorithm,
            source_auxiliary_state_enabled=(
                source_metadata.source_auxiliary_state_enabled
            ),
            source_auxiliary_state_head_arch=(
                source_metadata.source_auxiliary_state_head_arch
            ),
            tensorboard_step_offset=run.lineage_step_offset,
        )
    if run.source_run_id is not None and run.source_artifact is not None:
        source_run = store.get_run(run.source_run_id)
        if source_run is None:
            raise RuntimeError(f"source run not found for forked run: {run.source_run_id}")
        return build_managed_fork_train_app_config(
            run.config,
            run_id=run.id,
            run_dir=run.run_dir,
            source_run_dir=source_run.run_dir,
            source_artifact=run.source_artifact,
            source_config=source_run.config,
            tensorboard_step_offset=run.lineage_step_offset,
        )
    return build_managed_train_app_config(
        run.config,
        run_id=run.id,
        run_dir=run.run_dir,
    )


def _resume_train_config(*, run: ManagedRun):
    # Resume config is projected from SQLite. train_manifest.yaml is only the
    # synchronized mirror written after materialization, never a resume fallback.
    train_config = build_managed_train_app_config(
        run.config,
        run_id=run.id,
        run_dir=run.run_dir,
    )
    return apply_managed_resume_train_config(
        train_config,
        run_dir=run.run_dir,
        tensorboard_step_offset=run.lineage_step_offset,
    )


def _run_paths(run: ManagedRun, *, resume: bool) -> RunPaths:
    if resume:
        return continue_run_paths(run.run_dir)
    return RunPaths(
        run_dir=run.run_dir,
        fresh_run=True,
        runtime_root=run.run_dir / RUN_LAYOUT.runtime_dirname,
        tensorboard_dir=run.run_dir / RUN_LAYOUT.tensorboard_dirname,
        checkpoints_dir=run.run_dir / RUN_LAYOUT.checkpoints_dirname,
        track_sampling_state_path=run.run_dir
        / RUN_LAYOUT.runtime_dirname
        / RUN_LAYOUT.track_sampling_state_filename,
        latest_model_path=run.run_dir / RUN_LAYOUT.model_artifacts.latest,
        latest_policy_path=run.run_dir / RUN_LAYOUT.policy_artifacts.latest,
        best_model_path=run.run_dir / RUN_LAYOUT.model_artifacts.best,
        best_policy_path=run.run_dir / RUN_LAYOUT.policy_artifacts.best,
        final_model_path=run.run_dir / RUN_LAYOUT.model_artifacts.final,
        final_policy_path=run.run_dir / RUN_LAYOUT.policy_artifacts.final,
        baselines_dir=run.run_dir / RUN_LAYOUT.baselines_dirname,
        baseline_state_path=run.run_dir
        / RUN_LAYOUT.baselines_dirname
        / RUN_LAYOUT.baseline_filename,
    )
