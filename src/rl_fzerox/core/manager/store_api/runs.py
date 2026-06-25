# src/rl_fzerox/core/manager/store_api/runs.py
"""Run lifecycle and runtime methods mixed into the public ManagerStore facade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedRunEvent,
    ManagedRunSummary,
    RunCommand,
    RunStatus,
)
from rl_fzerox.core.manager.registry import lineages as lineage_registry
from rl_fzerox.core.manager.registry import runs as run_registry
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.store_api.common import manager_store as _manager_store

if TYPE_CHECKING:
    from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
    from rl_fzerox.core.training.session.callbacks.track_sampling import (
        TrackSamplingAltBaseline,
        TrackSamplingMaterializedArtifact,
        TrackSamplingRuntimeState,
    )


class RunStoreMixin:
    """ManagerStore facade methods for managed run records and sidecars."""

    def create_run(
        self,
        *,
        run_id: str | None = None,
        name: str,
        config: ManagedRunConfig,
        managed_runs_root: Path | None = None,
        explicit_run_dir: Path | None = None,
        lineage_id: str | None = None,
        lineage_step_offset: int = 0,
        parent_run_id: str | None = None,
        source_run_id: str | None = None,
        source_artifact: Literal["latest", "best"] | None = None,
        source_snapshot_dir: Path | None = None,
        source_num_timesteps: int | None = None,
        exclude_draft_id: str | None = None,
    ) -> ManagedRun:
        return run_registry.create_run(
            _manager_store(self),
            run_id=run_id,
            name=name,
            config=config,
            managed_runs_root=managed_runs_root,
            explicit_run_dir=explicit_run_dir,
            lineage_id=lineage_id,
            lineage_step_offset=lineage_step_offset,
            parent_run_id=parent_run_id,
            source_run_id=source_run_id,
            source_artifact=source_artifact,
            source_snapshot_dir=source_snapshot_dir,
            source_num_timesteps=source_num_timesteps,
            exclude_draft_id=exclude_draft_id,
        )

    def get_run(self, run_id: str) -> ManagedRun | None:
        return run_registry.get_run(_manager_store(self), run_id)

    def list_runs(self) -> tuple[ManagedRun, ...]:
        return run_registry.list_runs(_manager_store(self))

    def list_visible_runs(self) -> tuple[ManagedRun, ...]:
        return run_registry.list_visible_runs(_manager_store(self))

    def list_visible_run_summaries(self) -> tuple[ManagedRunSummary, ...]:
        return run_registry.list_visible_run_summaries(_manager_store(self))

    def list_recent_run_events(
        self,
        run_ids: tuple[str, ...],
        *,
        limit_per_run: int = 6,
    ) -> dict[str, tuple[ManagedRunEvent, ...]]:
        return run_registry.list_recent_run_events(
            _manager_store(self),
            run_ids,
            limit_per_run=limit_per_run,
        )

    def update_run_status(
        self,
        *,
        run_id: str,
        status: RunStatus,
        message: str,
        started_at: str | None = None,
        stopped_at: str | None = None,
    ) -> ManagedRun | None:
        return run_registry.update_run_status(
            _manager_store(self),
            run_id=run_id,
            status=status,
            message=message,
            started_at=started_at,
            stopped_at=stopped_at,
        )

    def update_run_name(self, *, run_id: str, name: str) -> ManagedRun | None:
        return run_registry.update_run_name(_manager_store(self), run_id=run_id, name=name)

    def clear_run_runtime(self, run_id: str) -> None:
        run_registry.clear_run_runtime(_manager_store(self), run_id)

    def get_run_alt_baselines(
        self,
        run_id: str,
    ) -> tuple[TrackSamplingAltBaseline, ...]:
        return run_registry.get_run_alt_baselines(_manager_store(self), run_id)

    def active_run_alt_baselines(self, run_id: str) -> tuple[TrackSamplingAltBaseline, ...]:
        return run_registry.active_run_alt_baselines(_manager_store(self), run_id)

    def upsert_run_alt_baseline(
        self,
        *,
        baseline: TrackSamplingAltBaseline,
    ) -> None:
        run_registry.upsert_run_alt_baseline(_manager_store(self), baseline=baseline)

    def delete_run_alt_baseline(
        self,
        *,
        run_id: str,
        baseline_id: str,
    ) -> bool:
        return run_registry.delete_run_alt_baseline(
            _manager_store(self),
            run_id=run_id,
            baseline_id=baseline_id,
        )

    def clear_run_alt_baselines(self, run_id: str) -> int:
        return run_registry.clear_run_alt_baselines(_manager_store(self), run_id)

    def clear_run_alt_baselines_for_course(self, *, run_id: str, course_key: str) -> int:
        return run_registry.clear_run_alt_baselines_for_course(
            _manager_store(self),
            run_id=run_id,
            course_key=course_key,
        )

    def clear_run_track_sampling_state(self, run_id: str) -> None:
        run_registry.clear_run_track_sampling_state(_manager_store(self), run_id)

    def get_run_track_sampling_state(
        self,
        run_id: str,
    ) -> TrackSamplingRuntimeState | None:
        return run_registry.get_run_track_sampling_state(_manager_store(self), run_id)

    def upsert_run_track_sampling_state(
        self,
        *,
        run_id: str,
        state: TrackSamplingRuntimeState,
        updated_at: str | None = None,
    ) -> None:
        run_registry.upsert_run_track_sampling_state(
            _manager_store(self),
            run_id=run_id,
            state=state,
            updated_at=updated_at,
        )

    def get_run_track_sampling_artifacts(
        self,
        run_id: str,
    ) -> tuple[TrackSamplingMaterializedArtifact, ...]:
        return run_registry.get_run_track_sampling_artifacts(_manager_store(self), run_id)

    def replace_run_track_sampling_artifacts(
        self,
        *,
        run_id: str,
        artifacts: tuple[TrackSamplingMaterializedArtifact, ...],
    ) -> None:
        run_registry.replace_run_track_sampling_artifacts(
            _manager_store(self),
            run_id=run_id,
            artifacts=artifacts,
        )

    def get_run_generated_x_cup_slots(
        self,
        run_id: str,
    ) -> tuple[GeneratedXCupSlot, ...]:
        return run_registry.get_run_generated_x_cup_slots(_manager_store(self), run_id)

    def replace_run_generated_x_cup_slots(
        self,
        *,
        run_id: str,
        slots: tuple[GeneratedXCupSlot, ...],
        updated_at: str | None = None,
    ) -> None:
        run_registry.replace_run_generated_x_cup_slots(
            _manager_store(self),
            run_id=run_id,
            slots=slots,
            updated_at=updated_at,
        )

    def register_run_worker(
        self,
        *,
        run_id: str,
        launch_token: str,
        pid: int,
        launched_at: str,
    ) -> bool:
        return run_registry.register_run_worker(
            _manager_store(self),
            run_id=run_id,
            launch_token=launch_token,
            pid=pid,
            launched_at=launched_at,
        )

    def heartbeat_run_worker(
        self,
        *,
        run_id: str,
        launch_token: str,
        heartbeat_at: str,
    ) -> bool:
        return run_registry.heartbeat_run_worker(
            _manager_store(self),
            run_id=run_id,
            launch_token=launch_token,
            heartbeat_at=heartbeat_at,
        )

    def clear_run_worker(self, run_id: str, *, launch_token: str | None = None) -> None:
        run_registry.clear_run_worker(_manager_store(self), run_id, launch_token=launch_token)

    def mark_worker_boot_failure(
        self,
        *,
        run_id: str,
        launch_token: str,
        message: str,
        failed_at: str,
    ) -> bool:
        return run_registry.mark_worker_boot_failure(
            _manager_store(self),
            run_id=run_id,
            launch_token=launch_token,
            message=message,
            failed_at=failed_at,
        )

    def upsert_run_runtime(
        self,
        *,
        run_id: str,
        total_timesteps: int,
        num_timesteps: int,
        progress_fraction: float,
        updated_at: str,
        fps: float | None = None,
        episode_reward_mean: float | None = None,
        episode_length_mean: float | None = None,
        approx_kl: float | None = None,
        entropy_loss: float | None = None,
        value_loss: float | None = None,
        policy_gradient_loss: float | None = None,
    ) -> None:
        run_registry.upsert_run_runtime(
            _manager_store(self),
            run_id=run_id,
            total_timesteps=total_timesteps,
            num_timesteps=num_timesteps,
            progress_fraction=progress_fraction,
            updated_at=updated_at,
            fps=fps,
            episode_reward_mean=episode_reward_mean,
            episode_length_mean=episode_length_mean,
            approx_kl=approx_kl,
            entropy_loss=entropy_loss,
            value_loss=value_loss,
            policy_gradient_loss=policy_gradient_loss,
        )

    def append_run_event(
        self,
        *,
        run_id: str,
        kind: str,
        message: str,
        created_at: str | None = None,
    ) -> None:
        run_registry.append_run_event(
            _manager_store(self),
            run_id=run_id,
            kind=kind,
            message=message,
            created_at=created_at,
        )

    def update_run_fork_source(
        self,
        *,
        run_id: str,
        source_snapshot_dir: Path,
        source_num_timesteps: int,
        lineage_step_offset: int | None = None,
    ) -> ManagedRun | None:
        return run_registry.update_run_fork_source(
            _manager_store(self),
            run_id=run_id,
            source_snapshot_dir=source_snapshot_dir,
            source_num_timesteps=source_num_timesteps,
            lineage_step_offset=lineage_step_offset,
        )

    def request_run_command(
        self,
        *,
        run_id: str,
        command: RunCommand,
    ) -> ManagedRun | None:
        return run_registry.request_run_command(
            _manager_store(self), run_id=run_id, command=command
        )

    def pending_run_command(self, run_id: str) -> RunCommand | None:
        return run_registry.pending_run_command(_manager_store(self), run_id)

    def clear_run_command(
        self,
        run_id: str,
        *,
        command: RunCommand | None = None,
    ) -> None:
        run_registry.clear_run_command(_manager_store(self), run_id, command=command)

    def delete_run(self, run_id: str) -> bool:
        return lineage_registry.delete_run(_manager_store(self), run_id)

    def delete_lineage(self, lineage_id: str) -> bool:
        return lineage_registry.delete_lineage(_manager_store(self), lineage_id)

    def update_lineage_groups(
        self,
        *,
        lineage_id: str,
        group_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        return lineage_registry.update_lineage_groups(
            _manager_store(self),
            lineage_id=lineage_id,
            group_names=group_names,
        )

    def reconcile_orphaned_runs(self) -> None:
        run_registry.reconcile_orphaned_runs(_manager_store(self))

    def _drain_pending_filesystem_operations(self) -> None:
        run_registry.drain_pending_filesystem_operations(_manager_store(self))
