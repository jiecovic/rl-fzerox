# src/rl_fzerox/apps/run_manager/launching/service.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

from rl_fzerox.apps.run_manager.launching.engine_tuning_source import EngineTuningSourceAction
from rl_fzerox.apps.run_manager.launching.run_lifecycle import (
    fork_run,
    launch_run,
    request_run_control,
    resume_run,
)
from rl_fzerox.apps.run_manager.launching.save_games import launch_career_mode_runner
from rl_fzerox.apps.run_manager.launching.watch import WatchLaunchStatus, launch_watch_artifact
from rl_fzerox.apps.run_manager.launching.worker import spawn_manager_worker
from rl_fzerox.core.manager import ManagedRun, ManagedRunConfig, ManagerStore


class ManagerRunLauncher:
    """Launch and control manager-owned training runs."""

    def __init__(self, store: ManagerStore) -> None:
        self._store = store

    def launch(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
        draft_id: str | None = None,
        source_run_id: str | None = None,
        source_artifact: Literal["latest", "best"] | None = None,
        copy_alt_baselines: bool = True,
        engine_tuning_source_action: EngineTuningSourceAction = "convert",
    ) -> ManagedRun:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("run name is required")
        if source_run_id is not None and source_artifact is not None:
            return self._launch_fork_source(
                name=normalized_name,
                config=config,
                draft_id=draft_id,
                source_run_id=source_run_id,
                source_artifact=source_artifact,
                copy_alt_baselines=copy_alt_baselines,
                engine_tuning_source_action=engine_tuning_source_action,
            )
        return launch_run(
            self._store,
            name=normalized_name,
            config=config,
            draft_id=draft_id,
            spawn_worker=self._spawn_worker_for_lifecycle,
        )

    def fork(
        self,
        *,
        run_id: str,
        artifact: Literal["latest", "best"],
        name: str | None = None,
        config: ManagedRunConfig | None = None,
        exclude_draft_id: str | None = None,
        source_snapshot_dir: Path | None = None,
        source_num_timesteps: int | None = None,
        copy_alt_baselines: bool = True,
        engine_tuning_source_action: EngineTuningSourceAction = "convert",
    ) -> ManagedRun:
        return fork_run(
            self._store,
            run_id=run_id,
            artifact=artifact,
            name=name,
            config=config,
            exclude_draft_id=exclude_draft_id,
            source_snapshot_dir=source_snapshot_dir,
            source_num_timesteps=source_num_timesteps,
            copy_alt_baselines=copy_alt_baselines,
            engine_tuning_source_action=engine_tuning_source_action,
            spawn_worker=self._spawn_worker_for_lifecycle,
        )

    def resume(self, *, run_id: str) -> ManagedRun:
        return resume_run(
            self._store,
            run_id=run_id,
            spawn_worker=self._spawn_worker_for_lifecycle,
        )

    def request_pause(self, *, run_id: str) -> ManagedRun:
        return request_run_control(self._store, run_id=run_id, command="pause")

    def request_stop(self, *, run_id: str) -> ManagedRun:
        return request_run_control(self._store, run_id=run_id, command="stop")

    def watch_artifact(
        self,
        *,
        run_id: str,
        artifact: str,
        device: Literal["cpu", "cuda"],
        renderer: Literal["angrylion", "gliden64"] | None,
        deterministic_policy: bool,
    ) -> WatchLaunchStatus:
        """Launch the desktop watch app against one saved artifact for one run."""

        return launch_watch_artifact(
            store=self._store,
            run_id=run_id,
            artifact=artifact,
            device=device,
            renderer=renderer,
            deterministic_policy=deterministic_policy,
        )

    def start_career_mode(
        self,
        *,
        save_game_id: str,
        device: Literal["cpu", "cuda"],
        renderer: Literal["angrylion", "gliden64"] | None,
        attempt_seed: int | None,
        deterministic_policy: bool,
        recording_enabled: bool,
        recording_input_hud_enabled: bool,
        recording_path: Path | None,
        target_kind: str | None = None,
        difficulty: str | None = None,
        cup_id: str | None = None,
        course_id: str | None = None,
        single_target: bool = False,
    ) -> WatchLaunchStatus:
        """Launch the visible Career Mode runner for one save game."""

        return launch_career_mode_runner(
            store=self._store,
            save_game_id=save_game_id,
            device=device,
            renderer=renderer,
            attempt_seed=attempt_seed,
            deterministic_policy=deterministic_policy,
            recording_enabled=recording_enabled,
            recording_input_hud_enabled=recording_input_hud_enabled,
            recording_path=recording_path,
            target_kind=target_kind,
            difficulty=difficulty,
            cup_id=cup_id,
            course_id=course_id,
            single_target=single_target,
        )

    def _spawn_worker(self, *, run_id: str, resume: bool) -> None:
        spawn_manager_worker(store=self._store, run_id=run_id, resume=resume)

    def _spawn_worker_for_lifecycle(
        self,
        *,
        store: ManagerStore,
        run_id: str,
        resume: bool,
    ) -> None:
        del store
        self._spawn_worker(run_id=run_id, resume=resume)

    def _launch_fork_source(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
        draft_id: str | None,
        source_run_id: str,
        source_artifact: Literal["latest", "best"],
        copy_alt_baselines: bool,
        engine_tuning_source_action: EngineTuningSourceAction,
    ) -> ManagedRun:
        if draft_id is None:
            return self.fork(
                run_id=source_run_id,
                artifact=source_artifact,
                name=name,
                config=config,
                copy_alt_baselines=copy_alt_baselines,
                engine_tuning_source_action=engine_tuning_source_action,
            )
        draft = self._store.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"draft not found: {draft_id}")
        if draft.source_run_id != source_run_id or draft.source_artifact != source_artifact:
            raise ValueError("fork draft source no longer matches the requested source run")
        if draft.source_snapshot_dir is None or draft.source_num_timesteps is None:
            raise ValueError(
                "fork draft is missing its pinned checkpoint snapshot; recreate the fork draft"
            )
        return self.fork(
            run_id=source_run_id,
            artifact=source_artifact,
            name=name,
            config=config,
            exclude_draft_id=draft_id,
            source_snapshot_dir=draft.source_snapshot_dir,
            source_num_timesteps=draft.source_num_timesteps,
            copy_alt_baselines=copy_alt_baselines,
            engine_tuning_source_action=engine_tuning_source_action,
        )
