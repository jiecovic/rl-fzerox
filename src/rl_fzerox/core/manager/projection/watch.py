# src/rl_fzerox/core/manager/projection/watch.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from rl_fzerox.core.manager.models import ManagedRun
from rl_fzerox.core.manager.projection.launches import build_managed_train_app_config
from rl_fzerox.core.manager.projection.runtime import restore_managed_runtime_track_sampling
from rl_fzerox.core.manager.store import ManagerStore, default_manager_db_path
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig, WatchAppConfig, WatchConfig
from rl_fzerox.core.runtime_spec.watch_overrides import (
    apply_watch_config_delta,
    watch_config_delta_from_dotlist,
)
from rl_fzerox.core.training.runs import (
    apply_train_run_to_watch_config,
    continue_run_paths,
    load_train_run_config_for_watch,
    materialize_train_run_config,
    materialize_watch_session_config,
)


def resolve_watch_app_config(
    *,
    policy_run_dir: Path | None,
    policy_artifact: Literal["latest", "best", "final"] | None,
    manager_db_path: Path | None,
    managed_run_id: str | None,
    session_name: str | None = None,
    overrides: Sequence[str],
) -> WatchAppConfig:
    """Resolve watch config from the canonical managed or saved-run surfaces."""

    source_count = sum(value is not None for value in (policy_run_dir, managed_run_id))
    if source_count > 1:
        raise ValueError("--run-dir and --managed-run-id are mutually exclusive")
    cli_run_dir = policy_run_dir.expanduser().resolve() if policy_run_dir is not None else None
    resolved_manager_db_path = (
        manager_db_path.expanduser().resolve()
        if manager_db_path is not None
        else default_manager_db_path().resolve()
    )
    cli_override_delta = watch_config_delta_from_dotlist(overrides) if overrides else {}
    train_config: TrainAppConfig | None = None
    lineage_frame_offset: int | None = None
    resolved_policy_artifact: Literal["latest", "best", "final"] = policy_artifact or "latest"
    if policy_artifact is not None and cli_run_dir is None and managed_run_id is None:
        raise ValueError("--artifact requires --run-dir or --managed-run-id")
    if source_count == 0:
        raise ValueError("--run-dir or --managed-run-id is required")
    if managed_run_id is not None:
        cli_run_dir, train_config, lineage_frame_offset = managed_watch_train_config(
            db_path=resolved_manager_db_path,
            run_id=managed_run_id,
        )
    elif cli_run_dir is not None:
        train_config = load_train_run_config_for_watch(cli_run_dir)
    else:
        raise ValueError("watch config resolution requires a run directory")

    config = default_watch_config_from_train_run(
        train_config,
        run_dir=cli_run_dir,
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

    resolved_run_dir = cli_run_dir if cli_run_dir is not None else config.watch.policy_run_dir
    if policy_artifact is not None and resolved_run_dir is None:
        raise ValueError("--artifact requires --run-dir or --managed-run-id")
    if resolved_run_dir is not None:
        resolved_train_config = train_config or load_train_run_config_for_watch(resolved_run_dir)
        config = apply_train_run_to_watch_config(
            config,
            run_dir=resolved_run_dir,
            train_config=resolved_train_config,
        )
        if cli_override_delta:
            config = apply_watch_config_delta(config, cli_override_delta)
        if policy_artifact is not None:
            config = config.model_copy(
                update={
                    "watch": config.watch.model_copy(update={"policy_artifact": policy_artifact})
                }
            )
    elif cli_override_delta:
        config = apply_watch_config_delta(config, cli_override_delta)

    if managed_run_id is not None:
        config = config.model_copy(
            update={
                "watch": config.watch.model_copy(
                    update={
                        "manager_db_path": resolved_manager_db_path,
                        "managed_run_id": managed_run_id,
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
    artifact: Literal["latest", "best", "final"],
) -> WatchAppConfig:
    """Build one minimal watch config directly from a saved train run."""

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
) -> tuple[Path, TrainAppConfig, int | None]:
    """Resolve one manager-owned run into a materialized runtime training config."""

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
    return (
        run.run_dir,
        materialize_train_run_config(
            train_config,
            run_paths=continue_run_paths(run.run_dir),
        ),
        lineage_frame_offset,
    )


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
