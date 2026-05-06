# src/rl_fzerox/apps/watch_cli/resolve.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from rl_fzerox.apps._cli import normalize_hydra_overrides
from rl_fzerox.apps.watch_cli.delta import (
    apply_watch_config_delta,
    watch_config_delta,
    watch_config_delta_from_dotlist,
)
from rl_fzerox.core.config import load_watch_app_config
from rl_fzerox.core.config.schema import TrainAppConfig, WatchAppConfig, WatchConfig
from rl_fzerox.core.manager import ManagerStore, default_manager_db_path
from rl_fzerox.core.manager.training import build_managed_train_app_config
from rl_fzerox.core.training.runs import (
    apply_train_run_to_watch_config,
    continue_run_paths,
    load_train_run_config_for_watch,
    materialize_train_run_config,
    materialize_watch_session_config,
)


def resolve_watch_app_config(
    *,
    config_path: Path | None,
    policy_run_dir: Path | None,
    policy_artifact: Literal["latest", "best", "final"] | None,
    manager_db_path: Path | None,
    managed_run_id: str | None,
    overrides: Sequence[str],
) -> WatchAppConfig:
    """Resolve watch config with the same precedence used by the watch CLI."""

    normalized_overrides = normalize_hydra_overrides(overrides)
    if config_path is not None and managed_run_id is not None:
        raise ValueError("--config cannot be combined with --managed-run-id")
    if policy_run_dir is not None and managed_run_id is not None:
        raise ValueError("--run-dir cannot be combined with --managed-run-id")
    cli_run_dir = policy_run_dir.expanduser().resolve() if policy_run_dir is not None else None
    resolved_manager_db_path = (
        manager_db_path.expanduser().resolve()
        if manager_db_path is not None
        else default_manager_db_path().resolve()
    )
    cli_override_delta: dict[str, object] = {}
    train_config: TrainAppConfig | None = None
    if config_path is None:
        if cli_run_dir is None and managed_run_id is None:
            raise ValueError(
                "--config is required unless --run-dir or --managed-run-id is provided"
            )
        if normalized_overrides:
            cli_override_delta = watch_config_delta_from_dotlist(normalized_overrides)
        if managed_run_id is not None:
            cli_run_dir, train_config = _managed_watch_train_config(
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
            artifact=policy_artifact or "latest",
        )
    else:
        config = load_watch_app_config(config_path)
        if normalized_overrides:
            overridden_config = load_watch_app_config(
                config_path,
                overrides=normalized_overrides,
            )
            cli_override_delta = watch_config_delta(
                config,
                overridden_config,
                normalized_overrides,
            )

    resolved_run_dir = cli_run_dir if cli_run_dir is not None else config.watch.policy_run_dir
    if cli_run_dir is None and cli_override_delta:
        resolved_run_dir = apply_watch_config_delta(config, cli_override_delta).watch.policy_run_dir
    if policy_artifact is not None and resolved_run_dir is None:
        raise ValueError(
            "--artifact requires --run-dir, --managed-run-id, or watch.policy_run_dir in the config"
        )
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

    config = apply_x_cup_watch_overrides(config)
    return materialize_watch_session_config(
        config,
        run_dir=config.watch.policy_run_dir,
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
        curriculum=train_config.curriculum,
        train=train_config.train,
        watch=WatchConfig(
            policy_run_dir=run_dir,
            policy_artifact=artifact,
            policy_algorithm=train_config.train.algorithm,
        ),
    )


def _managed_watch_train_config(*, db_path: Path, run_id: str) -> tuple[Path, TrainAppConfig]:
    store = ManagerStore(db_path)
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"managed run not found: {run_id}")
    train_config = build_managed_train_app_config(
        run.config,
        run_id=run.id,
        run_dir=run.run_dir,
    )
    return run.run_dir, materialize_train_run_config(
        train_config,
        run_paths=continue_run_paths(run.run_dir),
    )


def apply_x_cup_watch_overrides(config: WatchAppConfig) -> WatchAppConfig:
    """Disable track-sampling inheritance for watch-only X Cup bootstraps."""

    if not config.watch.x_cup.enabled or not config.env.track_sampling.enabled:
        return config
    return config.model_copy(
        update={
            "env": config.env.model_copy(
                update={
                    "track_sampling": config.env.track_sampling.model_copy(
                        update={"enabled": False}
                    )
                }
            )
        }
    )
