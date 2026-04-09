# src/rl_fzerox/core/training/runs.py
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from rl_fzerox.core.config.schema import TrainAppConfig, WatchAppConfig

RUN_CONFIG_FILENAME = "train_config.yaml"
RUN_BASELINE_FILENAME = "baseline.state"
WATCH_RUNTIME_ROOTNAME = "watch"
WATCH_SESSION_BASELINE_FILENAME = "baseline.state"


@dataclass(frozen=True)
class RunPaths:
    """Filesystem layout for one training run."""

    run_dir: Path
    runtime_root: Path
    tensorboard_dir: Path
    latest_model_path: Path
    latest_policy_path: Path
    best_model_path: Path
    best_policy_path: Path
    final_model_path: Path
    final_policy_path: Path
    baseline_state_path: Path

    def env_runtime_dir(self, env_index: int) -> Path:
        """Return the writable runtime directory for one train env instance."""

        return self.runtime_root / f"env_{env_index:03d}"


@dataclass(frozen=True)
class WatchSessionPaths:
    """Filesystem layout for one interactive watch session."""

    session_dir: Path
    runtime_dir: Path
    baseline_state_path: Path | None


def build_run_paths(*, output_root: Path, run_name: str) -> RunPaths:
    """Build the standard directory layout for one PPO run."""

    resolved_output_root = output_root.expanduser().resolve()
    run_dir = _next_run_dir(resolved_output_root, run_name)
    return RunPaths(
        run_dir=run_dir,
        runtime_root=run_dir / "runtime",
        tensorboard_dir=run_dir / "tensorboard",
        latest_model_path=run_dir / "latest_model.zip",
        latest_policy_path=run_dir / "latest_policy.zip",
        best_model_path=run_dir / "best_model.zip",
        best_policy_path=run_dir / "best_policy.zip",
        final_model_path=run_dir / "final_model.zip",
        final_policy_path=run_dir / "final_policy.zip",
        baseline_state_path=run_dir / RUN_BASELINE_FILENAME,
    )


def ensure_run_dirs(paths: RunPaths) -> None:
    """Create the directories needed by the current run."""

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.runtime_root.mkdir(parents=True, exist_ok=True)
    paths.tensorboard_dir.mkdir(parents=True, exist_ok=True)


def build_watch_session_paths(
    *,
    run_dir: Path | None,
    runtime_dir: Path | None,
    baseline_state_path: Path | None,
    session_name: str | None = None,
) -> WatchSessionPaths:
    """Build one isolated runtime/baseline layout for a watch session."""

    session_root = _watch_session_root(run_dir=run_dir, runtime_dir=runtime_dir)
    session_dir = session_root / (session_name or _watch_session_name())
    return WatchSessionPaths(
        session_dir=session_dir,
        runtime_dir=session_dir / "runtime",
        baseline_state_path=(
            None if baseline_state_path is None else session_dir / WATCH_SESSION_BASELINE_FILENAME
        ),
    )


def ensure_watch_session_dirs(paths: WatchSessionPaths) -> None:
    """Create the directories needed by the current watch session."""

    paths.session_dir.mkdir(parents=True, exist_ok=True)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)


def materialize_watch_session_config(
    watch_config: WatchAppConfig,
    *,
    run_dir: Path | None,
    session_name: str | None = None,
) -> WatchAppConfig:
    """Rewrite one watch config to use isolated per-session scratch paths."""

    paths = build_watch_session_paths(
        run_dir=run_dir,
        runtime_dir=watch_config.emulator.runtime_dir,
        baseline_state_path=watch_config.emulator.baseline_state_path,
        session_name=session_name,
    )
    ensure_watch_session_dirs(paths)
    _copy_state_file(
        source=watch_config.emulator.baseline_state_path,
        destination=paths.baseline_state_path,
    )
    return watch_config.model_copy(
        update={
            "emulator": watch_config.emulator.model_copy(
                update={
                    "runtime_dir": paths.runtime_dir,
                    "baseline_state_path": paths.baseline_state_path,
                }
            )
        }
    )


def materialize_train_run_config(
    config: TrainAppConfig,
    *,
    run_paths: RunPaths,
) -> TrainAppConfig:
    """Rewrite one train config to use run-local runtime and baseline files."""

    baseline_state_path = config.emulator.baseline_state_path
    if baseline_state_path is not None:
        _copy_state_file(
            source=baseline_state_path,
            destination=run_paths.baseline_state_path,
        )
    return config.model_copy(
        update={
            "emulator": config.emulator.model_copy(
                update={
                    "runtime_dir": run_paths.runtime_root,
                    "baseline_state_path": (
                        None if baseline_state_path is None else run_paths.baseline_state_path
                    ),
                }
            )
        }
    )


def save_train_run_config(*, config: TrainAppConfig, run_dir: Path) -> Path:
    """Persist one resolved train config snapshot next to a training run."""

    config_path = run_dir / RUN_CONFIG_FILENAME
    data = config.model_dump(mode="json", exclude_none=True)
    OmegaConf.save(config=OmegaConf.create(data), f=str(config_path))
    return config_path


def load_train_run_config(run_dir: Path) -> TrainAppConfig:
    """Load one previously saved resolved train config snapshot."""

    config_path = resolve_train_run_config_path(run_dir)
    loaded = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Train run config must resolve to a mapping: {config_path}")
    return TrainAppConfig.model_validate(loaded)


def resolve_train_run_config_path(run_dir: Path) -> Path:
    """Resolve the saved train config snapshot path for one run directory."""

    resolved_run_dir = run_dir.expanduser().resolve()
    config_path = resolved_run_dir / RUN_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No saved train config could be found under run directory {resolved_run_dir}"
        )
    return config_path


def resolve_latest_model_path(run_dir: Path) -> Path:
    """Resolve the newest full PPO artifact from a run directory."""

    return resolve_model_artifact_path(run_dir, artifact="latest")


def resolve_latest_policy_path(run_dir: Path) -> Path:
    """Resolve the newest policy-only artifact from a run directory."""

    return resolve_policy_artifact_path(run_dir, artifact="latest")


def resolve_model_artifact_path(
    run_dir: Path,
    *,
    artifact: str,
) -> Path:
    """Resolve one full-model artifact from a run directory."""

    return _resolve_artifact_path(
        run_dir=run_dir,
        artifact=artifact,
        latest_filename="latest_model.zip",
        best_filename="best_model.zip",
        final_filename="final_model.zip",
    )


def resolve_policy_artifact_path(
    run_dir: Path,
    *,
    artifact: str,
) -> Path:
    """Resolve one policy-only artifact from a run directory."""

    return _resolve_artifact_path(
        run_dir=run_dir,
        artifact=artifact,
        latest_filename="latest_policy.zip",
        best_filename="best_policy.zip",
        final_filename="final_policy.zip",
    )


def apply_train_run_to_watch_config(
    watch_config: WatchAppConfig,
    *,
    run_dir: Path,
    train_config: TrainAppConfig,
) -> WatchAppConfig:
    """Inherit emulator/env settings from a training run for policy watch mode."""

    merged_emulator_config = train_config.emulator
    if (
        watch_config.emulator.runtime_dir is not None
        and merged_emulator_config.runtime_dir != watch_config.emulator.runtime_dir
    ):
        merged_emulator_config = merged_emulator_config.model_copy(
            update={
                "runtime_dir": watch_config.emulator.runtime_dir,
            }
        )

    if (
        merged_emulator_config.baseline_state_path is None
        and watch_config.emulator.baseline_state_path is not None
    ):
        merged_emulator_config = merged_emulator_config.model_copy(
            update={
                "baseline_state_path": watch_config.emulator.baseline_state_path,
            }
        )

    return watch_config.model_copy(
        update={
            "seed": train_config.seed,
            "emulator": merged_emulator_config,
            "env": train_config.env,
            "reward": train_config.reward,
            "curriculum": train_config.curriculum,
            "watch": watch_config.watch.model_copy(
                update={"policy_run_dir": run_dir.expanduser().resolve()}
            ),
        }
    )


def _copy_state_file(*, source: Path | None, destination: Path | None) -> None:
    if source is None or destination is None:
        return
    resolved_source = source.expanduser().resolve()
    resolved_destination = destination.expanduser().resolve()
    if resolved_source == resolved_destination:
        return
    if not resolved_source.is_file():
        return
    resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_source, resolved_destination)


def _watch_session_root(*, run_dir: Path | None, runtime_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir.expanduser().resolve() / WATCH_RUNTIME_ROOTNAME
    if runtime_dir is not None:
        return runtime_dir.expanduser().resolve().parent / WATCH_RUNTIME_ROOTNAME
    return Path("local/watch").expanduser().resolve()


def _watch_session_name() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{os.getpid()}"


def _resolve_artifact_path(
    *,
    run_dir: Path,
    artifact: str,
    latest_filename: str,
    best_filename: str,
    final_filename: str,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    if artifact == "latest":
        preferred_filenames = (latest_filename, final_filename, best_filename)
    elif artifact == "best":
        preferred_filenames = (best_filename,)
    elif artifact == "final":
        preferred_filenames = (final_filename,)
    else:
        raise ValueError(f"Unsupported artifact kind: {artifact!r}")

    for filename in preferred_filenames:
        resolved_path = resolved_run_dir / filename
        if resolved_path.is_file():
            return resolved_path

    raise FileNotFoundError(f"No artifact could be found under run directory {resolved_run_dir}")


def _next_run_dir(output_root: Path, run_name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_name}_"
    next_index = 1

    for child in output_root.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        suffix = child.name.removeprefix(prefix)
        if suffix.isdigit():
            next_index = max(next_index, int(suffix) + 1)

    return output_root / f"{run_name}_{next_index:04d}"
