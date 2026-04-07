# src/rl_fzerox/core/training/runs.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from rl_fzerox.core.config.schema import TrainAppConfig, WatchAppConfig

RUN_CONFIG_FILENAME = "train_config.yaml"


@dataclass(frozen=True)
class RunPaths:
    """Filesystem layout for one training run."""

    run_dir: Path
    checkpoints_dir: Path
    policy_checkpoints_dir: Path
    tensorboard_dir: Path
    final_model_path: Path
    final_policy_path: Path


def build_run_paths(*, output_root: Path, run_name: str) -> RunPaths:
    """Build the standard directory layout for one PPO run."""

    resolved_output_root = output_root.expanduser().resolve()
    run_dir = _next_run_dir(resolved_output_root, run_name)
    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        policy_checkpoints_dir=run_dir / "policy_checkpoints",
        tensorboard_dir=run_dir / "tensorboard",
        final_model_path=run_dir / "final_model.zip",
        final_policy_path=run_dir / "final_policy.zip",
    )


def ensure_run_dirs(paths: RunPaths) -> None:
    """Create the directories needed by the current run."""

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.policy_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.tensorboard_dir.mkdir(parents=True, exist_ok=True)


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

    return _resolve_latest_path(
        directory_name="checkpoints",
        final_filename="final_model.zip",
        run_dir=run_dir,
    )


def resolve_latest_policy_path(run_dir: Path) -> Path:
    """Resolve the newest policy-only artifact from a run directory."""

    return _resolve_latest_path(
        directory_name="policy_checkpoints",
        final_filename="final_policy.zip",
        run_dir=run_dir,
    )


def apply_train_run_to_watch_config(
    watch_config: WatchAppConfig,
    *,
    run_dir: Path,
    train_config: TrainAppConfig,
) -> WatchAppConfig:
    """Inherit emulator/env settings from a training run for policy watch mode."""

    return watch_config.model_copy(
        update={
            "seed": train_config.seed,
            "emulator": train_config.emulator,
            "env": train_config.env,
            "watch": watch_config.watch.model_copy(
                update={"policy_run_dir": run_dir.expanduser().resolve()}
            ),
        }
    )


def _resolve_latest_path(*, directory_name: str, final_filename: str, run_dir: Path) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    artifact_dir = resolved_run_dir / directory_name
    candidates = sorted(artifact_dir.glob("*.zip"), key=lambda path: path.name)
    if candidates:
        return candidates[-1]

    final_path = resolved_run_dir / final_filename
    if final_path.is_file():
        return final_path

    raise FileNotFoundError(
        f"No artifact could be found under run directory {resolved_run_dir}"
    )


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
