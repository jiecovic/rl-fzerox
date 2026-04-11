# src/rl_fzerox/core/training/runs/paths.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactFilenames:
    """The latest/best/final filenames for one artifact family."""

    latest: str
    best: str
    final: str


RUN_CONFIG_FILENAME = "train_config.yaml"
BASELINE_STATE_FILENAME = "baseline.state"
RUN_BASELINE_FILENAME = BASELINE_STATE_FILENAME
WATCH_RUNTIME_ROOTNAME = "watch"
WATCH_SESSION_BASELINE_FILENAME = BASELINE_STATE_FILENAME
RUNTIME_DIRNAME = "runtime"
TENSORBOARD_DIRNAME = "tensorboard"
MODEL_ARTIFACT_FILENAMES = ArtifactFilenames(
    latest="latest_model.zip",
    best="best_model.zip",
    final="final_model.zip",
)
POLICY_ARTIFACT_FILENAMES = ArtifactFilenames(
    latest="latest_policy.zip",
    best="best_policy.zip",
    final="final_policy.zip",
)


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
    """Build the standard directory layout for one training run."""

    resolved_output_root = output_root.expanduser().resolve()
    run_dir = _next_run_dir(resolved_output_root, run_name)
    return RunPaths(
        run_dir=run_dir,
        runtime_root=run_dir / RUNTIME_DIRNAME,
        tensorboard_dir=run_dir / TENSORBOARD_DIRNAME,
        latest_model_path=run_dir / MODEL_ARTIFACT_FILENAMES.latest,
        latest_policy_path=run_dir / POLICY_ARTIFACT_FILENAMES.latest,
        best_model_path=run_dir / MODEL_ARTIFACT_FILENAMES.best,
        best_policy_path=run_dir / POLICY_ARTIFACT_FILENAMES.best,
        final_model_path=run_dir / MODEL_ARTIFACT_FILENAMES.final,
        final_policy_path=run_dir / POLICY_ARTIFACT_FILENAMES.final,
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
        runtime_dir=session_dir / RUNTIME_DIRNAME,
        baseline_state_path=(
            None if baseline_state_path is None else session_dir / WATCH_SESSION_BASELINE_FILENAME
        ),
    )


def ensure_watch_session_dirs(paths: WatchSessionPaths) -> None:
    """Create the directories needed by the current watch session."""

    paths.session_dir.mkdir(parents=True, exist_ok=True)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)


def resolve_train_run_config_path(run_dir: Path) -> Path:
    """Resolve the saved train config snapshot path for one run directory."""

    resolved_run_dir = run_dir.expanduser().resolve()
    config_path = resolved_run_dir / RUN_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No saved train config could be found under run directory {resolved_run_dir}"
        )
    return config_path


def _watch_session_root(*, run_dir: Path | None, runtime_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir.expanduser().resolve() / WATCH_RUNTIME_ROOTNAME
    if runtime_dir is not None:
        return runtime_dir.expanduser().resolve().parent / WATCH_RUNTIME_ROOTNAME
    return Path("local/watch").expanduser().resolve()


def _watch_session_name() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{os.getpid()}"


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
