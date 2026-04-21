# src/rl_fzerox/core/training/runs/paths.py
"""Canonical filesystem layout metadata and path builders for training runs."""

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


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Canonical filenames and directory names inside one run directory."""

    config_filename: str
    baseline_filename: str
    baselines_dirname: str
    watch_rootname: str
    runtime_dirname: str
    tensorboard_dirname: str
    model_artifacts: ArtifactFilenames
    policy_artifacts: ArtifactFilenames


RUN_LAYOUT = RunLayout(
    config_filename="train_config.yaml",
    baseline_filename="baseline.state",
    baselines_dirname="baselines",
    watch_rootname="watch",
    runtime_dirname="runtime",
    tensorboard_dirname="tensorboard",
    model_artifacts=ArtifactFilenames(
        latest="latest_model.zip",
        best="best_model.zip",
        final="final_model.zip",
    ),
    policy_artifacts=ArtifactFilenames(
        latest="latest_policy.zip",
        best="best_policy.zip",
        final="final_policy.zip",
    ),
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
    baselines_dir: Path
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
    return _run_paths(run_dir)


def reserve_run_paths(*, output_root: Path, run_name: str) -> RunPaths:
    """Atomically reserve a fresh training run directory.

    Training must never write into an existing run directory. This function
    creates the selected top-level run directory with ``exist_ok=False`` and
    retries the next numeric suffix if another process wins the same name.
    """

    resolved_output_root = output_root.expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_name}_"
    next_index = _next_run_index(resolved_output_root, prefix=prefix)

    while True:
        run_dir = resolved_output_root / f"{run_name}_{next_index:04d}"
        try:
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            next_index += 1
            continue
        return _run_paths(run_dir)


def _run_paths(run_dir: Path) -> RunPaths:
    return RunPaths(
        run_dir=run_dir,
        runtime_root=run_dir / RUN_LAYOUT.runtime_dirname,
        tensorboard_dir=run_dir / RUN_LAYOUT.tensorboard_dirname,
        latest_model_path=run_dir / RUN_LAYOUT.model_artifacts.latest,
        latest_policy_path=run_dir / RUN_LAYOUT.policy_artifacts.latest,
        best_model_path=run_dir / RUN_LAYOUT.model_artifacts.best,
        best_policy_path=run_dir / RUN_LAYOUT.policy_artifacts.best,
        final_model_path=run_dir / RUN_LAYOUT.model_artifacts.final,
        final_policy_path=run_dir / RUN_LAYOUT.policy_artifacts.final,
        baselines_dir=run_dir / RUN_LAYOUT.baselines_dirname,
        baseline_state_path=run_dir / RUN_LAYOUT.baselines_dirname / RUN_LAYOUT.baseline_filename,
    )


def ensure_run_dirs(paths: RunPaths) -> None:
    """Create the directories needed by the current run."""

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.runtime_root.mkdir(parents=True, exist_ok=True)
    paths.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    paths.baselines_dir.mkdir(parents=True, exist_ok=True)


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
        runtime_dir=session_dir / RUN_LAYOUT.runtime_dirname,
        baseline_state_path=(
            None if baseline_state_path is None else session_dir / RUN_LAYOUT.baseline_filename
        ),
    )


def ensure_watch_session_dirs(paths: WatchSessionPaths) -> None:
    """Create the directories needed by the current watch session."""

    paths.session_dir.mkdir(parents=True, exist_ok=True)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)


def resolve_train_run_config_path(run_dir: Path) -> Path:
    """Resolve the saved train config snapshot path for one run directory."""

    resolved_run_dir = run_dir.expanduser().resolve()
    config_path = resolved_run_dir / RUN_LAYOUT.config_filename
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No saved train config could be found under run directory {resolved_run_dir}"
        )
    return config_path


def _watch_session_root(*, run_dir: Path | None, runtime_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir.expanduser().resolve() / RUN_LAYOUT.watch_rootname
    if runtime_dir is not None:
        return runtime_dir.expanduser().resolve().parent / RUN_LAYOUT.watch_rootname
    return Path("local/watch").expanduser().resolve()


def _watch_session_name() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{os.getpid()}"


def _next_run_dir(output_root: Path, run_name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_name}_"
    next_index = _next_run_index(output_root, prefix=prefix)

    return output_root / f"{run_name}_{next_index:04d}"


def _next_run_index(output_root: Path, *, prefix: str) -> int:
    next_index = 1

    for child in output_root.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        suffix = child.name.removeprefix(prefix)
        if suffix.isdigit():
            next_index = max(next_index, int(suffix) + 1)
    return next_index
