# src/rl_fzerox/ui/watch/runtime/policy.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rl_fzerox.core.training.inference import PolicyRunner


class _CurriculumStagePolicyRunner(Protocol):
    @property
    def checkpoint_curriculum_stage_index(self) -> int | None: ...

    @property
    def supports_action_masks(self) -> bool: ...

    def refresh(self) -> None: ...
    def reset(self) -> None: ...


class _CheckpointStageSyncEnv(Protocol):
    def sync_checkpoint_curriculum_stage(self, stage_index: int | None) -> None: ...


def _load_policy_runner(
    policy_run_dir: Path | None,
    *,
    artifact: str,
    device: str,
) -> PolicyRunner | None:
    if policy_run_dir is None:
        return None
    from rl_fzerox.core.training.inference import load_policy_runner

    return load_policy_runner(policy_run_dir, artifact=artifact, device=device)


def _policy_label(policy_runner: PolicyRunner | None) -> str | None:
    if policy_runner is None:
        return None
    return policy_runner.label


def _policy_reload_age_seconds(policy_runner: PolicyRunner | None) -> float | None:
    if policy_runner is None:
        return None
    return policy_runner.reload_age_seconds


def _policy_reload_error(policy_runner: PolicyRunner | None) -> str | None:
    if policy_runner is None:
        return None
    return policy_runner.last_reload_error


def _policy_curriculum_stage(policy_runner: PolicyRunner | None) -> str | None:
    if policy_runner is None:
        return None
    return policy_runner.checkpoint_curriculum_stage


def _policy_deterministic(policy_runner: PolicyRunner | None, deterministic: bool) -> bool | None:
    if policy_runner is None:
        return None
    return deterministic


def _reset_policy_runner(policy_runner: _CurriculumStagePolicyRunner | None) -> None:
    if policy_runner is None:
        return
    policy_runner.reset()


def _sync_policy_curriculum_stage(
    policy_runner: _CurriculumStagePolicyRunner | None,
    env: _CheckpointStageSyncEnv,
) -> None:
    if policy_runner is None:
        return
    policy_runner.refresh()
    env.sync_checkpoint_curriculum_stage(policy_runner.checkpoint_curriculum_stage_index)


def _persist_reload_error(
    *,
    reload_error: str | None,
    runtime_dir: Path | None,
    last_logged_reload_error: str | None,
) -> str | None:
    if reload_error is None or runtime_dir is None or reload_error == last_logged_reload_error:
        return last_logged_reload_error

    log_path = runtime_dir.parent / "reload_error.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(reload_error + "\n", encoding="utf-8")
    return reload_error
