# src/rl_fzerox/apps/run_manager/launching/save_games.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal

from rl_fzerox.apps.run_manager.launching.processes import (
    fresh_process_log,
    reap_child_when_done,
)
from rl_fzerox.apps.run_manager.launching.watch import (
    WatchLaunchStatus,
    raise_if_watch_exited_early,
    watch_config_overrides,
)
from rl_fzerox.core.manager import ManagedSaveAttempt, ManagerStore
from rl_fzerox.core.manager.registry.common import new_record_id
from rl_fzerox.core.manager.registry.viewers import viewer_lease_is_fresh
from rl_fzerox.core.runtime_spec.paths import project_root_dir
from rl_fzerox.core.runtime_spec.renderers import RendererName

type WatchRenderer = RendererName


def launch_career_mode_runner(
    *,
    store: ManagerStore,
    save_game_id: str,
    device: Literal["cpu", "cuda"],
    renderer: WatchRenderer | None,
    attempt_seed: int | None,
    deterministic_policy: bool,
    recording_enabled: bool = False,
    recording_input_hud_enabled: bool = False,
    recording_upscale_factor: int = 2,
    recording_path: Path | None = None,
    target_kind: str | None = None,
    difficulty: str | None = None,
    cup_id: str | None = None,
    course_id: str | None = None,
    single_target: bool = False,
    perfect_run: bool = False,
    keep_failed_recordings: bool = True,
    target_clear_goal: int = 0,
    reload_policy_between_attempts: bool = True,
) -> WatchLaunchStatus:
    """Launch the visible Career Mode runner for one manager-owned save game."""

    save_game = store.get_save_game(save_game_id)
    if save_game is None:
        raise ValueError(f"save game not found: {save_game_id}")
    if not 1 <= recording_upscale_factor <= 4:
        raise ValueError("recording upscale factor must be an integer from 1 to 4")
    if target_clear_goal < 0:
        raise ValueError("target clear goal must be non-negative")
    if not recording_enabled and (target_clear_goal > 0 or not keep_failed_recordings):
        raise ValueError("target recording options require recording to be enabled")
    if (perfect_run or target_clear_goal > 0 or not keep_failed_recordings) and not single_target:
        raise ValueError("target fishing options require a selected single target")
    lease_id = store.viewer_lease_id(kind="career_mode", owner_id=save_game_id)
    if (
        active_career_mode_runner_pid(
            store=store,
            lease_id=lease_id,
            save_game_id=save_game_id,
        )
        is not None
    ):
        return "already_running"
    store.discard_running_save_attempts(save_game_id=save_game_id)
    attempt = _start_runner_attempt(
        store=store,
        save_game_id=save_game_id,
        target_kind=target_kind,
        difficulty=difficulty,
        cup_id=cup_id,
        course_id=course_id,
    )
    overrides = watch_config_overrides(
        device=device,
        renderer=renderer,
        deterministic_policy=deterministic_policy,
    )
    overrides = (
        *overrides,
        f"watch.reload_policy_between_attempts={str(reload_policy_between_attempts).lower()}",
    )
    if recording_enabled:
        resolved_recording_path = recording_path or default_career_recording_path(
            save_game_id=save_game.id,
            save_game_name=save_game.name,
        )
        overrides = (
            *overrides,
            "watch.recording.enabled=true",
            f"watch.recording.path={resolved_recording_path}",
            f"watch.recording.session_mp4_enabled={str(not single_target).lower()}",
            f"watch.recording.keep_failed_segments={str(keep_failed_recordings).lower()}",
            f"watch.recording.upscale_factor={recording_upscale_factor}",
        )
        if recording_input_hud_enabled:
            overrides = (*overrides, "watch.recording.render_input_hud=true")
    log_path = manager_career_mode_log_path(save_game_id)
    command = [
        sys.executable,
        "-m",
        "rl_fzerox.apps.career_mode",
        "--manager-db-path",
        str(store.db_path),
        "--save-game-id",
        save_game_id,
        "--save-attempt-id",
        attempt.id,
        "--viewer-lease-id",
        lease_id,
        *(("--attempt-seed", str(attempt_seed)) if attempt_seed is not None else ()),
        *(("--single-target",) if single_target else ()),
        *(("--perfect-run",) if perfect_run else ()),
        *(("--target-clear-goal", str(target_clear_goal)) if target_clear_goal > 0 else ()),
        "--policy-mode",
        "deterministic" if deterministic_policy else "stochastic",
        "--",
        *overrides,
    ]
    try:
        cwd = project_root_dir()
        with fresh_process_log(log_path, command=command, cwd=cwd) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError:
        store.discard_running_save_attempts(save_game_id=save_game_id)
        raise
    store.upsert_viewer_lease(
        lease_id=lease_id,
        kind="career_mode",
        owner_id=save_game_id,
        pid=process.pid,
    )
    try:
        raise_if_watch_exited_early(process=process, log_path=log_path)
    except RuntimeError:
        store.clear_viewer_lease(lease_id=lease_id, pid=process.pid)
        store.discard_running_save_attempts(save_game_id=save_game_id)
        raise
    reap_child_when_done(process)
    return "started"


def _start_runner_attempt(
    *,
    store: ManagerStore,
    save_game_id: str,
    target_kind: str | None,
    difficulty: str | None,
    cup_id: str | None,
    course_id: str | None,
) -> ManagedSaveAttempt:
    if not any(value is not None for value in (target_kind, difficulty, cup_id, course_id)):
        return store.start_next_save_attempt(save_game_id)
    if target_kind is None or difficulty is None or cup_id is None:
        raise ValueError("target_kind, difficulty, and cup_id are required together")
    return store.start_target_save_attempt(
        save_game_id,
        target_kind=target_kind,
        difficulty=difficulty,
        cup_id=cup_id,
        course_id=course_id,
    )


def manager_career_mode_log_path(save_game_id: str) -> Path:
    return (
        project_root_dir() / "local" / "manager" / "logs" / f"{save_game_id}.career-mode.log"
    ).resolve()


def default_career_recording_path(*, save_game_id: str, save_game_name: str) -> Path:
    """Return the managed output file for one Career runner recording session."""

    session_dir = new_record_id(save_game_name or save_game_id)
    return Path("local") / "recordings" / "career" / save_game_id / session_dir / "career.mkv"


def active_career_mode_runner_pid(
    *,
    store: ManagerStore,
    lease_id: str,
    save_game_id: str,
) -> int | None:
    lease = store.get_viewer_lease(lease_id)
    if lease is None:
        return None
    if lease.kind != "career_mode" or lease.owner_id != save_game_id:
        store.clear_viewer_lease(lease_id=lease_id)
        return None
    if not viewer_lease_is_fresh(lease):
        store.clear_viewer_lease(lease_id=lease_id, pid=lease.pid)
        return None
    if career_mode_process_matches(pid=lease.pid, save_game_id=save_game_id):
        return lease.pid
    store.clear_viewer_lease(lease_id=lease_id, pid=lease.pid)
    return None


def career_mode_process_matches(*, pid: int, save_game_id: str) -> bool:
    proc_dir = Path("/proc") / str(pid)
    if not proc_dir.is_dir():
        return False
    try:
        cmdline = (proc_dir / "cmdline").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    normalized = cmdline.replace("\x00", " ")
    return (
        "rl_fzerox.apps.career_mode" in normalized
        and f"--save-game-id {save_game_id}" in normalized
    )
