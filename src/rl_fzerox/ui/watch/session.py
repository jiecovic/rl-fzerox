# src/rl_fzerox/ui/watch/session.py
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from fzerox_emulator import Emulator, FZeroXTelemetry

if TYPE_CHECKING:
    from rl_fzerox.core.training.inference import PolicyRunner


def _telemetry_from_info(info: dict[str, object]) -> FZeroXTelemetry | None:
    telemetry = info.get("telemetry")
    if isinstance(telemetry, FZeroXTelemetry):
        return telemetry
    return None


def _load_policy_runner(
    policy_run_dir: Path | None,
    *,
    artifact: str,
) -> PolicyRunner | None:
    if policy_run_dir is None:
        return None
    from rl_fzerox.core.training.inference import load_policy_runner

    return load_policy_runner(policy_run_dir, artifact=artifact)


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


def _persist_reload_error(
    *,
    reload_error: str | None,
    runtime_dir: Path | None,
    last_logged_reload_error: str | None,
) -> str | None:
    if (
        reload_error is None
        or runtime_dir is None
        or reload_error == last_logged_reload_error
    ):
        return last_logged_reload_error

    log_path = runtime_dir.parent / "reload_error.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(reload_error + "\n", encoding="utf-8")
    return reload_error


def _save_baseline_state(*, emulator: Emulator, baseline_state_path: Path | None) -> None:
    emulator.capture_current_as_baseline(baseline_state_path)


def _with_viewer_fps(
    info: dict[str, object],
    *,
    last_draw_time: float | None,
    current_viewer_fps: float,
) -> tuple[dict[str, object], float, float]:
    now = time.perf_counter()
    if last_draw_time is None:
        viewer_fps = current_viewer_fps
    else:
        dt = now - last_draw_time
        instant_fps = 0.0 if dt <= 0.0 else 1.0 / dt
        viewer_fps = instant_fps if current_viewer_fps <= 0.0 else (
            (0.8 * current_viewer_fps) + (0.2 * instant_fps)
        )
    draw_info = dict(info)
    draw_info["viewer_fps"] = viewer_fps
    return draw_info, now, viewer_fps
