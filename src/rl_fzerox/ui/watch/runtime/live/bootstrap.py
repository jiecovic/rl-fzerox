# src/rl_fzerox/ui/watch/runtime/live/bootstrap.py
from __future__ import annotations

import os
from pathlib import Path

from rl_fzerox.core.runtime_spec.schema import WatchAppConfig


def _watch_bootstrap_error_message(config: WatchAppConfig, error: Exception) -> str:
    track_sampling = config.env.track_sampling
    return "\n".join(
        (
            f"watch worker failed during bootstrap: {type(error).__name__}: {error}",
            f"core_path={_path_report(config.emulator.core_path)}",
            f"rom_path={_path_report(config.emulator.rom_path)}",
            f"runtime_dir={_path_report(config.emulator.runtime_dir)}",
            f"baseline_state_path={_path_report(config.emulator.baseline_state_path)}",
            f"renderer={config.emulator.renderer}",
            "track_sampling="
            f"enabled={track_sampling.enabled} entries={len(track_sampling.entries)}",
        )
    )


def _path_report(path: Path | None) -> str:
    if path is None:
        return "-"
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return (
            f"{resolved} file size={resolved.stat().st_size} "
            f"readable={_yes_no(os.access(resolved, os.R_OK))}"
        )
    if resolved.is_dir():
        return (
            f"{resolved} dir readable={_yes_no(os.access(resolved, os.R_OK))} "
            f"writable={_yes_no(os.access(resolved, os.W_OK))}"
        )
    return f"{resolved} missing parent_exists={_yes_no(resolved.parent.exists())}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
