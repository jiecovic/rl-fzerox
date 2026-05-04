# src/rl_fzerox/apps/run_manager/desktop.py
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def open_directory(path: Path) -> None:
    """Open one directory in the host file manager."""

    resolved_path = path.expanduser().resolve()
    target_path = resolved_path if resolved_path.exists() else resolved_path.parent
    command = _open_directory_command(target_path)
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"desktop opener not available: {command[0]}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = stderr or f"{command[0]} exited with status {completed.returncode}"
        raise RuntimeError(f"failed to open run directory: {detail}")


def _open_directory_command(path: Path) -> list[str]:
    if _is_wsl():
        return [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"Start-Process '{_powershell_single_quoted(_wsl_windows_path(path))}'",
        ]
    if sys.platform.startswith("linux"):
        if shutil.which("xdg-open") is not None:
            return ["xdg-open", str(path)]
        if shutil.which("gio") is not None:
            return ["gio", "open", str(path)]
        raise RuntimeError("no Linux desktop opener found (expected xdg-open or gio)")
    if sys.platform == "darwin":
        return ["open", str(path)]
    if sys.platform.startswith("win"):
        return ["explorer", str(path)]
    raise RuntimeError(f"unsupported desktop platform: {sys.platform}")


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8")
    except OSError:
        return False
    return "microsoft" in version.lower()


def _wsl_windows_path(path: Path) -> str:
    completed = subprocess.run(
        ["wslpath", "-w", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = stderr or f"wslpath exited with status {completed.returncode}"
        raise RuntimeError(f"failed to convert run directory for Windows Explorer: {detail}")
    resolved = completed.stdout.strip()
    if not resolved:
        raise RuntimeError("wslpath returned an empty Windows path")
    return resolved


def _powershell_single_quoted(value: str) -> str:
    return value.replace("'", "''")
