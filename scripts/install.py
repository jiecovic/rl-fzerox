#!/usr/bin/env python3
# scripts/install.py
"""Bootstrap the local rl-fzerox development and runtime environment.

This script owns the first-install path. The Justfile wrappers call into it so
README instructions, manual installs, and local developer setup stay aligned.
It intentionally stops before ROM/libretro asset validation; `scripts/doctor.py`
reports those machine-local runtime requirements after install.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = REPO_ROOT / ".venv"
PYTHON_VERSION = "3.12"

TorchMode = Literal["cpu", "cu128", "skip"]

TORCH_INDEX_URLS: dict[TorchMode, str] = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu128": "https://download.pytorch.org/whl/cu128",
}


@dataclass(frozen=True)
class InstallOptions:
    venv: Path
    python: str | None
    torch: TorchMode
    skip_native: bool
    skip_web: bool


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    require_command("git", reason="pip installs sb3x from GitHub")
    if not options.skip_native:
        require_command("cargo", reason="the native emulator extension is a Rust/PyO3 build")
    if not options.skip_web:
        require_command("npm", reason="the run manager frontend uses npm dependencies")

    create_local_dirs()
    venv_python = ensure_venv(options)
    ensure_pip(venv_python)

    run([str(venv_python), "-m", "pip", "install", "-U", "pip"])
    install_torch(venv_python, options.torch)
    run([str(venv_python), "-m", "pip", "install", "-e", ".[dev,watch,train]"])

    if not options.skip_native:
        run([str(venv_python), "-m", "maturin", "develop", "-r", "-q", "--skip-install"])
    if not options.skip_web:
        run(["npm", "install", "--prefix", "web/run-manager"])

    print()
    print("Install complete.")
    print(f"Python environment: {options.venv}")
    print("Next:")
    print("  ./doctor")
    print("  ./fzerox")
    return 0


def parse_args(argv: list[str] | None) -> InstallOptions:
    parser = argparse.ArgumentParser(description="Install rl-fzerox locally.")
    parser.add_argument(
        "--venv",
        default=str(DEFAULT_VENV),
        help="virtual environment path, default: .venv",
    )
    parser.add_argument(
        "--python",
        help=(
            "python executable used to create the venv. If omitted, uv with "
            "Python 3.12 is preferred, then python3.12."
        ),
    )
    parser.add_argument(
        "--torch",
        choices=("cpu", "cu128", "skip"),
        default="cpu",
        help="install a CPU torch wheel, CUDA 12.8 torch wheel, or skip torch preinstall",
    )
    parser.add_argument(
        "--skip-native",
        action="store_true",
        help="skip the Rust/PyO3 native release build",
    )
    parser.add_argument(
        "--skip-web",
        action="store_true",
        help="skip run-manager npm dependencies",
    )
    args = parser.parse_args(argv)
    return InstallOptions(
        venv=Path(args.venv).expanduser(),
        python=args.python,
        torch=args.torch,
        skip_native=args.skip_native,
        skip_web=args.skip_web,
    )


def ensure_venv(options: InstallOptions) -> Path:
    venv = options.venv
    if not venv.is_absolute():
        venv = REPO_ROOT / venv
    python = venv / "bin" / "python"
    if python.exists():
        return python

    if options.python is not None:
        run([options.python, "-m", "venv", str(venv)])
        return python

    uv = shutil.which("uv")
    if uv is not None:
        run([uv, "python", "install", PYTHON_VERSION])
        run(
            [uv, "venv", "--seed", "--python", PYTHON_VERSION, str(venv)],
            env={"UV_LINK_MODE": "copy"},
        )
        return python

    python312 = shutil.which("python3.12")
    if python312 is None:
        raise SystemExit(
            "Could not create .venv: install uv or provide Python 3.12 with "
            "`python scripts/install.py --python /path/to/python3.12`."
        )
    run([python312, "-m", "venv", str(venv)])
    return python


def ensure_pip(venv_python: Path) -> None:
    if command_succeeds([str(venv_python), "-m", "pip", "--version"]):
        return
    run([str(venv_python), "-m", "ensurepip", "--upgrade"])
    if not command_succeeds([str(venv_python), "-m", "pip", "--version"]):
        raise SystemExit(
            "The virtual environment has no pip. Recreate it with uv "
            "(`uv venv --seed --python 3.12 .venv`) or install ensurepip support."
        )


def install_torch(venv_python: Path, mode: TorchMode) -> None:
    if mode == "skip":
        return
    index_url = TORCH_INDEX_URLS[mode]
    run([str(venv_python), "-m", "pip", "install", "torch", "--index-url", index_url])


def create_local_dirs() -> None:
    for relative in (
        "local/libretro",
        "local/roms",
        "local/manager",
        "local/runs",
        "local/checkpoint_bundles",
    ):
        (REPO_ROOT / relative).mkdir(parents=True, exist_ok=True)


def require_command(name: str, *, reason: str) -> None:
    if shutil.which(name) is not None:
        return
    raise SystemExit(f"Missing required command `{name}`: {reason}.")


def command_succeeds(args: list[str]) -> bool:
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def run(args: list[str], *, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    print(f"$ {shlex.join(args)}")
    subprocess.run(args, cwd=REPO_ROOT, env=merged_env, check=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
