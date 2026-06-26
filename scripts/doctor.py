#!/usr/bin/env python3
# scripts/doctor.py
"""Check whether the local rl-fzerox install can run the app.

The doctor is intentionally local-machine focused. It validates the virtual
environment, native extension, frontend dependencies, and user-provided runtime
assets that cannot be committed to the repository.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    label: str
    detail: str
    required: bool = True


def main() -> int:
    results = [
        command_check("git"),
        command_check("cargo"),
        command_check("npm"),
        command_check("just", required=False),
        path_check(".venv python", VENV_PYTHON),
        python_check("python version", "import sys; print(sys.version.split()[0])"),
        python_check(
            "native extension",
            "import fzerox_emulator._native as native; print(native.__file__)",
        ),
        python_check(
            "torch",
            "import torch; print(f'{torch.__version__}; cuda={torch.cuda.is_available()}')",
        ),
        path_check(
            "run-manager node_modules",
            REPO_ROOT / "web" / "run-manager" / "node_modules",
        ),
        path_check(
            "libretro core",
            REPO_ROOT / "local" / "libretro" / "mupen64plus_next_libretro.so",
        ),
        rom_path_check(),
        checkpoint_catalog_check(),
    ]

    failed = False
    for result in results:
        marker = "ok" if result.ok else ("missing" if result.required else "warn")
        print(f"[{marker}] {result.label}: {result.detail}")
        failed = failed or (result.required and not result.ok)
    return 1 if failed else 0


def command_check(name: str, *, required: bool = True) -> CheckResult:
    path = shutil.which(name)
    return CheckResult(path is not None, name, path or "not found", required=required)


def path_check(label: str, path: Path) -> CheckResult:
    return CheckResult(path.exists(), label, str(path))


def rom_path_check() -> CheckResult:
    return python_check(
        "F-Zero X ROM",
        "\n".join(
            (
                "from rl_fzerox.core.runtime_spec.roms import (",
                "    FZeroXRomResolutionError,",
                "    resolve_fzerox_rom_path,",
                ")",
                "try:",
                "    print(resolve_fzerox_rom_path())",
                "except FZeroXRomResolutionError as exc:",
                "    print(exc)",
                "    raise SystemExit(1)",
            )
        ),
    )


def python_check(label: str, code: str) -> CheckResult:
    if not VENV_PYTHON.exists():
        return CheckResult(False, label, f"{VENV_PYTHON} does not exist")
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return CheckResult(True, label, result.stdout.strip() or "ok")
    detail = (result.stderr or result.stdout).strip().splitlines()
    return CheckResult(False, label, detail[-1] if detail else "failed")


def checkpoint_catalog_check() -> CheckResult:
    catalog_path = REPO_ROOT / "published_checkpoints.json"
    if not catalog_path.exists():
        return CheckResult(False, "checkpoint catalog", str(catalog_path))
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CheckResult(False, "checkpoint catalog", f"invalid json: {exc}")
    entries = data.get("entries")
    if not isinstance(entries, list):
        return CheckResult(False, "checkpoint catalog", "missing entries list")
    return CheckResult(True, "checkpoint catalog", f"{len(entries)} published checkpoint(s)")


if __name__ == "__main__":
    raise SystemExit(main())
