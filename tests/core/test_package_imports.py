# tests/core/test_package_imports.py
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_importing_fzerox_emulator_package_does_not_load_native_extension() -> None:
    result = _probe_import(
        module_name="fzerox_emulator",
        watched_modules=("fzerox_emulator._native",),
    )

    assert result["after_import"]["fzerox_emulator._native"] is False


def test_importing_rl_fzerox_package_does_not_load_env_or_native() -> None:
    result = _probe_import(
        module_name="rl_fzerox",
        watched_modules=(
            "rl_fzerox.core.envs",
            "rl_fzerox.core.envs.env",
            "fzerox_emulator._native",
        ),
    )

    assert result["after_import"]["rl_fzerox.core.envs"] is False
    assert result["after_import"]["rl_fzerox.core.envs.env"] is False
    assert result["after_import"]["fzerox_emulator._native"] is False


def test_importing_training_runs_package_does_not_load_config_or_native() -> None:
    result = _probe_import(
        module_name="rl_fzerox.core.training.runs",
        watched_modules=(
            "rl_fzerox.core.training.runs.config",
            "rl_fzerox.core.training.runs.baseline_materializer",
            "fzerox_emulator._native",
        ),
    )

    assert result["after_import"]["rl_fzerox.core.training.runs.config"] is False
    assert (
        result["after_import"]["rl_fzerox.core.training.runs.baseline_materializer"] is False
    )
    assert result["after_import"]["fzerox_emulator._native"] is False


def test_loading_training_run_config_helper_stays_native_free() -> None:
    result = _probe_import(
        module_name="rl_fzerox.core.training.runs",
        attribute_name="load_train_run_config",
        watched_modules=(
            "rl_fzerox.core.training.runs.config",
            "rl_fzerox.core.training.runs.baseline_materializer",
            "fzerox_emulator._native",
        ),
    )

    assert result["after_import"]["rl_fzerox.core.training.runs.config"] is False
    assert result["after_access"]["rl_fzerox.core.training.runs.config"] is True
    assert result["after_access"]["rl_fzerox.core.training.runs.baseline_materializer"] is False
    assert result["after_access"]["fzerox_emulator._native"] is False


def test_loading_training_run_paths_helper_stays_materializer_free() -> None:
    result = _probe_import(
        module_name="rl_fzerox.core.training.runs",
        attribute_name="build_run_paths",
        watched_modules=(
            "rl_fzerox.core.training.runs.paths",
            "rl_fzerox.core.training.runs.config",
            "rl_fzerox.core.training.runs.baseline_materializer",
            "fzerox_emulator._native",
        ),
    )

    assert result["after_access"]["rl_fzerox.core.training.runs.paths"] is True
    assert result["after_access"]["rl_fzerox.core.training.runs.config"] is False
    assert result["after_access"]["rl_fzerox.core.training.runs.baseline_materializer"] is False
    assert result["after_access"]["fzerox_emulator._native"] is False


def _probe_import(
    *,
    module_name: str,
    watched_modules: tuple[str, ...],
    attribute_name: str | None = None,
) -> dict[str, dict[str, bool]]:
    code = """
import importlib
import json
import sys

module_name = sys.argv[1]
attribute_name = None if sys.argv[2] == "__none__" else sys.argv[2]
watched = sys.argv[3:]
module = importlib.import_module(module_name)
after_import = {name: name in sys.modules for name in watched}
if attribute_name is not None:
    getattr(module, attribute_name)
after_access = {name: name in sys.modules for name in watched}
print(json.dumps({"after_import": after_import, "after_access": after_access}))
"""
    env = dict(os.environ)
    src_path = str(REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path
        if not existing_pythonpath
        else os.pathsep.join((src_path, existing_pythonpath))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            module_name,
            "__none__" if attribute_name is None else attribute_name,
            *watched_modules,
        ],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
    )
    return json.loads(completed.stdout)
