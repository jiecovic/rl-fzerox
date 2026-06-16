# tests/core/career_mode/test_policy_resolver.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from rl_fzerox.core.career_mode.runner import policy_resolver as resolver_module
from rl_fzerox.core.career_mode.runner.policy_resolver import CareerPolicyResolver
from rl_fzerox.core.manager import default_managed_run_config
from rl_fzerox.core.manager.models import ManagedRun, ManagedSaveCourseSetup
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig


def test_career_policy_resolver_preloads_and_refreshes_cached_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs = {
        "run-a": _managed_run("run-a", tmp_path),
        "run-b": _managed_run("run-b", tmp_path),
    }
    loaded: list[tuple[str, str]] = []
    runners: dict[tuple[str, str], _PolicyRunnerStub] = {}

    def fake_load_policy_runner(
        run_dir: Path,
        *,
        artifact: str,
        device: str,
        algorithm: str,
    ) -> _PolicyRunnerStub:
        key = (run_dir.name, artifact)
        loaded.append(key)
        runner = _PolicyRunnerStub()
        runners[key] = runner
        return runner

    monkeypatch.setattr(resolver_module, "load_policy_runner", fake_load_policy_runner)

    resolver = CareerPolicyResolver(
        store=_PolicyRunStore(runs),
        setup=_race_setup(),
        course_setups=(
            _course_setup("run-a", course_id="mute_city", artifact="latest"),
            _course_setup("run-a", course_id="silence", artifact="latest"),
            _course_setup("run-b", course_id="sand_ocean", artifact="best"),
        ),
        device="cpu",
    )

    assert loaded == [("run-a", "latest"), ("run-b", "best")]
    loaded.clear()

    resolution = resolver.resolve({"course_index": 0})

    assert loaded == []
    assert resolution is not None
    assert resolution.control.runner is runners[("run-a", "latest")]
    assert runners[("run-a", "latest")].refresh_count == 0

    refreshed_resolution = resolver.resolve({"course_index": 0}, refresh_artifact=True)

    assert loaded == []
    assert refreshed_resolution is not None
    assert refreshed_resolution.control.runner is runners[("run-a", "latest")]
    assert runners[("run-a", "latest")].refresh_count == 1


class _PolicyRunnerStub:
    def __init__(self) -> None:
        self.refresh_count = 0

    def refresh(self) -> None:
        self.refresh_count += 1


class _PolicyRunStore:
    def __init__(self, runs: dict[str, ManagedRun]) -> None:
        self._runs = runs

    def get_run(self, run_id: str) -> ManagedRun | None:
        return self._runs.get(run_id)


def _managed_run(run_id: str, tmp_path: Path) -> ManagedRun:
    return ManagedRun(
        id=run_id,
        name=run_id,
        status="finished",
        config=default_managed_run_config(),
        config_hash=f"{run_id}-hash",
        run_dir=tmp_path / run_id,
        created_at="2026-01-01T00:00:00Z",
        lineage_id=run_id,
    )


def _race_setup() -> CareerModeRaceSetupConfig:
    return CareerModeRaceSetupConfig(
        difficulty="novice",
        cup_id="jack",
        course_id=None,
        vehicle_id="blue_falcon",
        vehicle_display_name="Blue Falcon",
        character_index=0,
        machine_select_slot=0,
        machine_select_row=0,
        machine_select_column=0,
        engine_setting_raw_value=50,
    )


def _course_setup(
    run_id: str,
    *,
    course_id: str,
    artifact: Literal["latest", "best"],
) -> ManagedSaveCourseSetup:
    return ManagedSaveCourseSetup(
        id=f"{run_id}-{course_id}-{artifact}",
        save_game_id="save",
        policy_run_id=run_id,
        policy_artifact=artifact,
        engine_setting_raw_value=50,
        difficulty="novice",
        cup_id="jack",
        course_id=course_id,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
