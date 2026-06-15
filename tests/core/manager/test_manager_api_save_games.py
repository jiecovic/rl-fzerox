# tests/core/manager/test_manager_api_save_games.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.apps.run_manager.api import create_manager_api_app
from rl_fzerox.core.engine_tuning import (
    ENGINE_TUNING_STATE_VERSION,
    EngineTuningCandidateState,
    EngineTuningContext,
    EngineTuningRuntimeState,
    save_engine_tuning_runtime_state,
)
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session.artifacts import engine_tuning_checkpoint_path
from tests.core.manager.manager_api_support import _ApiClient

pytestmark = pytest.mark.anyio


async def test_save_engine_tuning_import_uses_fresh_fork_source_snapshot(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    parent = store.create_run(
        run_id="parent-run",
        name="Parent",
        config=_adaptive_bandit_config(),
        explicit_run_dir=tmp_path / "runs" / "parent-run",
    )
    source_snapshot_dir = tmp_path / "runs" / "child-run" / "fork-source"
    source_policy_path = source_snapshot_dir / RUN_LAYOUT.policy_artifacts.latest
    source_policy_path.parent.mkdir(parents=True, exist_ok=True)
    source_policy_path.write_bytes(b"policy")
    _write_engine_tuning_state(engine_tuning_checkpoint_path(source_policy_path))
    child = store.create_run(
        run_id="child-run",
        name="Child",
        config=_adaptive_bandit_config(),
        explicit_run_dir=tmp_path / "runs" / "child-run",
        source_run_id=parent.id,
        source_artifact="latest",
        source_snapshot_dir=source_snapshot_dir,
        source_num_timesteps=123,
    )
    save_game = store.create_save_game(
        name="Career",
        save_games_root=tmp_path / "save-games",
    )
    client = _ApiClient(create_manager_api_app(store))

    response = await client.post(
        f"/api/save-games/{save_game.id}/course-setups/import-engine-tuning",
        json={
            "policy_run_id": child.id,
            "policy_artifact": "latest",
            "course_setups": [
                {
                    "cup_id": "jack",
                    "course_id": "mute_city",
                    "vehicle_id": "blue_falcon",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["recommendations"] == [
        {
            "difficulty": None,
            "cup_id": "jack",
            "course_id": "mute_city",
            "vehicle_id": "blue_falcon",
            "engine_setting_raw_value": 70,
            "mean_score": -80.0,
            "finish_count": 2,
        }
    ]


def _adaptive_bandit_config() -> ManagedRunConfig:
    config = default_managed_run_config()
    return config.model_copy(
        update={
            "vehicle": config.vehicle.model_copy(
                update={
                    "engine_mode": "adaptive_tuner",
                    "engine_setting_min_raw_value": 50,
                    "engine_setting_max_raw_value": 70,
                    "adaptive_engine_tuner_backend": "bandit",
                    "adaptive_engine_bandit_bucket_size": 10,
                }
            )
        }
    )


def _write_engine_tuning_state(state_path: Path) -> None:
    context = EngineTuningContext(course_key="mute_city", vehicle_id="blue_falcon")
    save_engine_tuning_runtime_state(
        state_path,
        EngineTuningRuntimeState(
            version=ENGINE_TUNING_STATE_VERSION,
            update_count=2,
            candidates=(
                EngineTuningCandidateState(
                    context_key=context.key,
                    course_key=context.course_key,
                    vehicle_id=context.vehicle_id,
                    engine_setting_raw_value=50,
                    finish_count=2,
                    decayed_count=2.0,
                    decayed_score_total=-200.0,
                    score_total=-200.0,
                    best_score=-90.0,
                    best_time_ms=90_000,
                ),
                EngineTuningCandidateState(
                    context_key=context.key,
                    course_key=context.course_key,
                    vehicle_id=context.vehicle_id,
                    engine_setting_raw_value=70,
                    finish_count=2,
                    decayed_count=2.0,
                    decayed_score_total=-160.0,
                    score_total=-160.0,
                    best_score=-75.0,
                    best_time_ms=75_000,
                ),
            ),
        ),
    )
