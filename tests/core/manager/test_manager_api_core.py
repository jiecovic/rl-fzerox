# tests/core/manager/test_manager_api_core.py
from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

import pytest

import rl_fzerox.apps.run_manager.api.handlers.save_games as manager_api_save_games
from rl_fzerox.apps.run_manager.api.contracts import WatchRenderer
from rl_fzerox.apps.run_manager.api.routes import _run_sync
from rl_fzerox.core.career_mode.progress import default_unlock_targets
from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
from rl_fzerox.core.manager import (
    ManagerStore,
    default_managed_run_config,
)
from tests.core.manager.manager_api_support import (
    _ApiClient,
    _client,
    _LauncherStub,
    _write_policy_artifact,
)

pytestmark = pytest.mark.anyio


async def test_run_sync_uses_worker_thread() -> None:
    caller_thread = threading.get_ident()

    worker_thread = await _run_sync(threading.get_ident)

    assert worker_thread != caller_thread


async def test_manager_api_lists_default_template(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = await client.get("/api/templates")

    assert response.status_code == 200
    payload = response.json()
    assert payload["templates"][0]["id"] == "all_cups_recurrent_ppo"


async def test_manager_api_creates_draft(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")

    response = await client.post("/api/drafts", json={"name": "Draft", "config": config})

    assert response.status_code == 201
    payload = response.json()
    assert payload["draft"]["name"] == "Draft"


async def test_manager_api_creates_save_game(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = await client.post("/api/save-games", json={"name": "Unlock Run"})

    assert response.status_code == 201
    payload = response.json()
    assert payload["save_game"]["name"] == "Unlock Run"
    assert "seed" not in payload["save_game"]
    assert payload["save_game"]["status"] == "created"
    assert payload["save_game"]["runner_settings"] == {
        "attempt_seed": None,
        "device": "cuda",
        "policy_mode": "deterministic",
        "recording_enabled": False,
        "recording_input_hud_enabled": False,
        "recording_upscale_factor": 2,
        "recording_path": None,
        "renderer": "gliden64",
    }
    assert payload["save_game"]["unlock_progress"]["completed_count"] == 0
    assert payload["save_game"]["unlock_progress"]["total_count"] == len(default_unlock_targets())
    assert payload["save_game"]["unlock_progress"]["next_target"]["difficulty"] == "novice"
    assert payload["save_game"]["unlock_progress"]["next_target"]["cup_id"] == "jack"
    assert payload["save_game"]["attempts"] == []
    assert payload["save_game"]["course_setups"] == []
    assert payload["save_game"]["cup_setups"] == []

    list_response = await client.get("/api/save-games")

    assert list_response.status_code == 200
    assert list_response.json()["save_games"] == [payload["save_game"]]


async def test_manager_api_returns_slim_save_game_status(tmp_path: Path) -> None:
    client = _client(tmp_path)
    create_response = await client.post("/api/save-games", json={"name": "Unlock Run"})
    save_game_id = create_response.json()["save_game"]["id"]

    response = await client.get(f"/api/save-games/{save_game_id}/status")

    assert response.status_code == 200
    save_payload = response.json()["save_game"]
    assert save_payload["id"] == save_game_id
    assert save_payload["name"] == "Unlock Run"
    assert save_payload["status"] == "created"
    assert save_payload["runner_settings"] == {
        "attempt_seed": None,
        "device": "cuda",
        "policy_mode": "deterministic",
        "recording_enabled": False,
        "recording_input_hud_enabled": False,
        "recording_upscale_factor": 2,
        "recording_path": None,
        "renderer": "gliden64",
    }
    assert save_payload["unlock_progress"]["completed_count"] == 0
    assert "attempts" not in save_payload
    assert "course_setups" not in save_payload
    assert "cup_setups" not in save_payload


async def test_manager_api_updates_save_game_runner_settings(tmp_path: Path) -> None:
    client = _client(tmp_path)
    create_response = await client.post("/api/save-games", json={"name": "Unlock Run"})
    save_game_id = create_response.json()["save_game"]["id"]

    response = await client.put(
        f"/api/save-games/{save_game_id}/runner-settings",
        json={
            "attempt_seed": 12345,
            "device": "cpu",
            "policy_mode": "stochastic",
            "recording_enabled": True,
            "recording_input_hud_enabled": True,
            "recording_upscale_factor": 2,
            "recording_path": "local/recordings/career/test.mkv",
            "renderer": "angrylion",
        },
    )

    assert response.status_code == 200
    payload = response.json()["save_game"]
    assert payload["runner_settings"] == {
        "attempt_seed": 12345,
        "device": "cpu",
        "policy_mode": "stochastic",
        "recording_enabled": True,
        "recording_input_hud_enabled": True,
        "recording_upscale_factor": 2,
        "recording_path": "local/recordings/career/test.mkv",
        "renderer": "angrylion",
    }


async def test_manager_api_deletes_save_game(tmp_path: Path) -> None:
    client = _client(tmp_path)
    create_response = await client.post("/api/save-games", json={"name": "Delete Save"})
    save_game_id = create_response.json()["save_game"]["id"]

    response = await client.delete(f"/api/save-games/{save_game_id}")
    list_response = await client.get("/api/save-games")

    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert list_response.json()["save_games"] == []


async def test_manager_api_resets_stale_unstarted_save_game_status(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(name="Unlock Run", save_games_root=tmp_path / "saves")
    updated = store.update_save_game_status(save_game_id=save_game.id, status="paused")
    assert updated is not None
    client = _client(tmp_path, store=store)

    response = await client.get("/api/save-games")

    assert response.status_code == 200
    payload = response.json()
    assert payload["save_games"][0]["id"] == save_game.id
    assert payload["save_games"][0]["status"] == "created"
    assert payload["save_games"][0]["attempts"] == []


async def test_manager_api_marks_orphaned_running_save_attempt_failed(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(name="Unlock Run", save_games_root=tmp_path / "saves")
    attempt = store.start_save_attempt(
        save_game_id=save_game.id,
        target_kind="clear_gp_cup",
        difficulty="novice",
        cup_id="jack",
    )
    updated = store.update_save_game_status(save_game_id=save_game.id, status="running")
    assert updated is not None
    client = _client(tmp_path, store=store)

    response = await client.get("/api/save-games")

    assert response.status_code == 200
    payload = response.json()
    save_payload = payload["save_games"][0]
    assert save_payload["id"] == save_game.id
    assert save_payload["status"] == "created"
    assert save_payload["attempts"][0]["id"] == attempt.id
    assert save_payload["attempts"][0]["status"] == "failed"
    assert save_payload["attempts"][0]["failure_reason"] == "career mode runner process disappeared"


async def test_manager_api_ignores_old_race_truncation_artifacts_for_start_state(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(name="Unlock Run", save_games_root=tmp_path / "saves")
    attempt = store.start_save_attempt(
        save_game_id=save_game.id,
        target_kind="clear_gp_cup",
        difficulty="novice",
        cup_id="jack",
    )
    finished = store.finish_save_attempt(
        attempt_id=attempt.id,
        status="failed",
        finish_position=1,
        failure_reason="race truncated",
    )
    assert finished is not None
    updated = store.update_save_game_status(save_game_id=save_game.id, status="paused")
    assert updated is not None
    client = _client(tmp_path, store=store)

    response = await client.get("/api/save-games")

    assert response.status_code == 200
    payload = response.json()
    assert payload["save_games"][0]["id"] == save_game.id
    assert payload["save_games"][0]["status"] == "created"
    assert payload["save_games"][0]["attempts"][0]["failure_reason"] == "race truncated"


async def test_manager_api_upserts_save_course_setup(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="policy-run",
        name="Policy Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    client = _client(tmp_path, store=store)
    create_response = await client.post("/api/save-games", json={"name": "Unlock Run"})
    save_game_id = create_response.json()["save_game"]["id"]

    response = await client.put(
        f"/api/save-games/{save_game_id}/course-setups",
        json={
            "cup_id": "jack",
            "course_id": "mute_city",
            "policy_artifact": "best",
            "policy_run_id": run.id,
            "engine_setting_raw_value": 60,
        },
    )

    assert response.status_code == 200
    save_game = response.json()["save_game"]
    assignments = save_game["course_setups"]
    assert len(assignments) == 1
    assert assignments[0]["save_game_id"] == save_game_id
    assert assignments[0]["cup_id"] == "jack"
    assert assignments[0]["course_id"] == "mute_city"
    assert assignments[0]["policy_run_id"] == "policy-run"
    assert assignments[0]["policy_artifact"] == "best"
    assert assignments[0]["engine_setting_raw_value"] == 60

    cup_response = await client.put(
        f"/api/save-games/{save_game_id}/cup-setups",
        json={
            "cup_id": "jack",
            "vehicle_id": "blue_falcon",
        },
    )

    assert cup_response.status_code == 200
    cup_setups = cup_response.json()["save_game"]["cup_setups"]
    assert len(cup_setups) == 1
    assert cup_setups[0]["cup_id"] == "jack"
    assert cup_setups[0]["vehicle_id"] == "blue_falcon"


async def test_manager_api_starts_next_save_attempt(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="policy-run",
        name="Policy Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    client = _client(tmp_path, store=store)
    create_response = await client.post("/api/save-games", json={"name": "Unlock Run"})
    save_game_id = create_response.json()["save_game"]["id"]
    await _configure_api_gp_cup(client, save_game_id=save_game_id, run_id=run.id, cup_id="jack")

    response = await client.post(f"/api/save-games/{save_game_id}/attempts/next")

    assert response.status_code == 200
    save_game = response.json()["save_game"]
    assert len(save_game["attempts"]) == 1
    assert save_game["attempts"][0]["save_game_id"] == save_game_id
    assert save_game["attempts"][0]["target_kind"] == "clear_gp_cup"
    assert save_game["attempts"][0]["difficulty"] == "novice"
    assert save_game["attempts"][0]["cup_id"] == "jack"

    repeated_response = await client.post(f"/api/save-games/{save_game_id}/attempts/next")

    assert repeated_response.status_code == 400
    assert repeated_response.json()["error"] == "save game already has a running attempt"


async def test_manager_api_starts_career_mode_for_selected_target(tmp_path: Path) -> None:
    class _RecordingLauncher(_LauncherStub):
        request: dict[str, object] | None = None

        def start_career_mode(
            self,
            *,
            save_game_id: str,
            device: Literal["cpu", "cuda"],
            renderer: WatchRenderer | None,
            attempt_seed: int | None,
            deterministic_policy: bool,
            recording_enabled: bool,
            recording_input_hud_enabled: bool,
            recording_upscale_factor: int,
            recording_path: Path | None,
            target_kind: str | None,
            difficulty: str | None,
            cup_id: str | None,
            course_id: str | None,
            single_target: bool,
        ) -> Literal["started", "already_running"]:
            self.request = {
                "save_game_id": save_game_id,
                "device": device,
                "renderer": renderer,
                "attempt_seed": attempt_seed,
                "deterministic_policy": deterministic_policy,
                "recording_enabled": recording_enabled,
                "recording_input_hud_enabled": recording_input_hud_enabled,
                "recording_upscale_factor": recording_upscale_factor,
                "recording_path": recording_path,
                "target_kind": target_kind,
                "difficulty": difficulty,
                "cup_id": cup_id,
                "course_id": course_id,
                "single_target": single_target,
            }
            return "started"

    launcher = _RecordingLauncher()
    client = _client(tmp_path, launcher=launcher)

    response = await client.post(
        "/api/save-games/save-001/runner",
        json={
            "device": "cpu",
            "renderer": "gliden64",
            "attempt_seed": 42,
            "policy_mode": "stochastic",
            "recording_enabled": True,
            "recording_input_hud_enabled": True,
            "recording_upscale_factor": 3,
            "recording_path": None,
            "target_kind": "clear_gp_cup",
            "difficulty": "novice",
            "cup_id": "queen",
            "course_id": None,
            "single_target": True,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "started"}
    assert launcher.request == {
        "save_game_id": "save-001",
        "device": "cpu",
        "renderer": "gliden64",
        "attempt_seed": 42,
        "deterministic_policy": False,
        "recording_enabled": True,
        "recording_input_hud_enabled": True,
        "recording_upscale_factor": 3,
        "recording_path": None,
        "target_kind": "clear_gp_cup",
        "difficulty": "novice",
        "cup_id": "queen",
        "course_id": None,
        "single_target": True,
    }


async def test_manager_api_returns_save_attempt_execution_context(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="policy-run",
        name="Policy Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    policy_path = _write_policy_artifact(run.run_dir, "best")
    client = _client(tmp_path, store=store)
    create_response = await client.post("/api/save-games", json={"name": "Unlock Run"})
    save_game_id = create_response.json()["save_game"]["id"]
    await _configure_api_gp_cup(client, save_game_id=save_game_id, run_id=run.id, cup_id="jack")
    attempt_response = await client.post(f"/api/save-games/{save_game_id}/attempts/next")
    attempt = attempt_response.json()["save_game"]["attempts"][0]

    response = await client.get(f"/api/save-attempts/{attempt['id']}/execution-context")

    assert response.status_code == 200
    context = response.json()["execution_context"]
    assert context["attempt"]["id"] == attempt["id"]
    assert context["target"] == {
        "kind": "clear_gp_cup",
        "label": "Clear Novice Jack Cup",
        "difficulty": "novice",
        "cup_id": "jack",
        "course_id": None,
    }
    assert context["policy_run"]["id"] == "policy-run"
    assert context["policy_artifact"] == "best"
    assert context["policy_path"] == str(policy_path.resolve())

    plan_response = await client.get(f"/api/save-attempts/{attempt['id']}/execution-plan")

    assert plan_response.status_code == 200
    plan = plan_response.json()["execution_plan"]
    assert plan["attempt"]["id"] == attempt["id"]
    assert plan["policy"]["run_id"] == "policy-run"
    assert plan["policy"]["artifact"] == "best"
    assert plan["policy"]["path"] == str(policy_path.resolve())
    assert plan["race_setup"]["difficulty"] == "novice"
    assert plan["race_setup"]["cup_id"] == "jack"
    assert plan["race_setup"]["vehicle_id"] == "blue_falcon"


async def _configure_api_gp_cup(
    client: _ApiClient,
    *,
    save_game_id: str,
    run_id: str,
    cup_id: str,
) -> None:
    cup_response = await client.put(
        f"/api/save-games/{save_game_id}/cup-setups",
        json={
            "cup_id": cup_id,
            "vehicle_id": "blue_falcon",
        },
    )
    assert cup_response.status_code == 200
    for course in sorted(BUILT_IN_COURSES, key=lambda item: item.course_index):
        if course.cup != cup_id:
            continue
        response = await client.put(
            f"/api/save-games/{save_game_id}/course-setups",
            json={
                "cup_id": cup_id,
                "course_id": course.id,
                "policy_artifact": "best",
                "policy_run_id": run_id,
            },
        )
        assert response.status_code == 200


async def test_manager_api_rejects_duplicate_save_game_name(tmp_path: Path) -> None:
    client = _client(tmp_path)
    first_response = await client.post("/api/save-games", json={"name": "Unlock Run"})
    assert first_response.status_code == 201

    response = await client.post("/api/save-games", json={"name": "unlock run"})

    assert response.status_code == 409
    assert response.json()["error"] == "name already exists: unlock run"


async def test_manager_api_opens_save_game_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(tmp_path)
    create_response = await client.post("/api/save-games", json={"name": "Unlock Run"})
    payload = create_response.json()["save_game"]
    opened_paths: list[Path] = []

    monkeypatch.setattr(manager_api_save_games, "open_directory", opened_paths.append)

    response = await client.post(f"/api/save-games/{payload['id']}/open-dir")

    assert response.status_code == 200
    assert response.json() == {"opened": True}
    assert opened_paths == [Path(payload["save_path"]).parent]
