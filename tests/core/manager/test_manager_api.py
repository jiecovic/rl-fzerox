# tests/core/manager/test_manager_api.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient

import rl_fzerox.apps.run_manager.api as manager_api
from rl_fzerox.apps.run_manager.api import create_manager_api_app
from rl_fzerox.core.manager import (
    ManagedRun,
    ManagedRunConfig,
    ManagerStore,
    default_managed_run_config,
)


class _LauncherStub:
    def fork(
        self,
        *,
        run_id: str,
        artifact: Literal["latest", "best"],
        name: str | None,
        config: ManagedRunConfig | None,
    ) -> ManagedRun:
        del run_id, artifact, name, config
        raise AssertionError("fork should not be called")

    def request_pause(self, *, run_id: str) -> ManagedRun:
        del run_id
        raise AssertionError("pause should not be called")

    def request_stop(self, *, run_id: str) -> ManagedRun:
        del run_id
        raise AssertionError("stop should not be called")

    def resume(self, *, run_id: str) -> ManagedRun:
        del run_id
        raise AssertionError("resume should not be called")

    def watch_artifact(self, *, run_id: str, artifact: str) -> None:
        del run_id, artifact
        raise AssertionError("watch should not be called")


def test_manager_api_lists_default_template(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/templates")

    assert response.status_code == 200
    payload = response.json()
    assert payload["templates"][0]["id"] == "all_cups_recurrent_ppo"


def test_manager_api_creates_draft(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")

    response = client.post("/api/drafts", json={"name": "Draft", "config": config})

    assert response.status_code == 201
    payload = response.json()
    assert payload["draft"]["name"] == "Draft"


def test_manager_api_launches_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")

    class FakeLauncher(_LauncherStub):
        def launch(
            self,
            *,
            name: str,
            config: ManagedRunConfig,
            draft_id: str | None,
            source_run_id: str | None,
            source_artifact: Literal["latest", "best"] | None,
        ):
            del draft_id, source_artifact, source_run_id
            run = store.create_run(
                name=name,
                config=config,
                managed_runs_root=tmp_path / "runs",
            )
            launched = store.update_run_status(
                run_id=run.id,
                status="running",
                started_at="2026-05-03T12:00:00+00:00",
                stopped_at=None,
                message="worker launched",
            )
            if launched is None:
                raise RuntimeError("launch status update failed")
            return launched

    client = TestClient(create_manager_api_app(store, run_launcher=FakeLauncher()))
    config = default_managed_run_config().model_dump(mode="json")

    response = client.post("/api/runs", json={"name": "Launch Me", "config": config})

    assert response.status_code == 201
    payload = response.json()
    assert payload["run"]["name"] == "Launch Me"
    assert payload["run"]["status"] == "running"


def test_manager_api_launch_allows_same_name_as_source_draft(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")

    class FakeLauncher(_LauncherStub):
        def launch(
            self,
            *,
            name: str,
            config: ManagedRunConfig,
            draft_id: str | None,
            source_run_id: str | None,
            source_artifact: Literal["latest", "best"] | None,
        ):
            del source_artifact, source_run_id
            run = store.create_run(
                name=name,
                config=config,
                managed_runs_root=tmp_path / "runs",
                exclude_draft_id=draft_id,
            )
            launched = store.update_run_status(
                run_id=run.id,
                status="running",
                started_at="2026-05-03T12:00:00+00:00",
                stopped_at=None,
                message="worker launched",
            )
            if launched is None:
                raise RuntimeError("launch status update failed")
            return launched

    client = TestClient(create_manager_api_app(store, run_launcher=FakeLauncher()))
    config = default_managed_run_config().model_dump(mode="json")
    draft_response = client.post("/api/drafts", json={"name": "Shared", "config": config})
    draft_id = draft_response.json()["draft"]["id"]

    response = client.post(
        "/api/runs",
        json={"name": "Shared", "config": config, "draft_id": draft_id},
    )

    assert response.status_code == 201
    assert response.json()["run"]["name"] == "Shared"


def test_manager_api_forks_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    parent = store.create_run(
        run_id="parent-run",
        name="Parent Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "parent-run",
    )

    class FakeLauncher(_LauncherStub):
        def launch(
            self,
            *,
            name: str,
            config: ManagedRunConfig,
            draft_id: str | None,
            source_run_id: str | None,
            source_artifact: Literal["latest", "best"] | None,
        ):
            del config, draft_id, name, source_artifact, source_run_id
            raise AssertionError("launch should not be called")

        def fork(
            self,
            *,
            run_id: str,
            artifact: str,
            name: str | None,
            config: ManagedRunConfig | None,
        ):
            del config
            assert run_id == parent.id
            assert artifact == "best"
            child = store.create_run(
                run_id="child-run",
                name=name or "Parent Run best fork",
                config=default_managed_run_config(),
                explicit_run_dir=tmp_path / "runs" / "child-run",
                lineage_step_offset=816_040,
                parent_run_id=parent.id,
                source_run_id=parent.id,
                source_artifact=artifact,
                source_num_timesteps=816_040,
            )
            launched = store.update_run_status(
                run_id=child.id,
                status="running",
                started_at="2026-05-04T12:00:00+00:00",
                stopped_at=None,
                message="forked worker launched",
            )
            if launched is None:
                raise RuntimeError("fork status update failed")
            return launched

    client = TestClient(create_manager_api_app(store, run_launcher=FakeLauncher()))

    response = client.post(f"/api/runs/{parent.id}/fork", json={"artifact": "best"})

    assert response.status_code == 201
    payload = response.json()
    assert payload["run"]["parent_run_id"] == parent.id
    assert payload["run"]["source_run_id"] == parent.id
    assert payload["run"]["source_artifact"] == "best"
    assert payload["run"]["source_num_timesteps"] == 816040


def test_manager_api_reads_track_sampling_runtime_state(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="run-with-track-pool",
        name="Track Pool Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "run-with-track-pool",
    )
    store.update_run_status(
        run_id=run.id,
        status="running",
        started_at="2026-05-04T12:00:00+00:00",
        stopped_at=None,
        message="worker launched",
    )
    state_path = run.run_dir / "runtime" / "track_sampling_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
                {
                    "version": 1,
                    "sampling_mode": "step_balanced",
                    "action_repeat": 2,
                    "update_episodes": 4,
                    "ema_alpha": 0.5,
                    "max_weight_scale": 5.0,
                    "update_count": 3,
                    "episodes_since_update": 1,
                    "entries": [
                    {
                        "track_id": "mute",
                        "course_key": "mute_city",
                        "label": "Mute City",
                        "base_weight": 1.0,
                        "current_weight": 1.5,
                        "completed_frames": 1200,
                        "episode_count": 3,
                        "ema_episode_frames": 400.0,
                    },
                    {
                        "track_id": "silence",
                        "course_key": "silence",
                        "label": "Silence",
                        "base_weight": 1.0,
                        "current_weight": 0.5,
                        "completed_frames": 800,
                        "episode_count": 1,
                        "ema_episode_frames": 800.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    client = _client(tmp_path, store=store)

    response = client.get(f"/api/runs/{run.id}/track-sampling")

    assert response.status_code == 200
    payload = response.json()["state"]
    assert payload["update_episodes"] == 4
    assert payload["entries"][0]["label"] == "Mute City"
    assert payload["entries"][0]["current_probability"] == pytest.approx(0.75)
    assert payload["entries"][0]["completed_env_steps"] == 600
    assert payload["entries"][0]["step_share"] == pytest.approx(0.6)


def test_manager_api_deletes_lineage(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="root-run",
        name="Root Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    store.create_run(
        run_id="leaf-run",
        name="Leaf Run",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
        lineage_id=run.lineage_id,
        parent_run_id=run.id,
        source_run_id=run.id,
        source_artifact="latest",
        source_num_timesteps=111,
    )

    client = _client(tmp_path)
    response = client.delete(f"/api/lineages/{run.lineage_id}")

    assert response.status_code == 200
    assert response.json() == {"deleted": True}


def test_manager_api_metrics_full_mode_disables_recent_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = store.create_run(
        run_id="run-001",
        name="Metrics Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "run-001",
    )
    seen_limits: list[int | None] = []

    def fake_loader(
        managed_run: ManagedRun,
        *,
        limit: int | None,
    ) -> tuple[object, ...]:
        assert managed_run.id == run.id
        seen_limits.append(limit)
        return ()

    monkeypatch.setattr(manager_api, "load_run_metric_samples_from_tensorboard", fake_loader)

    class FakeLauncher(_LauncherStub):
        def launch(
            self,
            *,
            name: str,
            config: ManagedRunConfig,
            draft_id: str | None,
            source_run_id: str | None,
            source_artifact: Literal["latest", "best"] | None,
        ) -> ManagedRun:
            del name, config, draft_id, source_run_id, source_artifact
            raise AssertionError("launch should not be called")

    client = TestClient(create_manager_api_app(store, run_launcher=FakeLauncher()))

    recent_response = client.get(f"/api/runs/{run.id}/metrics")
    full_response = client.get(f"/api/runs/{run.id}/metrics?mode=full")

    assert recent_response.status_code == 200
    assert full_response.status_code == 200
    assert seen_limits == [240, None]


def test_manager_api_updates_draft(tmp_path: Path) -> None:
    client = _client(tmp_path)
    create_response = client.post(
        "/api/drafts",
        json={"name": "Draft", "config": default_managed_run_config().model_dump(mode="json")},
    )
    draft_id = create_response.json()["draft"]["id"]
    updated_config = default_managed_run_config().model_dump(mode="json")
    updated_config["seed"] = 999

    response = client.put(
        f"/api/drafts/{draft_id}",
        json={"name": "Updated draft", "config": updated_config},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["draft"]["name"] == "Updated draft"
    assert payload["draft"]["config"]["seed"] == 999


def test_manager_api_rejects_duplicate_draft_name(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    first_response = client.post("/api/drafts", json={"name": "Draft", "config": config})

    response = client.post("/api/drafts", json={"name": "draft", "config": config})

    assert first_response.status_code == 201
    assert response.status_code == 409
    assert response.json()["error"] == "name already exists: draft"


def test_manager_api_rejects_renaming_draft_to_existing_name(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    first_draft_id = client.post("/api/drafts", json={"name": "Alpha", "config": config}).json()[
        "draft"
    ]["id"]
    client.post("/api/drafts", json={"name": "Beta", "config": config})

    response = client.put(
        f"/api/drafts/{first_draft_id}",
        json={"name": "beta", "config": config},
    )

    assert response.status_code == 409
    assert response.json()["error"] == "name already exists: beta"


def test_manager_api_allows_draft_name_matching_existing_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.create_run(
        name="Shared Name",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    client = TestClient(create_manager_api_app(store))
    config = default_managed_run_config().model_dump(mode="json")

    response = client.post("/api/drafts", json={"name": "Shared Name", "config": config})

    assert response.status_code == 201
    assert response.json()["draft"]["name"] == "Shared Name"


def test_manager_api_rejects_invalid_json(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/drafts",
        content="{",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert "error" in response.json()


def test_manager_api_hides_unstarted_run_records(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.create_run(
        name="Created Only",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "runs",
    )
    client = TestClient(create_manager_api_app(store))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert response.json() == {"runs": []}


def test_manager_api_exposes_config_metadata(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/config-metadata")

    assert response.status_code == 200
    payload = response.json()
    assert "crop_60x76" in {
        preset["value"] for preset in payload["observation_presets"]
    }
    assert "nature_32_64_128" in {
        profile["value"] for profile in payload["conv_profiles"]
    }
    assert "custom" in {profile["value"] for profile in payload["conv_profiles"]}
    assert "x_cup" in {mode["value"] for mode in payload["track_pool_modes"]}
    assert "time_attack" in {mode["value"] for mode in payload["race_modes"]}
    assert "step_balanced" in {
        mode["value"] for mode in payload["track_sampling_modes"]
    }
    assert "jack" in {cup["id"] for cup in payload["track_cups"]}
    assert "mute_city" in {course["id"] for course in payload["built_in_courses"]}
    assert "blue_falcon" in {vehicle["id"] for vehicle in payload["vehicles"]}
    blue_falcon = next(
        vehicle for vehicle in payload["vehicles"] if vehicle["id"] == "blue_falcon"
    )
    assert blue_falcon["menu_row"] == 0
    assert blue_falcon["menu_column"] == 0
    red_gazelle = next(
        vehicle for vehicle in payload["vehicles"] if vehicle["id"] == "red_gazelle"
    )
    assert red_gazelle["menu_row"] == 0
    assert red_gazelle["menu_column"] == 5
    assert "balanced" in {
        preset["id"] for preset in payload["engine_setting_presets"]
    }
    assert "continuous" in {mode["value"] for mode in payload["steering_modes"]}
    assert "on_off" in {mode["value"] for mode in payload["drive_modes"]}
    assert "independent_buttons" in {
        mode["value"] for mode in payload["lean_output_modes"]
    }
    lean_modes = {mode["value"] for mode in payload["lean_modes"]}
    assert "release_cooldown" in lean_modes
    assert "raw" in lean_modes


def test_manager_api_accepts_frontend_action_layouts_even_if_runtime_support_lags(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["action"]["steering_mode"] = "discrete"
    config["action"]["drive_mode"] = "pwm"
    config["action"]["include_air_brake"] = False
    config["action"]["include_pitch"] = False

    response = client.post("/api/drafts", json={"name": "Draft", "config": config})

    assert response.status_code == 201


def test_manager_api_previews_policy_architecture(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")

    response = client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_shape"] == {"height": 60, "width": 76, "channels": 6}
    assert payload["total_params"] > 0
    assert payload["continuous_action_dims"] == 1
    assert payload["discrete_action_logits"] == 14
    assert payload["architecture_lanes"][0]["label"] == "Image branch"
    cnn_node = next(
        node for node in payload["architecture_lanes"][0]["nodes"] if node["id"] == "cnn"
    )
    assert cnn_node["params"] > 0
    fusion_nodes = payload["architecture_lanes"][2]["nodes"]
    assert {node["id"] for node in fusion_nodes} >= {
        "action_net",
        "policy_head",
        "value_head",
        "value_net",
    }
    action_branch_names = {branch["name"] for branch in payload["action_branches"]}
    assert action_branch_names == {
        "steer",
        "throttle",
        "air_brake",
        "boost",
        "lean",
        "pitch",
    }


def test_manager_api_previews_raw_state_fusion_without_state_mlp(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["policy"]["state_net_arch"] = []

    response = client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["state_features_dim"] == payload["state_dim"]
    state_nodes = payload["architecture_lanes"][1]["nodes"]
    state_mlp = next(node for node in state_nodes if node["id"] == "state_mlp")
    assert state_mlp["tone"] == "muted"


def test_manager_api_previews_custom_cnn_profile(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["policy"]["conv_profile"] = "custom"
    config["policy"]["custom_conv_layers"] = [
        {"out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1},
        {"out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 0},
        {"out_channels": 48, "kernel_size": 3, "stride": 1, "padding": 1},
    ]

    response = client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert [layer["out_channels"] for layer in payload["conv_layers"]] == [16, 32, 48]
    assert [layer["padding"] for layer in payload["conv_layers"]] == [1, 0, 1]
    assert payload["conv_layers"][0]["output_height"] == 30
    assert payload["conv_layers"][0]["output_width"] == 38


def test_manager_api_preview_keeps_masked_branch_logits_but_marks_status(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["action"]["enable_boost"] = False

    response = client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["discrete_action_logits"] == 14
    boost_branch = next(
        branch for branch in payload["action_branches"] if branch["name"] == "boost"
    )
    assert boost_branch["enabled"] is False
    assert boost_branch["mask_label"] == "masked idle"


def test_manager_api_preview_keeps_gas_head_but_marks_forced_full(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["action"]["force_full_throttle"] = True

    response = client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["continuous_action_dims"] == 1
    assert payload["discrete_action_logits"] == 14
    throttle_branch = next(
        branch for branch in payload["action_branches"] if branch["name"] == "throttle"
    )
    assert throttle_branch["enabled"] is False
    assert throttle_branch["mask_label"] == "forced engaged"


def test_manager_api_preview_removes_logits_when_branch_excluded(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["action"]["include_boost"] = False

    response = client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["discrete_action_logits"] == 12
    branch_names = {branch["name"] for branch in payload["action_branches"]}
    assert "boost" not in branch_names


def _client(tmp_path: Path, *, store: ManagerStore | None = None) -> TestClient:
    resolved_store = store or ManagerStore(tmp_path / "manager" / "runs.db")
    return TestClient(create_manager_api_app(resolved_store))
