# tests/core/manager/test_manager_api_metadata_policy.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.manager import (
    default_managed_run_config,
)
from tests.core.manager.manager_api_support import (
    _client,
)

pytestmark = pytest.mark.anyio


async def test_manager_api_exposes_config_metadata(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = await client.get("/api/config-metadata")

    assert response.status_code == 200
    payload = response.json()
    preset_values = {preset["value"] for preset in payload["observation_presets"]}
    assert preset_values == {"crop_72x96", "crop_84x84"}
    preset_labels = {preset["value"]: preset["label"] for preset in payload["observation_presets"]}
    assert preset_labels["crop_72x96"] == "72 x 96 IMPALA"
    assert preset_labels["crop_84x84"] == "84 x 84 DQN/Atari"
    assert "gliden64" in {source["renderer"] for source in payload["observation_source_geometries"]}
    assert "nature" in {profile["value"] for profile in payload["conv_profiles"]}
    assert "impala_small" in {profile["value"] for profile in payload["conv_profiles"]}
    assert "impala_large" in {profile["value"] for profile in payload["conv_profiles"]}
    assert "custom" in {profile["value"] for profile in payload["conv_profiles"]}
    assert "time_attack" in {mode["value"] for mode in payload["race_modes"]}
    assert "master" in {mode["value"] for mode in payload["gp_difficulties"]}
    assert "step_balanced" in {mode["value"] for mode in payload["track_sampling_modes"]}
    assert "adaptive_step_balanced" in {mode["value"] for mode in payload["track_sampling_modes"]}
    assert "deficit_budget" in {mode["value"] for mode in payload["track_sampling_modes"]}
    assert "jack" in {cup["id"] for cup in payload["track_cups"]}
    assert "mute_city" in {course["id"] for course in payload["built_in_courses"]}
    assert "blue_falcon" in {vehicle["id"] for vehicle in payload["vehicles"]}
    blue_falcon = next(vehicle for vehicle in payload["vehicles"] if vehicle["id"] == "blue_falcon")
    assert blue_falcon["menu_row"] == 0
    assert blue_falcon["menu_column"] == 0
    red_gazelle = next(vehicle for vehicle in payload["vehicles"] if vehicle["id"] == "red_gazelle")
    assert red_gazelle["menu_row"] == 0
    assert red_gazelle["menu_column"] == 5
    assert "engine_setting_presets" not in payload
    assert "continuous" in {mode["value"] for mode in payload["steering_modes"]}
    assert "on_off" in {mode["value"] for mode in payload["drive_modes"]}
    lean_output_modes = {mode["value"] for mode in payload["lean_output_modes"]}
    assert "four_way_categorical" in lean_output_modes
    assert "independent_buttons" in lean_output_modes
    lean_modes = {mode["value"] for mode in payload["lean_modes"]}
    assert "release_cooldown" in lean_modes
    assert "raw" in lean_modes


async def test_manager_api_accepts_frontend_action_layouts_even_if_runtime_support_lags(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["action"]["steering_mode"] = "discrete"
    config["action"]["drive_mode"] = "pwm"
    config["action"]["include_air_brake"] = False
    config["action"]["include_pitch"] = False

    response = await client.post("/api/drafts", json={"name": "Draft", "config": config})

    assert response.status_code == 201


async def test_manager_api_previews_policy_architecture(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_shape"] == {"height": 84, "width": 84, "channels": 6}
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
        "aux_head",
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


async def test_manager_api_previews_raw_state_fusion_without_state_mlp(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["policy"]["state_net_arch"] = []

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["state_features_dim"] == payload["state_dim"]
    state_nodes = payload["architecture_lanes"][1]["nodes"]
    state_mlp = next(node for node in state_nodes if node["id"] == "state_mlp")
    assert state_mlp["tone"] == "muted"


async def test_manager_api_previews_custom_cnn_profile(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["policy"]["conv_profile"] = "custom"
    config["policy"]["custom_conv_layers"] = [
        {"kind": "conv", "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1},
        {
            "kind": "residual_post",
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        },
        {"kind": "maxpool", "out_channels": 16, "kernel_size": 2, "stride": 2, "padding": 0},
        {"kind": "avgpool", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        {"kind": "conv", "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 0},
        {"kind": "conv", "out_channels": 48, "kernel_size": 3, "stride": 1, "padding": 1},
    ]

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert [layer["name"] for layer in payload["conv_layers"]] == [
        "conv1",
        "res2",
        "pool3",
        "avgpool4",
        "conv5",
        "conv6",
    ]
    assert [layer["kind"] for layer in payload["conv_layers"]] == [
        "conv",
        "residual_post",
        "maxpool",
        "avgpool",
        "conv",
        "conv",
    ]
    assert [layer["out_channels"] for layer in payload["conv_layers"]] == [16, 16, 16, 16, 32, 48]
    assert [layer["padding"] for layer in payload["conv_layers"]] == [1, 1, 0, 1, 0, 1]
    assert payload["conv_layers"][0]["output_height"] == 42
    assert payload["conv_layers"][0]["output_width"] == 42
    assert payload["conv_layers"][1]["output_height"] == 42
    assert payload["conv_layers"][1]["output_width"] == 42
    assert payload["conv_layers"][2]["output_height"] == 21
    assert payload["conv_layers"][2]["output_width"] == 21
    assert payload["conv_layers"][3]["output_height"] == 21
    assert payload["conv_layers"][3]["output_width"] == 21


async def test_manager_api_previews_impala_small_cnn_profile(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["observation"]["resolution"] = {"mode": "preset", "preset": "crop_72x96"}
    config["policy"]["conv_profile"] = "impala_small"

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    layer_shapes = [
        (layer["out_channels"], layer["output_height"], layer["output_width"])
        for layer in payload["conv_layers"]
    ]
    assert layer_shapes == [(16, 17, 23), (32, 7, 10)]
    assert payload["image_features_dim"] == 2_240


async def test_manager_api_previews_impala_large_cnn_profile(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["observation"]["resolution"] = {"mode": "preset", "preset": "crop_72x96"}
    config["policy"]["conv_profile"] = "impala_large"

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_shape"] == {"height": 72, "width": 96, "channels": 6}
    assert [layer["kind"] for layer in payload["conv_layers"]] == [
        "conv",
        "maxpool",
        "residual_pre",
        "residual_pre",
        "conv",
        "maxpool",
        "residual_pre",
        "residual_pre",
        "conv",
        "maxpool",
        "residual_pre",
        "residual_pre",
        "activation",
    ]
    assert [layer["post_activation"] for layer in payload["conv_layers"][:2]] == [False, True]
    layer_shapes = [
        (layer["out_channels"], layer["output_height"], layer["output_width"])
        for layer in payload["conv_layers"]
    ]
    pixel_drops = [
        (layer["dropped_height"], layer["dropped_width"]) for layer in payload["conv_layers"]
    ]
    assert layer_shapes == [
        (16, 72, 96),
        (16, 36, 48),
        (16, 36, 48),
        (16, 36, 48),
        (32, 36, 48),
        (32, 18, 24),
        (32, 18, 24),
        (32, 18, 24),
        (32, 18, 24),
        (32, 9, 12),
        (32, 9, 12),
        (32, 9, 12),
        (32, 9, 12),
    ]
    assert pixel_drops == [(0, 0)] * 13
    assert payload["image_features_dim"] == 3_456


async def test_manager_api_preview_keeps_masked_branch_logits_but_marks_status(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["action"]["enable_boost"] = False

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["discrete_action_logits"] == 14
    boost_branch = next(
        branch for branch in payload["action_branches"] if branch["name"] == "boost"
    )
    assert boost_branch["enabled"] is False
    assert boost_branch["mask_label"] == "masked idle"


async def test_manager_api_preview_keeps_gas_head_but_marks_forced_full(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["action"]["force_full_throttle"] = True

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["continuous_action_dims"] == 1
    assert payload["discrete_action_logits"] == 14
    throttle_branch = next(
        branch for branch in payload["action_branches"] if branch["name"] == "throttle"
    )
    assert throttle_branch["enabled"] is False
    assert throttle_branch["mask_label"] == "forced engaged"


async def test_manager_api_preview_removes_logits_when_branch_excluded(tmp_path: Path) -> None:
    client = _client(tmp_path)
    config = default_managed_run_config().model_dump(mode="json")
    config["action"]["include_boost"] = False

    response = await client.post("/api/policy-preview", json=config)

    assert response.status_code == 200
    payload = response.json()
    assert payload["discrete_action_logits"] == 12
    branch_names = {branch["name"] for branch in payload["action_branches"]}
    assert "boost" not in branch_names
