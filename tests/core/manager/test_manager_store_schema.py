# tests/core/manager/test_manager_store_schema.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

import rl_fzerox.core.manager.store as store_module
from rl_fzerox.core.manager import (
    ManagedRunConfig,
    ManagerStore,
    default_managed_run_config,
)
from rl_fzerox.core.manager.db import manager_session
from rl_fzerox.core.manager.db.models import (
    RunTemplateModel,
)
from rl_fzerox.core.manager.storage.serialization import config_hash, config_json, load_config_json
from tests.core.manager.manager_store_support import (
    _config_snapshot_json,
    _insert_config_snapshot,
    _insert_stale_draft_config,
    _table_columns,
)

SnapshotKind = Literal["run", "draft", "template", "import"]


def test_manager_store_seeds_default_template(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")

    templates = store.list_templates()

    assert len(templates) == 1
    assert templates[0].id == "all_cups_recurrent_ppo"
    assert templates[0].config == default_managed_run_config()


def test_manager_store_initializes_schema_only_once_per_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0
    original_initialize_schema = store_module.initialize_manager_schema

    def wrapped_initialize_schema(db_path: Path, *, applied_at: str) -> None:
        nonlocal call_count
        call_count += 1
        original_initialize_schema(db_path, applied_at=applied_at)

    monkeypatch.setattr(store_module, "initialize_manager_schema", wrapped_initialize_schema)

    store = ManagerStore(tmp_path / "runs.db")

    assert store.pending_run_command("missing-run") is None
    assert store.pending_run_command("missing-run") is None

    assert call_count == 1


def test_manager_store_refreshes_system_template_to_current_defaults(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")
    store.initialize()

    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["reward"]["energy_refill_progress_multiplier"] = 1.0
    stale_model = ManagedRunConfig.model_validate(stale_config)

    with manager_session(store.db_path) as session:
        snapshot_id = _insert_config_snapshot(
            session=session,
            snapshot_id="stale-template-config",
            kind="template",
            raw_config_json=config_json(stale_model),
            stored_config_hash=config_hash(stale_model),
            created_at="2026-05-01T00:00:00+00:00",
        )
        template = session.get(RunTemplateModel, "all_cups_recurrent_ppo")
        assert template is not None
        template.config_snapshot_id = snapshot_id
        template.updated_at = "2026-05-01T00:00:00+00:00"

    template = store.default_template()

    assert template.config == default_managed_run_config()
    assert template.config.reward.energy_refill_progress_multiplier == 3.0
    assert load_config_json(_config_snapshot_json(store, "stale-template-config")) == stale_model


def test_manager_store_normalizes_stale_draft_configs(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["reward"] = {"manual_boost_reward": 0.5}

    _insert_stale_draft_config(store, draft_id="old-draft", name="Old Draft", config=stale_config)

    draft = store.list_drafts()[0]

    assert draft.config.reward.manual_boost_reward == 0.5
    assert draft.config.reward.time_penalty_per_frame == 0.0


def test_manager_config_ignores_removed_adaptive_step_balance_fields() -> None:
    config_data = default_managed_run_config().model_dump(mode="json")
    config_data["tracks"].update(
        {
            "adaptive_step_balance_completion_weight": 1.0,
            "adaptive_step_balance_confidence_scale": 4.0,
            "adaptive_step_balance_min_confidence_episodes": 24,
            "adaptive_step_balance_target_completion": 0.8,
            "sampling_mode": "adaptive_step_balanced",
        }
    )

    config = ManagedRunConfig.model_validate(config_data)
    serialized_tracks = config.model_dump(mode="json")["tracks"]

    assert config.tracks.sampling_mode == "step_balanced"
    assert "adaptive_step_balance_completion_weight" not in serialized_tracks
    assert "adaptive_step_balance_confidence_scale" not in serialized_tracks
    assert "adaptive_step_balance_min_confidence_episodes" not in serialized_tracks
    assert "adaptive_step_balance_target_completion" not in serialized_tracks


def test_manager_config_maps_removed_track_sampling_mode_names() -> None:
    base_config_data = default_managed_run_config().model_dump(mode="json")

    for old_mode, expected_mode in {
        "adaptive_step_balanced": "step_balanced",
        "balanced": "equal",
        "random": "equal",
    }.items():
        config_data = {
            **base_config_data,
            "tracks": {
                **base_config_data["tracks"],
                "sampling_mode": old_mode,
            },
        }

        config = ManagedRunConfig.model_validate(config_data)

        assert config.tracks.sampling_mode == expected_mode


def test_mlp_engine_tuner_snapshot_omits_gp_only_fields() -> None:
    base_config = default_managed_run_config()
    config = base_config.model_copy(
        update={
            "vehicle": base_config.vehicle.model_copy(
                update={
                    "engine_mode": "adaptive_tuner",
                    "adaptive_engine_tuner_backend": "mlp_ensemble",
                }
            )
        }
    )

    vehicle = json.loads(config_json(config))["vehicle"]

    assert vehicle["adaptive_engine_tuner_backend"] == "mlp_ensemble"
    assert "adaptive_engine_uniform_exploration" in vehicle
    assert "adaptive_engine_stat_decay" not in vehicle
    assert "adaptive_engine_ensemble_members" in vehicle
    assert "adaptive_engine_exploration_scale" not in vehicle
    assert "adaptive_engine_randomized_prior_seconds" not in vehicle


def test_bandit_engine_tuner_snapshot_omits_experimental_backend_fields() -> None:
    base_config = default_managed_run_config()
    config = base_config.model_copy(
        update={
            "vehicle": base_config.vehicle.model_copy(
                update={
                    "engine_mode": "adaptive_tuner",
                    "adaptive_engine_tuner_backend": "bandit",
                    "adaptive_engine_bandit_bucket_raw_values": (44, 54, 64, 74, 84),
                }
            )
        }
    )

    vehicle = json.loads(config_json(config))["vehicle"]

    assert vehicle["adaptive_engine_tuner_backend"] == "bandit"
    assert vehicle["adaptive_engine_bandit_bucket_raw_values"] == [44, 54, 64, 74, 84]
    assert "adaptive_engine_uniform_exploration" in vehicle
    assert "adaptive_engine_stat_decay" not in vehicle
    assert "adaptive_engine_ensemble_members" not in vehicle


def test_manager_store_creates_current_runs_schema(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()

    columns = _table_columns(store, "runs")
    group_columns = _table_columns(store, "lineage_groups")
    snapshot_columns = _table_columns(store, "config_snapshots")
    save_game_columns = _table_columns(store, "save_games")
    save_attempt_columns = _table_columns(store, "save_game_attempts")
    course_setup_columns = _table_columns(store, "save_game_course_setups")
    evaluation_columns = _table_columns(store, "evaluations")
    evaluation_preset_columns = _table_columns(store, "evaluation_presets")
    generated_slot_columns = _table_columns(store, "run_track_sampling_generated_slots")

    assert columns == {
        "id",
        "name",
        "status",
        "config_snapshot_id",
        "run_dir",
        "lineage_id",
        "lineage_step_offset",
        "parent_run_id",
        "source_run_id",
        "source_artifact",
        "source_snapshot_dir",
        "source_num_timesteps",
        "created_at",
        "started_at",
        "stopped_at",
    }
    assert group_columns == {"lineage_id", "group_name", "updated_at"}
    assert snapshot_columns == {
        "id",
        "kind",
        "schema_version",
        "created_at",
        "config_json",
        "config_hash",
    }
    assert save_game_columns == {
        "id",
        "name",
        "status",
        "save_path",
        "created_at",
        "updated_at",
        "last_finished_at",
        "runner_device",
        "runner_renderer",
        "runner_policy_mode",
        "runner_attempt_seed",
        "runner_recording_enabled",
        "runner_recording_input_hud_enabled",
        "runner_recording_upscale_factor",
        "runner_recording_path",
        "runner_target_restart_on_retire",
        "runner_target_clear_goal",
        "runner_keep_failed_recordings",
        "runner_reload_policy_between_attempts",
    }
    assert save_attempt_columns == {
        "id",
        "save_game_id",
        "status",
        "target_kind",
        "difficulty",
        "cup_id",
        "course_id",
        "started_at",
        "finished_at",
        "finish_position",
        "finish_time_s",
        "failure_reason",
    }
    assert course_setup_columns == {
        "id",
        "save_game_id",
        "difficulty",
        "cup_id",
        "course_id",
        "policy_run_id",
        "policy_artifact",
        "engine_setting_raw_value",
        "created_at",
        "updated_at",
    }
    assert evaluation_columns == {
        "id",
        "name",
        "status",
        "evaluation_dir",
        "source_run_id",
        "source_artifact",
        "preset_id",
        "preset_version",
        "policy_mode",
        "seed",
        "target_json",
        "config_json",
        "checkpoint_json",
        "result_json_path",
        "error_message",
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
    }
    assert evaluation_preset_columns == {
        "id",
        "name",
        "version",
        "seed",
        "renderer",
        "target_json",
        "builtin",
        "created_at",
        "updated_at",
    }
    cup_setup_columns = _table_columns(store, "save_game_cup_setups")
    assert cup_setup_columns == {
        "id",
        "save_game_id",
        "difficulty",
        "cup_id",
        "vehicle_id",
        "created_at",
        "updated_at",
    }
    assert generated_slot_columns == {
        "run_id",
        "slot",
        "course_key",
        "generation",
        "course_id",
        "course_name",
        "course_hash",
        "course_seed",
        "segment_count",
        "course_length",
        "updated_at",
    }


def test_manager_store_rejects_removed_observation_fields(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["observation"] = {
        "frame_stack": 2,
        "minimap_layer": False,
        "preset": "crop_84x84",
        "progress_source": "segment_progress",
        "stack_mode": "rgb",
        "zero_edge_ratio": True,
        "zero_outside_track_bounds": True,
    }

    _insert_stale_draft_config(
        store,
        draft_id="old-observation",
        name="Old Observation",
        config=stale_config,
    )

    with pytest.raises(ValidationError):
        store.list_drafts()


def test_manager_store_rejects_removed_state_modes(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["observation"]["state_components"] = [
        {"name": "vehicle_state", "mode": "include"},
        {"name": "track_position", "mode": "zero", "progress_source": "segment_progress"},
        {"name": "course_context", "mode": "exclude", "encoding": "one_hot_builtin"},
    ]
    stale_config["observation"]["state_feature_modes"] = [
        {"name": "track_position.segment_progress", "mode": "include", "dropout_prob": 0.25},
        {"name": "track_position.edge_ratio", "mode": "exclude", "dropout_prob": 0.0},
    ]

    _insert_stale_draft_config(
        store,
        draft_id="old-state-modes",
        name="Old State Modes",
        config=stale_config,
    )

    with pytest.raises(ValidationError):
        store.list_drafts()


def test_manager_store_rejects_removed_vehicle_fields(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["vehicle"] = {
        "vehicle_id": "golden_fox",
        "engine_setting_raw_value": 65,
    }

    _insert_stale_draft_config(
        store,
        draft_id="old-vehicle",
        name="Old Vehicle",
        config=stale_config,
    )

    with pytest.raises(ValidationError):
        store.list_drafts()


def test_manager_store_normalizes_missing_action_config(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config.pop("action", None)

    _insert_stale_draft_config(store, draft_id="old-action", name="Old Action", config=stale_config)

    draft = store.list_drafts()[0]

    assert draft.config.action.action_repeat == 2
    assert draft.config.action.steering_mode == "continuous"
    assert draft.config.action.drive_mode == "on_off"
    assert draft.config.action.force_full_throttle is False
    assert draft.config.action.include_air_brake is True
    assert draft.config.action.enable_air_brake is True
    assert draft.config.action.boost_decision_interval_steps == 1
    assert draft.config.action.boost_unmask_max_speed_kph is None
    assert draft.config.action.boost_min_energy_fraction == 0.1
    assert draft.config.action.lean_output_mode == "three_way"
    assert draft.config.action.lean_mode == "release_cooldown"
    assert draft.config.action.lean_unmask_min_speed_kph is None
    assert draft.config.action.lean_initial_lockout_frames == 0
    assert draft.config.action.lean_episode_mask_probability == 0.0
    assert draft.config.action.air_brake_episode_mask_probability == 0.0
    assert draft.config.action.air_brake_pulse_frames == 0
    assert draft.config.action.spin_episode_mask_probability == 0.0
    assert draft.config.action.include_pitch is True
    assert draft.config.action.enable_pitch is True
    assert draft.config.action.pitch_mode == "discrete"
    assert draft.config.action.pitch_buckets == 5


def test_manager_store_rejects_removed_progress_suspend_field(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["reward"]["suspend_progress_while_outside_track_bounds"] = False
    stale_config["reward"].pop("suspend_progress_while_outside_track_bounds")
    stale_config["reward"]["suspend_progress_while_airborne"] = True

    _insert_stale_draft_config(
        store,
        draft_id="old-progress-suspend",
        name="Old Progress Suspend",
        config=stale_config,
    )

    with pytest.raises(ValidationError):
        store.list_drafts()
