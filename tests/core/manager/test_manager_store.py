# tests/core/manager/test_manager_store.py
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.manager.errors import ManagerNameConflictError


def test_manager_store_seeds_default_template(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")

    templates = store.list_templates()

    assert len(templates) == 1
    assert templates[0].id == "all_cups_recurrent_ppo"
    assert templates[0].config == default_managed_run_config()


def test_manager_store_refreshes_system_template_to_current_defaults(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")
    store.initialize()

    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["reward"]["energy_refill_progress_multiplier"] = 1.0

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            UPDATE run_templates
            SET config_json = ?, config_hash = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                json.dumps(stale_config),
                "stale",
                "2026-05-01T00:00:00+00:00",
                "all_cups_recurrent_ppo",
            ),
        )

    template = store.default_template()

    assert template.config == default_managed_run_config()
    assert template.config.reward.energy_refill_progress_multiplier == 3.0


def test_manager_store_saves_draft_without_filesystem_artifacts(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config().model_copy(update={"seed": 321})

    draft = store.create_draft(
        name="Prototype Run",
        config=config,
    )

    drafts = store.list_drafts()
    assert len(drafts) == 1
    assert drafts[0].id == draft.id
    assert drafts[0].name == "Prototype Run"
    assert drafts[0].config == config
    assert not (tmp_path / "managed_runs").exists()


def test_manager_store_deletes_draft(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    draft = store.create_draft(
        name="Delete Me",
        config=default_managed_run_config(),
    )

    assert store.delete_draft(draft.id)

    assert store.list_drafts() == ()


def test_manager_store_normalizes_stale_draft_configs(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["reward"] = {"manual_boost_reward": 0.5}

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            INSERT INTO run_drafts(
                id,
                name,
                config_json,
                config_hash,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "old-draft",
                "Old Draft",
                json.dumps(stale_config),
                "stale",
                "2026-05-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
            ),
        )

    draft = store.list_drafts()[0]

    assert draft.config.reward.manual_boost_reward == 0.5
    assert draft.config.reward.time_penalty_per_frame == -0.005
    assert draft.config.reward.step_reward_clip_max == 100.0


def test_manager_store_normalizes_legacy_observation_fields(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["observation"] = {
        "frame_stack": 2,
        "minimap_layer": False,
        "preset": "crop_60x76",
        "progress_source": "segment_progress",
        "stack_mode": "rgb",
        "zero_edge_ratio": True,
        "zero_outside_track_bounds": True,
    }

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            INSERT INTO run_drafts(
                id,
                name,
                config_json,
                config_hash,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "old-observation",
                "Old Observation",
                json.dumps(stale_config),
                "stale",
                "2026-05-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
            ),
        )

    draft = store.list_drafts()[0]

    assert draft.config.observation.state_components[2].name == "track_position"
    assert draft.config.observation.state_components[2].progress_source == "segment_progress"
    assert tuple(
        feature.model_dump(mode="json")
        for feature in draft.config.observation.state_feature_modes
    ) == (
        {"name": "track_position.edge_ratio", "mode": "zero"},
        {"name": "track_position.outside_track_bounds", "mode": "zero"},
    )


def test_manager_store_normalizes_legacy_vehicle_fields(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config["vehicle"] = {
        "vehicle_id": "golden_fox",
        "engine_setting_raw_value": 65,
    }

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            INSERT INTO run_drafts(
                id,
                name,
                config_json,
                config_hash,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "old-vehicle",
                "Old Vehicle",
                json.dumps(stale_config),
                "stale",
                "2026-05-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
            ),
        )

    draft = store.list_drafts()[0]

    assert draft.config.vehicle.selection_mode == "pool"
    assert draft.config.vehicle.selected_vehicle_ids == ("golden_fox",)
    assert draft.config.vehicle.engine_mode == "fixed"
    assert draft.config.vehicle.engine_setting_raw_value == 65
    assert draft.config.vehicle.engine_setting_min_raw_value == 65
    assert draft.config.vehicle.engine_setting_max_raw_value == 65


def test_manager_store_normalizes_missing_action_config(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.initialize()
    stale_config = default_managed_run_config().model_dump(mode="json")
    stale_config.pop("action", None)

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            INSERT INTO run_drafts(
                id,
                name,
                config_json,
                config_hash,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "old-action",
                "Old Action",
                json.dumps(stale_config),
                "stale",
                "2026-05-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
            ),
        )

    draft = store.list_drafts()[0]

    assert draft.config.action.action_repeat == 2
    assert draft.config.action.steering_mode == "continuous"
    assert draft.config.action.drive_mode == "on_off"
    assert draft.config.action.force_full_throttle is False
    assert draft.config.action.include_air_brake is True
    assert draft.config.action.enable_air_brake is True
    assert draft.config.action.boost_unmask_max_speed_kph is None
    assert draft.config.action.boost_min_energy_fraction == 0.1
    assert draft.config.action.lean_output_mode == "three_way"
    assert draft.config.action.lean_mode == "release_cooldown"
    assert draft.config.action.lean_unmask_min_speed_kph is None
    assert draft.config.action.lean_initial_lockout_frames == 0
    assert draft.config.action.include_pitch is True
    assert draft.config.action.enable_pitch is True
    assert draft.config.action.pitch_mode == "discrete"
    assert draft.config.action.pitch_buckets == 5



def test_manager_store_creates_run_record_without_filesystem_artifacts(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()

    run = store.create_run(
        name="Started Later",
        config=config,
        managed_runs_root=tmp_path / "managed_runs",
    )

    assert store.list_runs()[0].id == run.id
    assert not run.run_dir.exists()


def test_manager_store_visible_runs_exclude_created_records(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    store.create_run(
        name="Created Only",
        config=default_managed_run_config(),
        managed_runs_root=tmp_path / "managed_runs",
    )

    assert store.list_visible_runs() == ()


def test_manager_store_rejects_draft_name_used_by_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    store.create_run(name="Shared Name", config=config, managed_runs_root=tmp_path / "managed_runs")

    with pytest.raises(ManagerNameConflictError, match="name already exists: Shared Name"):
        store.create_draft(name="Shared Name", config=config)


def test_manager_store_rejects_run_name_used_by_draft(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    config = default_managed_run_config()
    store.create_draft(name="Shared Name", config=config)

    with pytest.raises(ManagerNameConflictError, match="name already exists: Shared Name"):
        store.create_run(
            name="Shared Name",
            config=config,
            managed_runs_root=tmp_path / "managed_runs",
        )
