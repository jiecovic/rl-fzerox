# tests/core/manager/test_manager_store.py
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from rl_fzerox.core.manager import ManagerStore, default_managed_run_config


def test_manager_store_seeds_default_template(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "runs.db")

    templates = store.list_templates()

    assert len(templates) == 1
    assert templates[0].id == "all_cups_recurrent_ppo"
    assert templates[0].config == default_managed_run_config()


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
