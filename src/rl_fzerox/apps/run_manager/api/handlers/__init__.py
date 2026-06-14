# src/rl_fzerox/apps/run_manager/api/handlers/__init__.py
from __future__ import annotations

from rl_fzerox.apps.run_manager.api.handlers.common import (
    run_response_for_id,
    validate_source_fields,
)
from rl_fzerox.apps.run_manager.api.handlers.config import (
    config_metadata_payload,
    policy_preview_payload,
)
from rl_fzerox.apps.run_manager.api.handlers.drafts import (
    create_draft_payload,
    delete_draft_payload,
    drafts_payload,
    templates_payload,
    update_draft_payload,
)
from rl_fzerox.apps.run_manager.api.handlers.lineages import (
    delete_lineage_payload,
    rebuild_tensorboard_views_payload,
    update_lineage_groups_payload,
)
from rl_fzerox.apps.run_manager.api.handlers.metrics import (
    clear_run_alt_baselines_payload,
    clear_run_course_alt_baselines_payload,
    reset_run_engine_tuning_payload,
    reset_run_track_sampling_payload,
    run_alt_baselines_payload,
    run_engine_tuning_payload,
    run_metrics_payload,
    run_track_sampling_payload,
)
from rl_fzerox.apps.run_manager.api.handlers.runs import (
    delete_run_payload,
    fork_run_payload,
    launch_run_payload,
    open_run_dir_payload,
    pause_run_payload,
    resume_run_payload,
    runs_payload,
    stop_run_payload,
    update_run_payload,
    watch_run_payload,
)
from rl_fzerox.apps.run_manager.api.handlers.save_game_attempts import (
    save_attempt_execution_context_payload_for_attempt,
    save_attempt_execution_plan_payload_for_attempt,
    start_next_save_attempt_payload,
)
from rl_fzerox.apps.run_manager.api.handlers.save_game_runner import (
    start_career_mode_payload,
)
from rl_fzerox.apps.run_manager.api.handlers.save_games import (
    create_save_game_payload,
    delete_save_game_payload,
    import_save_engine_tuning_payload,
    open_save_game_dir_payload,
    save_game_status_payload_for_id,
    save_games_payload,
    update_save_game_payload,
    update_save_game_runner_settings_payload,
    upsert_save_course_setup_payload,
    upsert_save_cup_setup_payload,
)
from rl_fzerox.apps.run_manager.api.handlers.transfer import (
    export_run_bundle_path,
    import_run_bundle_payload,
)

__all__ = [
    "config_metadata_payload",
    "clear_run_alt_baselines_payload",
    "clear_run_course_alt_baselines_payload",
    "create_draft_payload",
    "create_save_game_payload",
    "delete_draft_payload",
    "delete_lineage_payload",
    "delete_run_payload",
    "delete_save_game_payload",
    "drafts_payload",
    "export_run_bundle_path",
    "fork_run_payload",
    "import_run_bundle_payload",
    "launch_run_payload",
    "open_run_dir_payload",
    "open_save_game_dir_payload",
    "pause_run_payload",
    "policy_preview_payload",
    "rebuild_tensorboard_views_payload",
    "reset_run_engine_tuning_payload",
    "reset_run_track_sampling_payload",
    "run_alt_baselines_payload",
    "run_engine_tuning_payload",
    "resume_run_payload",
    "run_metrics_payload",
    "run_response_for_id",
    "run_track_sampling_payload",
    "runs_payload",
    "save_attempt_execution_context_payload_for_attempt",
    "save_attempt_execution_plan_payload_for_attempt",
    "save_game_status_payload_for_id",
    "save_games_payload",
    "start_career_mode_payload",
    "start_next_save_attempt_payload",
    "stop_run_payload",
    "templates_payload",
    "update_draft_payload",
    "update_lineage_groups_payload",
    "update_run_payload",
    "update_save_game_payload",
    "update_save_game_runner_settings_payload",
    "upsert_save_course_setup_payload",
    "import_save_engine_tuning_payload",
    "upsert_save_cup_setup_payload",
    "validate_source_fields",
    "watch_run_payload",
]
