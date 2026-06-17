# src/rl_fzerox/core/career_mode/runner/view.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.career_mode.navigation import (
    CareerPhase,
    DifficultyPopupState,
    MenuFacts,
    ObservedMenuScreen,
    camera_setting,
    course_id_from_info,
)
from rl_fzerox.core.career_mode.runner.policy import CareerModePolicyControl
from rl_fzerox.core.domain.camera import CameraSettingName
from rl_fzerox.core.manager.models import ManagedSaveUnlockProgress, SaveAttemptStatus
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig


@dataclass(frozen=True, slots=True)
class CareerControllerViewState:
    attempt_id: str | None
    setup: CareerModeRaceSetupConfig
    progress: ManagedSaveUnlockProgress
    phase: CareerPhase
    pending_step_count: int
    difficulty_popup_state: DifficultyPopupState
    camera_synced: bool
    camera_setting: CameraSettingName | None
    awaiting_new_race_after_terminal: bool
    continuing_race_result: bool
    fresh_race_ready_frames: int
    last_finished_attempt_id: str | None
    last_finished_attempt_status: SaveAttemptStatus | None
    last_finished_attempt_failure_reason: str | None


def career_viewer_info(
    *,
    info: dict[str, object],
    state: CareerControllerViewState,
    observed_screen: ObservedMenuScreen,
    active_policy_control: CareerModePolicyControl | None,
) -> dict[str, object]:
    """Add Career Mode control/progress context to viewer telemetry."""

    progress = state.progress
    current_target_label = _current_target_label(progress, state.setup)
    next_target = progress.next_target
    viewer_info = dict(info)
    viewer_info.update(
        {
            "career_mode_attempt_id": state.attempt_id,
            "career_mode_completed_targets": progress.completed_count,
            "career_mode_controller_context": career_debug_context(info=info, state=state),
            "career_mode_total_targets": progress.total_count,
            "career_mode_inspection_status": progress.inspection_status,
            "career_mode_next_target_label": (
                next_target.label if next_target is not None else None
            ),
            "career_mode_phase": state.phase.value,
            "career_mode_policy_active": active_policy_control is not None,
            "career_mode_target_label": current_target_label,
            "career_mode_last_finished_attempt_id": state.last_finished_attempt_id,
            "career_mode_last_finished_attempt_status": state.last_finished_attempt_status,
            "career_mode_last_finished_attempt_failure_reason": (
                state.last_finished_attempt_failure_reason
            ),
        }
    )
    viewer_info.update(
        _viewer_fsm_facts(
            info=info,
            state=state,
            observed_screen=observed_screen,
        )
    )
    if active_policy_control is not None:
        policy_runner = active_policy_control.runner
        viewer_info.update(
            {
                "career_mode_policy_artifact": active_policy_control.course_setup.policy_artifact,
                "career_mode_policy_checkpoint_local_num_timesteps": (
                    policy_runner.checkpoint_local_num_timesteps
                ),
                "career_mode_policy_checkpoint_mtime_ns": (
                    policy_runner.checkpoint_policy_mtime_ns
                ),
                "career_mode_policy_checkpoint_mtime_utc": (
                    policy_runner.checkpoint_policy_mtime_utc
                ),
                "career_mode_policy_checkpoint_num_timesteps": (
                    policy_runner.checkpoint_num_timesteps
                ),
                "career_mode_policy_checkpoint_path": str(policy_runner.checkpoint_policy_path),
                "career_mode_policy_checkpoint_stage": (policy_runner.checkpoint_curriculum_stage),
                "career_mode_policy_checkpoint_stage_index": (
                    policy_runner.checkpoint_curriculum_stage_index
                ),
                "career_mode_policy_run_id": active_policy_control.policy_run.id,
                "career_mode_policy_run_name": active_policy_control.policy_run.name,
                "career_mode_policy_course_id": (
                    active_policy_control.course_setup.course_id or course_id_from_info(info)
                ),
            }
        )
    return viewer_info


def career_debug_context(
    *,
    info: dict[str, object],
    state: CareerControllerViewState,
) -> str:
    fields = (
        ("phase", state.phase.value),
        ("pending_steps", state.pending_step_count),
        ("game_mode", info.get("game_mode")),
        ("game_mode_raw", info.get("game_mode_raw")),
        ("queued_game_mode", info.get("queued_game_mode")),
        ("queued_game_mode_raw", info.get("queued_game_mode_raw")),
        ("menu_selected_mode_raw", info.get("menu_selected_mode_raw")),
        ("menu_difficulty_state_raw", info.get("menu_difficulty_state_raw")),
        ("menu_difficulty_cursor_raw", info.get("menu_difficulty_cursor_raw")),
        ("menu_transition_state_raw", info.get("menu_transition_state_raw")),
        ("difficulty", info.get("difficulty")),
        ("difficulty_raw", info.get("difficulty_raw")),
        ("course_index", info.get("course_index")),
        ("camera_setting", camera_setting(info)),
        ("race_intro_timer", info.get("race_intro_timer")),
        ("race_time_ms", info.get("race_time_ms")),
        ("laps", _lap_summary(info)),
        ("position", info.get("position")),
        ("termination_reason", info.get("termination_reason")),
    )
    parts = [f"{name}={value!r}" for name, value in fields if value is not None]
    return ", ".join(parts)


def _viewer_fsm_facts(
    *,
    info: dict[str, object],
    state: CareerControllerViewState,
    observed_screen: ObservedMenuScreen,
) -> dict[str, object]:
    facts = MenuFacts.from_info(info)
    return {
        "career_mode_fsm_awaiting_fresh_race": state.awaiting_new_race_after_terminal,
        "career_mode_fsm_camera_synced": state.camera_synced,
        "career_mode_fsm_camera_target": state.camera_setting,
        "career_mode_fsm_completed_laps": facts.completed_laps,
        "career_mode_fsm_completion_fraction": facts.completion_fraction,
        "career_mode_fsm_continuing_result": state.continuing_race_result,
        "career_mode_fsm_course_index": facts.course_index,
        "career_mode_fsm_difficulty_cursor_raw": facts.difficulty_cursor_raw,
        "career_mode_fsm_difficulty_state_raw": facts.difficulty_state_raw,
        "career_mode_fsm_fresh_race_ready": facts.fresh_race_ready_for_policy,
        "career_mode_fsm_fresh_race_ready_frames": state.fresh_race_ready_frames,
        "career_mode_fsm_game_mode": facts.game_mode,
        "career_mode_fsm_intro_timer": facts.race_intro_timer,
        "career_mode_fsm_observed_screen": observed_screen.value,
        "career_mode_fsm_pending_steps": state.pending_step_count,
        "career_mode_fsm_popup_state": state.difficulty_popup_state.value,
        "career_mode_fsm_race_time_ms": facts.race_time_ms,
        "career_mode_fsm_selected_mode_raw": facts.selected_mode_raw,
        "career_mode_fsm_terminal_reason": facts.terminal_reason,
        "career_mode_fsm_terminal_result": facts.terminal_race_result,
        "career_mode_fsm_total_laps": facts.total_laps,
        "career_mode_fsm_transition_raw": facts.transition_state_raw,
    }


def _current_target_label(
    progress: ManagedSaveUnlockProgress,
    setup: CareerModeRaceSetupConfig,
) -> str:
    target = next(
        (
            target
            for target in progress.targets
            if target.kind == "clear_gp_cup"
            and target.difficulty == setup.difficulty
            and target.cup_id == setup.cup_id
        ),
        None,
    )
    if target is not None:
        return target.label
    return f"Clear {setup.difficulty.title()} {setup.cup_id.title()} Cup"


def _lap_summary(info: dict[str, object]) -> str | None:
    completed = info.get("race_laps_completed")
    total = info.get("total_lap_count")
    if (
        isinstance(completed, bool)
        or isinstance(total, bool)
        or not isinstance(completed, int | float)
        or not isinstance(total, int | float)
    ):
        return None
    return f"{int(completed)}/{int(total)}"
