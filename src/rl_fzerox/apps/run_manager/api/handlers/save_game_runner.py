# src/rl_fzerox/apps/run_manager/api/handlers/save_game_runner.py
from __future__ import annotations

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.contracts import RunLauncher, StartCareerModeRequest


def start_career_mode_payload(
    launcher: RunLauncher,
    save_game_id: str,
    request: StartCareerModeRequest,
) -> dict[str, str]:
    try:
        status = launcher.start_career_mode(
            save_game_id=save_game_id,
            device=request.device,
            renderer=request.renderer,
            attempt_seed=request.attempt_seed,
            deterministic_policy=request.policy_mode == "deterministic",
            recording_enabled=request.recording_enabled,
            recording_input_hud_enabled=request.recording_input_hud_enabled,
            recording_upscale_factor=request.recording_upscale_factor,
            recording_path=request.recording_path,
            target_kind=request.target_kind,
            difficulty=request.difficulty,
            cup_id=request.cup_id,
            course_id=request.course_id,
            single_target=request.single_target,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": status}
