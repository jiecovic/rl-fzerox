# src/rl_fzerox/ui/watch/runtime/career_mode/menu.py
"""Map Career Mode semantic menu inputs to emulator controller states."""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing.queues import Queue as ProcessQueue

from fzerox_emulator import MENU_BUTTON_MASKS, ControllerState, RaceControlState
from fzerox_emulator.arrays import Pcm16Samples
from rl_fzerox.core.career_mode.runner.controller import CareerModeController
from rl_fzerox.core.career_mode.runner.menu import MenuInput, RawMenuStep
from rl_fzerox.core.envs.engine.reset.camera import CAMERA_SYNC_CONTROLS
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.records import TrackRecordBook
from rl_fzerox.ui.watch.runtime.career_mode.recording import FrameRecorder
from rl_fzerox.ui.watch.runtime.career_mode.session import CareerModeRuntimeSession
from rl_fzerox.ui.watch.runtime.career_mode.timing import (
    measured_game_fps,
    target_game_fps,
    with_measured_game_fps,
)
from rl_fzerox.ui.watch.runtime.ipc import publish_worker_message
from rl_fzerox.ui.watch.runtime.snapshots import _build_snapshot
from rl_fzerox.ui.watch.runtime.timing import RateMeter


@dataclass(frozen=True, slots=True)
class RawControllerStep:
    """One emulator-controller pulse emitted from a semantic menu step."""

    controller_state: ControllerState
    frames: int
    phase: str

    def __post_init__(self) -> None:
        if self.frames <= 0:
            raise ValueError("frames must be positive")
        if not self.phase:
            raise ValueError("phase must not be empty")


@dataclass(frozen=True, slots=True)
class CareerMenuControls:
    """Controller states for Career Mode menu inputs."""

    neutral: ControllerState = ControllerState()
    start: ControllerState = ControllerState(joypad_mask=MENU_BUTTON_MASKS.start)
    accept: ControllerState = ControllerState(joypad_mask=MENU_BUTTON_MASKS.confirm)
    a_button: ControllerState = ControllerState(joypad_mask=MENU_BUTTON_MASKS.confirm)
    cancel: ControllerState = ControllerState(joypad_mask=MENU_BUTTON_MASKS.cancel)
    up: ControllerState = ControllerState(joypad_mask=MENU_BUTTON_MASKS.up)
    down: ControllerState = ControllerState(joypad_mask=MENU_BUTTON_MASKS.down)
    left: ControllerState = ControllerState(joypad_mask=MENU_BUTTON_MASKS.left)
    right: ControllerState = ControllerState(joypad_mask=MENU_BUTTON_MASKS.right)
    next_camera: ControllerState = CAMERA_SYNC_CONTROLS.next_camera


MENU_CONTROLS = CareerMenuControls()


_MENU_INPUT_STATES = {
    MenuInput.NEUTRAL: MENU_CONTROLS.neutral,
    MenuInput.ACCEPT: MENU_CONTROLS.accept,
    MenuInput.A_BUTTON: MENU_CONTROLS.a_button,
    MenuInput.CANCEL: MENU_CONTROLS.cancel,
    MenuInput.START: MENU_CONTROLS.start,
    MenuInput.UP: MENU_CONTROLS.up,
    MenuInput.DOWN: MENU_CONTROLS.down,
    MenuInput.LEFT: MENU_CONTROLS.left,
    MenuInput.RIGHT: MENU_CONTROLS.right,
    MenuInput.NEXT_CAMERA: MENU_CONTROLS.next_camera,
}


def controller_step_from_menu_step(step: RawMenuStep) -> RawControllerStep:
    return RawControllerStep(
        controller_state=_MENU_INPUT_STATES[step.menu_input],
        frames=step.frames,
        phase=step.phase,
    )


def neutral_controller_step(*, phase: str) -> RawControllerStep:
    return RawControllerStep(
        controller_state=MENU_CONTROLS.neutral,
        frames=1,
        phase=phase,
    )


def menu_viewer_info(session: CareerModeRuntimeSession) -> dict[str, object]:
    return reset_race_progress_info(session.menu_info())


def reset_race_progress_info(info: dict[str, object]) -> dict[str, object]:
    """Return menu-phase info without stale policy-race counters."""

    normalized = dict(info)
    normalized.update(
        {
            "episode_step": 0,
            "episode_return": 0.0,
            "step_reward": 0.0,
            "progress_frontier_stalled_frames": 0,
            "stalled_steps": 0,
            "frames_run": 0,
            "repeat_index": 0,
        }
    )
    normalized.pop("reward_breakdown", None)
    return normalized


def step_menu(
    *,
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    controller: CareerModeController,
    snapshot_queue: ProcessQueue,
    step: RawMenuStep | None,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode: int,
    episode_reward: float,
    control_rate: RateMeter,
    target_control_fps: float | None,
    native_frame_seconds: float | None,
    deterministic_policy: bool,
    track_record_book: TrackRecordBook,
    frame_recorder: FrameRecorder | None = None,
) -> None:
    if step is None:
        step = RawMenuStep(
            menu_input=MenuInput.NEUTRAL,
            frames=1,
            phase="menu:wait",
        )
    controller_step = controller_step_from_menu_step(step)
    session.emulator.set_controller_state(controller_step.controller_state)
    for frame_index in range(controller_step.frames):
        audio_samples: Pcm16Samples = ()
        if (
            frame_recorder is not None
            and session.native_sample_rate > 0.0
            and hasattr(session.emulator, "step_frames_with_audio")
        ):
            audio_samples = session.emulator.step_frames_with_audio(1, capture_video=True)
        else:
            session.emulator.step_frames(1, capture_video=True)
        info = controller.viewer_info(
            info=menu_viewer_info(session),
            active_policy_control=None,
        )
        info.update(
            {
                "career_mode_last_input": step.menu_input.value,
                "career_mode_last_step": controller_step.phase,
                "career_mode_last_step_frames": controller_step.frames,
            }
        )
        control_rate.tick()
        info = with_measured_game_fps(
            info,
            game_fps=measured_game_fps(
                control_fps=control_rate.rate_hz(),
                action_repeat=1,
            ),
            game_fps_target=target_game_fps(
                target_control_fps=target_control_fps,
                action_repeat=1,
            ),
        )
        raw_frame = session.render()
        if frame_recorder is not None:
            frame_recorder.record_frame(raw_frame, info=info, audio_samples=audio_samples)
        publish_worker_message(
            snapshot_queue,
            _build_snapshot(
                config=config,
                env=session,
                emulator=session.emulator,
                raw_frame=raw_frame,
                observation=None,
                info=info,
                reset_info=reset_info,
                episode=episode,
                episode_reward=0.0,
                control_fps=control_rate.rate_hz(),
                target_control_fps=target_control_fps,
                action_repeat=1,
                control_state=RaceControlState(),
                gas_level=0.0,
                boost_lamp_level=0.0,
                action_mask_branches=session.action_mask_branches(),
                policy_action=None,
                policy_runner=None,
                deterministic_policy=deterministic_policy,
                manual_control_enabled=False,
                policy_reload_error=None,
                cnn_activations=None,
                active_track_sampling=None,
                track_record_book=track_record_book,
                action_hold_frame=frame_index + 1,
                action_hold_frames=controller_step.frames,
                policy_decision_frame=False,
            ),
        )
        if native_frame_seconds is not None:
            time.sleep(native_frame_seconds)
    session.emulator.set_controller_state(
        neutral_controller_step(phase="menu:neutral").controller_state
    )
