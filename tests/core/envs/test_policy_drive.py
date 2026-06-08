# tests/core/envs/test_policy_drive.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from fzerox_emulator import (
    BackendStepResult,
    FZeroXTelemetry,
    ObservationStackMode,
    RaceControlState,
)
from rl_fzerox.core.envs.policy_drive import PolicyDriveRuntime
from rl_fzerox.core.runtime_spec.schema import EmulatorConfig, EnvConfig, TrainAppConfig
from rl_fzerox.core.runtime_spec.schema.actions import ActionMaskConfig
from tests.support.action_configs import configured_discrete_action
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import make_step_status, make_step_summary, make_telemetry


class ScriptedPolicyDriveBackend(SyntheticBackend):
    def __init__(self, results: list[BackendStepResult]) -> None:
        super().__init__()
        self._results = list(results)
        self._telemetry = make_telemetry(race_distance=0.0)
        self.last_spin_request: object = "none"

    def try_read_telemetry(self) -> FZeroXTelemetry | None:
        return self._telemetry

    def step_repeat_watch_raw(
        self,
        control_state: RaceControlState,
        *,
        action_repeat: int,
        preset: str | None = None,
        height: int | None = None,
        width: int | None = None,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        minimap_layer: bool = False,
        resize_filter: object = "nearest",
        minimap_resize_filter: object = "nearest",
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
        spin_request: object = "none",
        spin_cooldown_frames: int = 120,
    ) -> BackendStepResult:
        _ = (
            action_repeat,
            stuck_min_speed_kph,
            energy_loss_epsilon,
            max_episode_steps,
            progress_frontier_stall_limit_frames,
            progress_frontier_epsilon,
            terminate_on_energy_depleted,
            lean_timer_assist,
            spin_cooldown_frames,
        )
        self._last_race_control_state = control_state
        self.last_spin_request = spin_request
        result = self._results.pop(0)
        if result.telemetry is None:
            raise AssertionError("scripted policy-drive steps must include telemetry")
        self._state.frame_index = result.summary.final_frame_index
        self._state.progress = result.summary.max_race_distance
        self._telemetry = result.telemetry
        display_frame = self.render_display(preset=preset, height=height, width=width)
        frames_run = int(result.summary.frames_run)
        return BackendStepResult(
            observation=self.render_observation(
                preset=preset,
                height=height,
                width=width,
                frame_stack=frame_stack,
                stack_mode=stack_mode,
                minimap_layer=minimap_layer,
                resize_filter=resize_filter,
                minimap_resize_filter=minimap_resize_filter,
            ),
            summary=result.summary,
            status=result.status,
            telemetry=result.telemetry,
            display_frames=np.stack([display_frame] * frames_run),
            display_controller_masks=np.full(
                frames_run,
                control_state.control_mask,
                dtype=np.uint16,
            ),
        )


def test_policy_drive_uses_live_episode_step_instead_of_backend_counter(
    tmp_path: Path,
) -> None:
    runtime = PolicyDriveRuntime(
        emulator=ScriptedPolicyDriveBackend(
            [
                _backend_step(
                    frames_run=3,
                    race_distance=1_500.0,
                    native_step_count=12_345,
                ),
                _backend_step(
                    frames_run=3,
                    race_distance=2_600.0,
                    native_step_count=12_348,
                ),
            ]
        ),
        train_config=_train_config(tmp_path),
    )

    _, begin_info = runtime.begin(seed=7, course_id="mute_city")
    first = runtime.step_manual(RaceControlState())
    second = runtime.step_manual(RaceControlState())

    assert begin_info["episode_return"] == 0.0
    assert first.info["episode_step"] == 3
    assert second.info["episode_step"] == 6
    assert "truncation_reason" not in first.info


def test_policy_drive_episode_return_comes_from_shared_engine_state(tmp_path: Path) -> None:
    runtime = PolicyDriveRuntime(
        emulator=ScriptedPolicyDriveBackend(
            [
                _backend_step(
                    frames_run=3,
                    race_distance=1_500.0,
                    native_step_count=12_345,
                ),
                _backend_step(
                    frames_run=3,
                    race_distance=2_600.0,
                    native_step_count=12_348,
                ),
            ]
        ),
        train_config=_train_config(tmp_path),
    )

    runtime.begin(seed=7, course_id="mute_city")
    first = runtime.step_manual(RaceControlState())
    second = runtime.step_manual(RaceControlState())

    assert first.info["episode_return"] == first.reward
    assert second.info["episode_return"] == first.reward + second.reward
    assert first.reward > -1.0


def test_policy_drive_manual_step_forwards_spin_request(tmp_path: Path) -> None:
    backend = ScriptedPolicyDriveBackend(
        [
            _backend_step(
                frames_run=3,
                race_distance=1_500.0,
                native_step_count=12_345,
            )
        ]
    )
    runtime = PolicyDriveRuntime(
        emulator=backend,
        train_config=_train_config(tmp_path),
    )

    runtime.begin(seed=7, course_id="mute_city")
    step = runtime.step_manual(RaceControlState(), spin_request="left")

    assert backend.last_spin_request == "left"
    assert step.info["spin_requested"] is True
    assert step.info["spin_request"] == "left"


def test_policy_drive_manual_spin_bypasses_policy_spin_mask(tmp_path: Path) -> None:
    backend = ScriptedPolicyDriveBackend(
        [
            _backend_step(
                frames_run=3,
                race_distance=1_500.0,
                native_step_count=12_345,
            )
        ]
    )
    train_config = _train_config(tmp_path).model_copy(
        update={
            "env": EnvConfig(
                action=configured_discrete_action(
                    "steer",
                    "gas",
                    "boost",
                    "lean",
                    "spin",
                    mask=ActionMaskConfig(spin=(0,)),
                )
            )
        }
    )
    runtime = PolicyDriveRuntime(
        emulator=backend,
        train_config=train_config,
    )

    runtime.begin(seed=7, course_id="mute_city")
    runtime.step_manual(RaceControlState(), spin_request="right")

    assert backend.last_spin_request == "right"


def _backend_step(
    *,
    frames_run: int,
    race_distance: float,
    native_step_count: int,
) -> BackendStepResult:
    return BackendStepResult(
        observation=np.zeros((84, 84, 3), dtype=np.uint8),
        summary=make_step_summary(
            frames_run=frames_run,
            max_race_distance=race_distance,
            final_frame_index=frames_run,
        ),
        status=make_step_status(
            step_count=native_step_count,
            progress_frontier_stalled_frames=9_000,
            truncation_reason="timeout",
        ),
        telemetry=make_telemetry(race_distance=race_distance, speed_kph=760.0),
        display_frames=(),
        display_controller_masks=(),
    )


def _train_config(tmp_path: Path) -> TrainAppConfig:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    return TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action_repeat=3),
    )
