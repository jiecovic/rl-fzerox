# tests/ui/test_career_mode_loop_runtime.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fzerox_emulator import RaceControlState
from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.runtime_spec.schema import EmulatorConfig, WatchAppConfig, WatchConfig
from rl_fzerox.ui.watch.runtime.career_mode.loop.policy_runtime import (
    CareerPolicyRuntime,
)
from rl_fzerox.ui.watch.runtime.career_mode.loop.runtime import (
    apply_career_attempt_menu_jitter,
    policy_intro_wait_required,
)
from rl_fzerox.ui.watch.runtime.career_mode.policy_step import CareerPolicyStepResult
from rl_fzerox.ui.watch.runtime.career_mode.session import CareerModeRuntimeSession


class _JitterEmulator:
    def __init__(self) -> None:
        self.step_frame_calls: list[tuple[int, bool]] = []

    def step_frames(self, count: int, *, capture_video: bool = True) -> None:
        self.step_frame_calls.append((count, capture_video))

    def randomize_game_rng(self, seed: int) -> tuple[int, int, int, int]:
        raise AssertionError(f"Career attempt jitter must not patch game RNG: {seed}")


class _JitterController:
    def active_attempt_id(self) -> str:
        return "attempt-1"


def test_apply_career_attempt_menu_jitter_uses_neutral_frames_only(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    emulator: Any = _JitterEmulator()
    session = CareerModeRuntimeSession(
        config=WatchAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            watch=WatchConfig(attempt_seed=1_234),
        ),
        emulator=emulator,
        native_fps=60.0,
        native_sample_rate=48_000.0,
        native_control_fps=60.0,
        target_control_fps=60.0,
        target_control_seconds=1.0 / 60.0,
        watch_zeroed_state_features=frozenset(),
        auxiliary_target_names=(),
    )
    controller: Any = _JitterController()

    jitter_frames = apply_career_attempt_menu_jitter(
        config=session.config,
        session=session,
        controller=controller,
    )

    assert jitter_frames == 67
    assert emulator.step_frame_calls == [(67, False)]


def test_policy_intro_wait_required_until_training_target() -> None:
    assert policy_intro_wait_required(
        info={"race_intro_timer": 80},
        target_timer=39,
    )
    assert not policy_intro_wait_required(
        info={"race_intro_timer": 39},
        target_timer=39,
    )
    assert not policy_intro_wait_required(
        info={"race_intro_timer": 10},
        target_timer=39,
    )


def test_policy_intro_wait_required_skips_missing_or_disabled_target() -> None:
    assert not policy_intro_wait_required(
        info={"race_intro_timer": 80},
        target_timer=None,
    )
    assert not policy_intro_wait_required(
        info={},
        target_timer=39,
    )
    assert not policy_intro_wait_required(
        info={"race_intro_timer": "80"},
        target_timer=39,
    )


def test_career_policy_runtime_clear_and_menu_reset_keep_distinct_semantics() -> None:
    runtime = _policy_runtime(started=True, manual_enabled=True, episode_reward=12.5)

    runtime.reset_menu_controls()

    assert runtime.started is True
    assert runtime.manual_enabled is True
    assert runtime.episode_reward == 0.0
    assert runtime.control_state == RaceControlState()
    assert runtime.gas_level == 0.0
    assert runtime.boost_lamp_level == 0.0

    runtime.clear(reset_episode_reward=False)

    assert runtime.started is False
    assert runtime.manual_enabled is False
    assert runtime.policy_action is None
    assert runtime.episode_reward == 0.0


def test_career_policy_runtime_applies_policy_step_result() -> None:
    runtime = _policy_runtime()
    control_state = RaceControlState.from_mask(1, stick_x=0.5, pitch=-0.25)

    runtime.apply_policy_step(
        CareerPolicyStepResult(
            raw_observation=_observation(),
            raw_info={"raw": True},
            observation=_observation(),
            info={"episode_return": 42.0},
            episode_reward=42.0,
            control_state=control_state,
            gas_level=0.75,
            boost_lamp_level=0.5,
            policy_action=3,
            cnn_activations=None,
            telemetry=None,
            auxiliary_predictions={"speed": 1.0},
            auxiliary_targets={"speed": 2.0},
            last_live_series_publish_time=123.0,
        )
    )

    assert runtime.episode_reward == 42.0
    assert runtime.control_state == control_state
    assert runtime.gas_level == 0.75
    assert runtime.boost_lamp_level == 0.5
    assert runtime.policy_action == 3
    assert runtime.auxiliary_predictions == {"speed": 1.0}
    assert runtime.auxiliary_targets == {"speed": 2.0}


def _policy_runtime(
    *,
    started: bool = False,
    manual_enabled: bool = False,
    episode_reward: float = 0.0,
) -> CareerPolicyRuntime:
    return CareerPolicyRuntime(
        policy_control=None,
        started=started,
        manual_enabled=manual_enabled,
        policy_action=7,
        control_state=RaceControlState.from_mask(1, stick_x=1.0, pitch=-1.0),
        gas_level=1.0,
        boost_lamp_level=1.0,
        episode_reward=episode_reward,
        cnn_activations=None,
        auxiliary_predictions={"old": 1.0},
        auxiliary_targets={"old": 2.0},
    )


def _observation() -> RgbFrame:
    frame: RgbFrame = np.zeros((2, 2, 3), dtype=np.uint8)
    return frame
