# tests/ui/test_career_mode_loop_runtime.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_fzerox.core.runtime_spec.schema import EmulatorConfig, WatchAppConfig, WatchConfig
from rl_fzerox.ui.watch.runtime.career_mode.loop.runtime import (
    apply_career_attempt_menu_jitter,
    policy_intro_wait_required,
)
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
