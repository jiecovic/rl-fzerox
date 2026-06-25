# tests/ui/test_viewer_runtime_career_session.py
"""Watch runtime tests for Career Mode session and timing behavior.

The cases keep menu cadence, policy timing, runtime target metadata, snapshot
frame-rate fields, and viewer command timing updates separate from record and
observation tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fzerox_emulator.arrays import UInt8Array
from rl_fzerox.core.runtime_spec.schema import (
    CustomResolutionChoice,
    EmulatorConfig,
    EnvConfig,
    ObservationConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.ui.watch.runtime.career_mode.menu import reset_race_progress_info
from rl_fzerox.ui.watch.runtime.career_mode.session import (
    CareerModeRuntimeSession,
    open_career_mode_runtime_session,
)
from rl_fzerox.ui.watch.runtime.career_mode.timing import (
    active_policy_timing,
    set_session_control_timing,
    snapshot_action_repeat,
    snapshot_target_control_fps,
    target_game_fps,
)
from rl_fzerox.ui.watch.runtime.timing import (
    RateMeter,
    _adjust_control_fps,
    _resolve_control_fps,
    _resolve_render_fps,
)
from rl_fzerox.ui.watch.view.screen.render import _add_career_mode_info


def test_career_mode_render_info_keeps_runtime_target_and_attempt(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    info: dict[str, object] = {
        "career_mode_target_label": "Clear Expert Jack Cup",
        "career_mode_attempt_id": "live-attempt",
    }
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        watch=WatchConfig(
            manager_db_path=db_path,
            managed_save_game_id="save",
            save_attempt_id="launch-attempt",
            unlock_target_label="Clear Novice Joker Cup",
        ),
    )

    _add_career_mode_info(info, config)

    assert info["career_mode_target_label"] == "Clear Expert Jack Cup"
    assert info["career_mode_attempt_id"] == "live-attempt"


def test_career_mode_session_renders_display_without_policy_crop(tmp_path: Path) -> None:
    class _Emulator:
        native_fps = 60.0
        native_sample_rate = 48_000

        def __init__(self) -> None:
            self.render_count = 0
            self.render_display_count = 0

        def render(self) -> UInt8Array:
            self.render_count += 1
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def render_display(
            self,
            *,
            preset: str | None = None,
            height: int | None = None,
            width: int | None = None,
        ) -> UInt8Array:
            _ = (preset, height, width)
            self.render_display_count += 1
            return np.zeros((2, 2, 3), dtype=np.uint8)

    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    emulator: Any = _Emulator()
    session = CareerModeRuntimeSession(
        config=WatchAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(
                observation=ObservationConfig(
                    resolution=CustomResolutionChoice(height=72, width=96),
                ),
            ),
        ),
        emulator=emulator,
        native_fps=60.0,
        native_sample_rate=48_000.0,
        native_control_fps=30.0,
        target_control_fps=30.0,
        target_control_seconds=1.0 / 30.0,
        watch_zeroed_state_features=frozenset(),
        auxiliary_target_names=(),
    )

    session.render()

    assert emulator.render_count == 0
    assert emulator.render_display_count == 1


def test_career_mode_session_seeds_only_from_runtime_attempt_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Emulator:
        native_fps = 60.0
        native_sample_rate = 48_000

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def close(self) -> None:
            pass

    seeds: list[int] = []
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.session.Emulator",
        _Emulator,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.session.seed_process",
        seeds.append,
    )
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()

    config = WatchAppConfig(
        seed=99,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
    )
    session = open_career_mode_runtime_session(config)
    session.close()
    seeded_config = config.model_copy(
        update={"watch": config.watch.model_copy(update={"attempt_seed": 1234})}
    )
    seeded_session = open_career_mode_runtime_session(seeded_config)
    seeded_session.close()

    assert seeds == [1234]


def test_career_mode_session_uses_native_menu_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Emulator:
        native_fps = 60.0
        native_sample_rate = 48_000

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.session.Emulator",
        _Emulator,
    )
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()

    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action_repeat=4),
    )
    session = open_career_mode_runtime_session(config)
    session.close()

    assert session.native_control_fps == 60.0
    assert session.target_control_fps == 60.0
    assert session.target_control_seconds == pytest.approx(1.0 / 60.0)


def test_career_mode_menu_info_clears_policy_race_progress() -> None:
    info = reset_race_progress_info(
        {
            "episode_step": 123,
            "episode_return": 456.0,
            "step_reward": 7.0,
            "progress_frontier_stalled_frames": 8,
            "stalled_steps": 9,
            "frames_run": 10,
            "repeat_index": 1,
            "reward_breakdown": {"progress": 1.0},
        }
    )

    assert info["episode_step"] == 0
    assert info["episode_return"] == 0.0
    assert info["step_reward"] == 0.0
    assert info["progress_frontier_stalled_frames"] == 0
    assert info["stalled_steps"] == 0
    assert info["frames_run"] == 0
    assert info["repeat_index"] == 0
    assert "reward_breakdown" not in info


def test_watch_fps_helpers_resolve_split_control_and_render_rates() -> None:
    assert _resolve_control_fps("auto", native_control_fps=30.0) == 30.0
    assert _resolve_control_fps("unlimited", native_control_fps=30.0) is None
    assert _resolve_control_fps(120.0, native_control_fps=30.0) == 120.0
    assert _resolve_render_fps(None, native_fps=60.0) == 60.0
    assert _resolve_render_fps("auto", native_fps=60.0) == 60.0
    assert _resolve_render_fps("unlimited", native_fps=60.0) is None


def test_watch_control_fps_adjustment_supports_uncapped_mode() -> None:
    assert _adjust_control_fps(60.0, 1, native_control_fps=60.0) == 65.0
    assert _adjust_control_fps(60.0, -1, native_control_fps=60.0) == 55.0
    assert _adjust_control_fps(None, 1, native_control_fps=60.0) is None
    assert _adjust_control_fps(None, -1, native_control_fps=60.0) == 55.0


def test_rate_meter_reset_discards_previous_phase_timing() -> None:
    meter = RateMeter(window=4)
    meter.tick(0.0)
    meter.tick(1.0)

    assert meter.rate_hz() == pytest.approx(1.0)

    meter.reset()

    assert meter.rate_hz() == 0.0
    meter.tick(10.0)
    meter.tick(10.5)
    assert meter.rate_hz() == pytest.approx(2.0)


def test_career_mode_active_policy_timing_preserves_speed_multiplier(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action_repeat=2),
    )
    policy_config = config.model_copy(
        update={"env": config.env.model_copy(update={"action_repeat": 4})}
    )

    class _Session:
        native_fps = 60.0

        @staticmethod
        def snapshot_config(base_config: WatchAppConfig) -> WatchAppConfig:
            return policy_config

    timing = active_policy_timing(
        config,
        _Session(),
        native_control_fps=30.0,
        target_control_fps=60.0,
    )

    assert timing.target_fps == 30.0
    assert timing.target_seconds == pytest.approx(1.0 / 30.0)


def test_career_mode_snapshot_target_control_fps_tracks_current_controller(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action_repeat=2),
    )
    policy_config = config.model_copy(
        update={"env": config.env.model_copy(update={"action_repeat": 4})}
    )

    class _Session:
        native_fps = 60.0

        @staticmethod
        def snapshot_config(base_config: WatchAppConfig) -> WatchAppConfig:
            return policy_config

    assert (
        snapshot_target_control_fps(
            config=config,
            session=_Session(),
            native_control_fps=60.0,
            target_control_fps=60.0,
            policy_active=False,
        )
        == 60.0
    )
    assert (
        snapshot_target_control_fps(
            config=config,
            session=_Session(),
            native_control_fps=60.0,
            target_control_fps=60.0,
            policy_active=True,
        )
        == 15.0
    )


def test_career_mode_target_game_fps_uses_active_action_repeat() -> None:
    assert target_game_fps(target_control_fps=15.0, action_repeat=4) == 60.0
    assert target_game_fps(target_control_fps=None, action_repeat=4) is None


def test_career_mode_menu_snapshots_use_native_frame_repeat(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action_repeat=4),
    )

    assert snapshot_action_repeat(config, policy_active=False) == 1
    assert snapshot_action_repeat(config, policy_active=True) == 4


def test_career_mode_session_timing_updates_with_viewer_commands() -> None:
    class _Session:
        target_control_fps: float | None = 30.0
        target_control_seconds: float | None = 1.0 / 30.0

    session = _Session()

    set_session_control_timing(
        session,
        target_control_fps=60.0,
        target_control_seconds=1.0 / 60.0,
    )

    assert session.target_control_fps == 60.0
    assert session.target_control_seconds == pytest.approx(1.0 / 60.0)
