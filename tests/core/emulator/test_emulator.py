# tests/core/emulator/test_emulator.py
from pathlib import Path

import numpy as np
import pytest

from fzerox_emulator import ControllerState, Emulator, FZeroXTelemetry
from tests.support.native_objects import make_step_status, make_step_summary, make_telemetry


def test_emulator_rejects_missing_core(tmp_path: Path) -> None:
    missing_core = tmp_path / "missing_core.so"
    rom_path = tmp_path / "fzerox.n64"
    rom_path.touch()

    with pytest.raises(FileNotFoundError, match="Libretro core not found"):
        Emulator(core_path=missing_core, rom_path=rom_path)


def test_emulator_rejects_missing_rom(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    missing_rom = tmp_path / "missing_rom.n64"
    core_path.touch()

    with pytest.raises(FileNotFoundError, match="ROM not found"):
        Emulator(core_path=core_path, rom_path=missing_rom)


def test_emulator_rejects_unsupported_renderer(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    with pytest.raises(RuntimeError, match="gliden64"):
        Emulator(core_path=core_path, rom_path=rom_path, renderer="gliden64")


def test_try_read_telemetry_returns_none_when_native_binding_reports_no_snapshot() -> None:
    emulator = object.__new__(Emulator)
    emulator.__dict__["_native"] = type(
        "NativeStub",
        (),
        {"telemetry": staticmethod(lambda: None)},
    )()

    assert emulator.try_read_telemetry() is None


def test_try_read_telemetry_returns_native_snapshot_object() -> None:
    telemetry = make_telemetry(race_distance=42.0)
    emulator = object.__new__(Emulator)
    emulator.__dict__["_native"] = type(
        "NativeStub",
        (),
        {"telemetry": staticmethod(lambda: telemetry)},
    )()

    assert emulator.try_read_telemetry() is telemetry


def test_step_repeat_raw_returns_native_summary_and_telemetry_objects() -> None:
    emulator = object.__new__(Emulator)
    emulator._observation_specs = {}

    class NativeStub:
        def observation_spec(self, preset: str) -> dict[str, object]:
            assert preset == "native_crop_v1"
            return {
                "preset": preset,
                "width": 222,
                "height": 78,
                "channels": 3,
                "display_width": 592,
                "display_height": 444,
            }

        def step_repeat_raw(self, **kwargs):
            assert kwargs["action_repeat"] == 2
            observation = np.zeros((78, 222, 6), dtype=np.uint8)
            summary = make_step_summary(
                frames_run=2,
                max_race_distance=42.0,
                reverse_active_frames=1,
                low_speed_frames=1,
                energy_loss_total=4.0,
                energy_gain_total=1.5,
                consecutive_low_speed_frames=2,
                final_frame_index=12,
            )
            status = make_step_status(step_count=12, stalled_steps=2, reverse_timer=75)
            telemetry = make_telemetry(
                race_distance=42.0,
                race_time_ms=5000,
                position=10,
                reverse_timer=75,
            )
            return observation, summary, status, telemetry

    emulator.__dict__["_native"] = NativeStub()

    result = emulator.step_repeat_raw(
        controller_state=ControllerState(),
        action_repeat=2,
        preset="native_crop_v1",
        frame_stack=2,
        stuck_min_speed_kph=50.0,
        energy_loss_epsilon=0.1,
        max_episode_steps=1_000,
        stuck_step_limit=240,
        wrong_way_timer_limit=180,
    )

    assert result.observation.shape == (78, 222, 6)
    assert result.summary.frames_run == 2
    assert result.summary.max_race_distance == 42.0
    assert result.summary.reverse_active_frames == 1
    assert result.summary.low_speed_frames == 1
    assert result.summary.energy_loss_total == 4.0
    assert result.summary.energy_gain_total == 1.5
    assert result.summary.final_frame_index == 12
    assert result.status.step_count == 12
    assert result.status.stalled_steps == 2
    assert result.status.reverse_timer == 75
    assert isinstance(result.telemetry, FZeroXTelemetry)
    assert result.telemetry.player.race_distance == 42.0
