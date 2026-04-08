# tests/core/emulator/test_emulator.py
from pathlib import Path

import numpy as np
import pytest

from rl_fzerox.core.emulator import Emulator
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.game import TelemetryDecodeError, TelemetryUnavailableError


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


def test_try_read_telemetry_returns_none_only_for_unavailable(monkeypatch) -> None:
    emulator = object.__new__(Emulator)

    monkeypatch.setattr(
        "rl_fzerox.core.emulator.emulator.read_telemetry",
        lambda _: (_ for _ in ()).throw(TelemetryUnavailableError("missing")),
    )

    assert emulator.try_read_telemetry() is None


def test_try_read_telemetry_propagates_decode_errors(monkeypatch) -> None:
    emulator = object.__new__(Emulator)

    monkeypatch.setattr(
        "rl_fzerox.core.emulator.emulator.read_telemetry",
        lambda _: (_ for _ in ()).throw(TelemetryDecodeError("bad payload")),
    )

    with pytest.raises(TelemetryDecodeError, match="bad payload"):
        emulator.try_read_telemetry()


def test_step_repeat_raw_decodes_native_summary_and_observation() -> None:
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
            summary = (2, 42.0, 3.5, 1, 4.0, 2, 0, 12)
            telemetry = (
                0x00800000,
                120,
                1,
                "gp_race",
                0,
                True,
                (
                    1 << 30,
                    ("active",),
                    0.0,
                    100.0,
                    178.0,
                    178.0,
                    0,
                    42.0,
                    0.0,
                    42.0,
                    42.0,
                    5000,
                    1,
                    0,
                    10,
                    0,
                    0,
                ),
            )
            return observation, summary, telemetry

    emulator.__dict__["_native"] = NativeStub()

    result = emulator.step_repeat_raw(
        controller_state=ControllerState(),
        action_repeat=2,
        preset="native_crop_v1",
        frame_stack=2,
        stuck_min_speed_kph=50.0,
        reverse_progress_epsilon=0.5,
        energy_loss_epsilon=0.1,
        wrong_way_progress_epsilon=2.0,
    )

    assert result.observation.shape == (78, 222, 6)
    assert result.summary.frames_run == 2
    assert result.summary.max_race_distance == 42.0
    assert result.summary.reverse_progress_total == 3.5
    assert result.summary.energy_loss_total == 4.0
    assert result.summary.final_frame_index == 12
    assert result.telemetry is not None
    assert result.telemetry.player.race_distance == 42.0
