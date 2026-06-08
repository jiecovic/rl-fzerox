# tests/core/emulator/test_emulator.py
from pathlib import Path

import numpy as np
import pytest

from fzerox_emulator import Emulator, FZeroXTelemetry, ObservationImageRecipe, RaceControlState
from fzerox_emulator.arrays import ObservationFrame
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


def test_emulator_passes_renderer_to_native_backend(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    class NativeStub:
        def __init__(
            self,
            core: str,
            rom: str,
            runtime_dir: str | None,
            baseline_state_path: str | None,
            renderer: str,
        ) -> None:
            assert core == str(core_path.resolve())
            assert rom == str(rom_path.resolve())
            assert runtime_dir is None
            assert baseline_state_path is None
            assert renderer == "gliden64"

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("fzerox_emulator.emulator.client.NativeEmulator", NativeStub)
    try:
        Emulator(core_path=core_path, rom_path=rom_path, renderer="gliden64")
    finally:
        monkeypatch.undo()


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
        def observation_spec(
            self,
            preset: str,
            *,
            height: int | None = None,
            width: int | None = None,
        ) -> dict[str, object]:
            assert preset == "crop_84x84"
            assert height is None
            assert width is None
            return {
                "preset": preset,
                "width": 84,
                "height": 84,
                "channels": 3,
                "display_width": 592,
                "display_height": 444,
            }

        def step_repeat_raw(self, request: dict[str, object]):
            step = request["step"]
            observation_request = request["observation"]
            assert isinstance(step, dict)
            assert isinstance(observation_request, dict)
            assert step["action_repeat"] == 2
            assert step["lean_timer_assist"] is False
            assert step["spin_cooldown_frames"] == 11
            assert observation_request["preset"] == "crop_84x84"
            observation = np.zeros((84, 84, 6), dtype=np.uint8)
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
        control_state=RaceControlState(),
        action_repeat=2,
        preset="crop_84x84",
        frame_stack=2,
        stuck_min_speed_kph=50.0,
        energy_loss_epsilon=0.1,
        max_episode_steps=1_000,
        progress_frontier_stall_limit_frames=900,
        progress_frontier_epsilon=100.0,
        terminate_on_energy_depleted=True,
        spin_cooldown_frames=11,
    )

    assert result.observation.shape == (84, 84, 6)
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
    assert result.status.progress_frontier_stalled_frames == 0
    assert isinstance(result.telemetry, FZeroXTelemetry)
    assert result.telemetry.player.race_distance == 42.0


def test_step_repeat_watch_raw_returns_display_frames() -> None:
    emulator = object.__new__(Emulator)
    emulator._observation_specs = {}

    class NativeStub:
        def observation_spec(
            self,
            preset: str,
            *,
            height: int | None = None,
            width: int | None = None,
        ) -> dict[str, object]:
            assert preset == "crop_84x84"
            assert height is None
            assert width is None
            return {
                "preset": preset,
                "width": 84,
                "height": 84,
                "channels": 3,
                "display_width": 592,
                "display_height": 444,
            }

        def step_repeat_watch_raw(self, request: dict[str, object]):
            step = request["step"]
            observation_request = request["observation"]
            assert isinstance(step, dict)
            assert isinstance(observation_request, dict)
            assert step["action_repeat"] == 2
            assert observation_request["preset"] == "crop_84x84"
            observation = np.zeros((84, 84, 6), dtype=np.uint8)
            display_frames = np.stack(
                (
                    np.full((444, 592, 3), 1, dtype=np.uint8),
                    np.full((444, 592, 3), 2, dtype=np.uint8),
                ),
            )
            display_controller_masks = np.array([1, 2], dtype=np.uint16)
            summary = make_step_summary(frames_run=2, max_race_distance=42.0)
            status = make_step_status(step_count=2, stalled_steps=0)
            telemetry = make_telemetry(race_distance=42.0)
            return observation, display_frames, display_controller_masks, summary, status, telemetry

    emulator.__dict__["_native"] = NativeStub()

    result = emulator.step_repeat_watch_raw(
        control_state=RaceControlState(),
        action_repeat=2,
        preset="crop_84x84",
        frame_stack=2,
        stuck_min_speed_kph=50.0,
        energy_loss_epsilon=0.1,
        max_episode_steps=1_000,
        progress_frontier_stall_limit_frames=900,
        progress_frontier_epsilon=100.0,
        terminate_on_energy_depleted=True,
    )

    assert result.observation.shape == (84, 84, 6)
    assert not isinstance(result.display_frames, tuple)
    assert result.display_frames.shape == (2, 444, 592, 3)
    assert result.display_frames[1][0, 0, 0] == 2
    assert not isinstance(result.display_controller_masks, tuple)
    assert result.display_controller_masks.tolist() == [1, 2]


def test_step_repeat_multi_observation_raw_returns_multiple_validated_views() -> None:
    emulator = object.__new__(Emulator)
    emulator._observation_specs = {}

    class NativeStub:
        def observation_spec(
            self,
            preset: str,
            *,
            height: int | None = None,
            width: int | None = None,
        ) -> dict[str, object]:
            if preset == "crop_84x84":
                return {
                    "preset": preset,
                    "width": 84,
                    "height": 84,
                    "channels": 3,
                    "display_width": 592,
                    "display_height": 444,
                }
            assert preset == ""
            assert height == 72
            assert width == 96
            return {
                "preset": "custom_72x96",
                "width": 96,
                "height": 72,
                "channels": 3,
                "display_width": 592,
                "display_height": 444,
            }

        def step_repeat_multi_observation_raw(self, request: dict[str, object]):
            requests = request["observations"]
            assert isinstance(requests, list)
            first_request = requests[0]
            second_request = requests[1]
            assert isinstance(first_request, dict)
            assert isinstance(second_request, dict)
            assert first_request["preset"] == "crop_84x84"
            assert second_request["preset"] == ""
            assert second_request["height"] == 72
            assert second_request["width"] == 96
            observations = [
                np.zeros((84, 84, 6), dtype=np.uint8),
                np.zeros((72, 96, 1), dtype=np.uint8),
            ]
            summary = make_step_summary(frames_run=2, max_race_distance=42.0)
            status = make_step_status(step_count=2, stalled_steps=0)
            telemetry = make_telemetry(race_distance=42.0)
            return observations, summary, status, telemetry

    emulator.__dict__["_native"] = NativeStub()

    result = emulator.step_repeat_multi_observation_raw(
        control_state=RaceControlState(),
        action_repeat=2,
        observation_recipes=(
            ObservationImageRecipe(preset="crop_84x84", frame_stack=2),
            ObservationImageRecipe(
                height=72,
                width=96,
                frame_stack=1,
                stack_mode="gray",
            ),
        ),
        stuck_min_speed_kph=50.0,
        energy_loss_epsilon=0.1,
        max_episode_steps=1_000,
        progress_frontier_stall_limit_frames=900,
        progress_frontier_epsilon=100.0,
        terminate_on_energy_depleted=True,
    )

    assert len(result.observations) == 2
    assert result.observations[0].shape == (84, 84, 6)
    assert result.observations[1].shape == (72, 96, 1)
    assert result.summary.frames_run == 2
    assert result.telemetry is not None
    assert result.telemetry.player.race_distance == 42.0


def test_render_observation_supports_custom_resolution() -> None:
    emulator = object.__new__(Emulator)
    emulator._observation_specs = {}

    class NativeStub:
        def observation_spec(
            self,
            preset: str,
            *,
            height: int | None = None,
            width: int | None = None,
        ) -> dict[str, object]:
            assert preset == ""
            assert height == 72
            assert width == 96
            return {
                "preset": "custom_72x96",
                "width": 96,
                "height": 72,
                "channels": 3,
                "display_width": 592,
                "display_height": 444,
            }

        def frame_observation(
            self,
            preset: str,
            frame_stack: int,
            options: dict[str, object] | None = None,
        ) -> ObservationFrame:
            assert preset == ""
            assert frame_stack == 2
            assert options is not None
            assert options["height"] == 72
            assert options["width"] == 96
            return np.zeros((72, 96, 6), dtype=np.uint8)

    emulator.__dict__["_native"] = NativeStub()

    frame = emulator.render_observation(height=72, width=96, frame_stack=2)

    assert frame.shape == (72, 96, 6)
