# src/rl_fzerox/core/emulator/emulator.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_fzerox._native import Emulator as NativeEmulator
from rl_fzerox.core.emulator.base import FrameStep, ObservationSpec, ResetState
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.emulator.video import display_size
from rl_fzerox.core.game.telemetry import (
    FZeroXTelemetry,
    PlayerTelemetry,
    TelemetryDecodeError,
    TelemetryUnavailableError,
    read_telemetry,
)


class Emulator:
    """Python wrapper over the native Rust libretro host."""

    def __init__(
        self,
        *,
        core_path: Path,
        rom_path: Path,
        runtime_dir: Path | None = None,
        baseline_state_path: Path | None = None,
        renderer: str = "angrylion",
    ) -> None:
        self._core_path = core_path.resolve()
        self._rom_path = rom_path.resolve()
        self._runtime_dir = runtime_dir.resolve() if runtime_dir is not None else None
        self._baseline_state_path = (
            baseline_state_path.resolve()
            if baseline_state_path is not None
            else None
        )
        self._renderer = renderer
        if self._renderer != "angrylion":
            raise RuntimeError(
                f"Renderer {self._renderer!r} is not supported by the current host. "
                "The libretro embedding still uses the software framebuffer path, "
                "so hardware-render plugins like gliden64 are not wired up yet."
            )
        self._native = NativeEmulator(
            str(self._core_path),
            str(self._rom_path),
            None if self._runtime_dir is None else str(self._runtime_dir),
            (
                None
                if self._baseline_state_path is None
                else str(self._baseline_state_path)
            ),
            self._renderer,
        )
        self._observation_specs: dict[str, ObservationSpec] = {}

    @property
    def name(self) -> str:
        return self._native.name

    @property
    def native_fps(self) -> float:
        return float(self._native.native_fps)

    @property
    def display_aspect_ratio(self) -> float:
        return float(self._native.display_aspect_ratio)

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        height, width, channels = self._native.frame_shape
        return int(height), int(width), int(channels)

    @property
    def display_size(self) -> tuple[int, int]:
        return display_size(self.frame_shape, self.display_aspect_ratio)

    @property
    def frame_index(self) -> int:
        return int(self._native.frame_index)

    @property
    def system_ram_size(self) -> int:
        return int(self._native.system_ram_size)

    @property
    def baseline_kind(self) -> str:
        return str(self._native.baseline_kind)

    def reset(self) -> ResetState:
        """Restore the deterministic episode baseline and return the first frame."""

        self._native.reset()
        return ResetState(
            frame=self.render(),
            info=self._frame_info(),
        )

    def step_frame(self) -> FrameStep:
        """Advance exactly one emulator frame."""

        self.step_frames(1, capture_video=True)
        return FrameStep(
            frame=self.render(),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=self._frame_info(),
        )

    def step_frames(self, count: int, *, capture_video: bool = True) -> None:
        """Advance the emulator by a fixed number of frames."""

        self._native.step_frames(count, capture_video)

    def set_controller_state(self, controller_state: ControllerState) -> None:
        """Set the held controller state used for subsequent frame stepping."""

        state = controller_state.clamped()
        self._native.set_controller_state(
            joypad_mask=state.joypad_mask,
            left_stick_x=state.left_stick_x,
            left_stick_y=state.left_stick_y,
            right_stick_x=state.right_stick_x,
            right_stick_y=state.right_stick_y,
        )

    def save_state(self, path: Path) -> None:
        """Serialize the current emulator state to a savestate file."""

        self._native.save_state(str(path.resolve()))

    def read_system_ram(self, offset: int, length: int) -> bytes:
        """Read a raw slice from the libretro system RAM buffer."""

        return bytes(self._native.read_system_ram(offset, length))

    def capture_current_as_baseline(self, path: Path | None = None) -> None:
        """Promote the current state to the active reset baseline."""

        resolved_path = None if path is None else str(path.resolve())
        self._native.capture_current_as_baseline(resolved_path)

    def render(self) -> np.ndarray:
        """Return the latest raw RGB frame as a NumPy array."""

        frame_bytes = self._native.frame_rgb()
        frame_height, frame_width, channels = self.frame_shape
        frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        expected_size = frame_height * frame_width * channels
        if frame.size != expected_size:
            raise RuntimeError(
                "Unexpected frame size from native emulator: "
                f"expected {expected_size} bytes, got {frame.size}"
            )
        return frame.reshape((frame_height, frame_width, channels))

    def render_display(
        self,
        *,
        preset: str,
    ) -> np.ndarray:
        """Return one native display frame for the requested observation preset."""

        spec = self.observation_spec(preset)
        frame = np.asarray(self._native.frame_display(preset), dtype=np.uint8)
        expected_size = spec.display_height * spec.display_width * 3
        if frame.size != expected_size:
            raise RuntimeError(
                "Unexpected display frame size from native emulator: "
                f"expected {expected_size} bytes, got {frame.size}"
            )
        expected_shape = (spec.display_height, spec.display_width, 3)
        if tuple(int(value) for value in frame.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected display frame shape from native emulator: "
                f"expected {expected_shape!r}, got {tuple(frame.shape)!r}"
            )
        return np.ascontiguousarray(frame)

    def observation_spec(self, preset: str) -> ObservationSpec:
        """Return the resolved native observation spec for one preset."""

        cached = self._observation_specs.get(preset)
        if cached is not None:
            return cached
        spec_data = self._native.observation_spec(preset)
        spec = ObservationSpec(
            preset=str(spec_data["preset"]),
            width=int(spec_data["width"]),
            height=int(spec_data["height"]),
            channels=int(spec_data["channels"]),
            display_width=int(spec_data["display_width"]),
            display_height=int(spec_data["display_height"]),
        )
        self._observation_specs[preset] = spec
        return spec

    def render_observation(self, *, preset: str, frame_stack: int) -> np.ndarray:
        """Return one native stacked observation tensor for the requested preset."""

        spec = self.observation_spec(preset)
        frame = np.asarray(self._native.frame_observation(preset, frame_stack), dtype=np.uint8)
        stacked_channels = spec.channels * frame_stack
        expected_size = spec.height * spec.width * stacked_channels
        if frame.size != expected_size:
            raise RuntimeError(
                "Unexpected observation size from native emulator: "
                f"expected {expected_size} bytes, got {frame.size}"
            )
        expected_shape = (spec.height, spec.width, stacked_channels)
        if tuple(int(value) for value in frame.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected observation frame shape from native emulator: "
                f"expected {expected_shape!r}, got {tuple(frame.shape)!r}"
            )
        return np.ascontiguousarray(frame)

    def telemetry_data(self) -> dict[str, object]:
        """Return the latest structured telemetry mapping from the native host."""

        data = self._native.telemetry()
        if not isinstance(data, dict):
            raise TelemetryDecodeError("Native telemetry did not resolve to a mapping")
        return data

    def try_read_telemetry(self) -> FZeroXTelemetry | None:
        """Return the latest telemetry snapshot, if the host can decode it."""

        try:
            native = getattr(self, "_native", None)
            if native is not None:
                return _telemetry_from_flat_tuple(native.telemetry_flat())
            return read_telemetry(self)
        except TelemetryUnavailableError:
            return None

    def close(self) -> None:
        """Release the native emulator host."""

        self._native.close()

    def _frame_info(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "frame_index": self.frame_index,
            "core_path": str(self._core_path),
            "rom_path": str(self._rom_path),
            "runtime_dir": None if self._runtime_dir is None else str(self._runtime_dir),
            "baseline_state_path": (
                None
                if self._baseline_state_path is None
                else str(self._baseline_state_path)
            ),
            "baseline_kind": self.baseline_kind,
            "renderer": self._renderer,
            "display_aspect_ratio": self.display_aspect_ratio,
            "native_fps": self.native_fps,
        }


def _telemetry_from_flat_tuple(data: object) -> FZeroXTelemetry:
    if not isinstance(data, tuple) or len(data) != 7:
        raise TelemetryDecodeError("Native telemetry payload must be a 7-item tuple")

    (
        system_ram_size,
        game_frame_count,
        game_mode_raw,
        game_mode_name,
        course_index,
        in_race_mode,
        player_data,
    ) = data
    if not isinstance(player_data, tuple) or len(player_data) != 17:
        raise TelemetryDecodeError("Native player telemetry payload must be a 17-item tuple")

    (
        state_flags,
        state_labels,
        speed_raw,
        speed_kph,
        energy,
        max_energy,
        boost_timer,
        race_distance,
        laps_completed_distance,
        lap_distance,
        race_distance_position,
        race_time_ms,
        lap,
        laps_completed,
        position,
        character,
        machine_index,
    ) = player_data

    if not isinstance(state_labels, tuple) or not all(
        isinstance(label, str) for label in state_labels
    ):
        raise TelemetryDecodeError("Native player telemetry labels must be a tuple[str, ...]")

    return FZeroXTelemetry(
        system_ram_size=_int_field(system_ram_size, "system_ram_size"),
        game_frame_count=_int_field(game_frame_count, "game_frame_count"),
        game_mode_raw=_int_field(game_mode_raw, "game_mode_raw"),
        game_mode_name=_str_field(game_mode_name, "game_mode_name"),
        course_index=_int_field(course_index, "course_index"),
        in_race_mode=_bool_field(in_race_mode, "in_race_mode"),
        player=PlayerTelemetry(
            state_flags=_int_field(state_flags, "player.state_flags"),
            state_labels=tuple(state_labels),
            speed_raw=_float_field(speed_raw, "player.speed_raw"),
            speed_kph=_float_field(speed_kph, "player.speed_kph"),
            energy=_float_field(energy, "player.energy"),
            max_energy=_float_field(max_energy, "player.max_energy"),
            boost_timer=_int_field(boost_timer, "player.boost_timer"),
            race_distance=_float_field(race_distance, "player.race_distance"),
            laps_completed_distance=_float_field(
                laps_completed_distance, "player.laps_completed_distance"
            ),
            lap_distance=_float_field(lap_distance, "player.lap_distance"),
            race_distance_position=_float_field(
                race_distance_position, "player.race_distance_position"
            ),
            race_time_ms=_int_field(race_time_ms, "player.race_time_ms"),
            lap=_int_field(lap, "player.lap"),
            laps_completed=_int_field(laps_completed, "player.laps_completed"),
            position=_int_field(position, "player.position"),
            character=_int_field(character, "player.character"),
            machine_index=_int_field(machine_index, "player.machine_index"),
        ),
    )


def _int_field(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TelemetryDecodeError(f"Native telemetry field {field_name!r} must be an int")
    return value


def _float_field(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TelemetryDecodeError(f"Native telemetry field {field_name!r} must be numeric")
    return float(value)


def _bool_field(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TelemetryDecodeError(f"Native telemetry field {field_name!r} must be a bool")
    return value


def _str_field(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TelemetryDecodeError(f"Native telemetry field {field_name!r} must be a str")
    return value
