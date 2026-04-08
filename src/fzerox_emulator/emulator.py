# src/fzerox_emulator/emulator.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from fzerox_emulator._native import Emulator as NativeEmulator
from fzerox_emulator._native import FZeroXTelemetry
from fzerox_emulator.base import (
    BackendStepResult,
    FrameStep,
    ObservationSpec,
    ResetState,
)
from fzerox_emulator.control import ControllerState
from fzerox_emulator.video import display_size


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
            baseline_state_path.resolve() if baseline_state_path is not None else None
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
            (None if self._baseline_state_path is None else str(self._baseline_state_path)),
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

    def step_repeat_raw(
        self,
        controller_state: ControllerState,
        *,
        action_repeat: int,
        preset: str,
        frame_stack: int,
        stuck_min_speed_kph: float,
        reverse_progress_epsilon: float,
        energy_loss_epsilon: float,
        wrong_way_progress_epsilon: float,
        max_episode_steps: int,
        stuck_step_limit: int,
        wrong_way_step_limit: int,
    ) -> BackendStepResult:
        """Execute one repeated env step natively and return the final payload."""

        state = controller_state.clamped()
        observation, summary, status, telemetry = self._native.step_repeat_raw(
            action_repeat=action_repeat,
            preset=preset,
            frame_stack=frame_stack,
            stuck_min_speed_kph=stuck_min_speed_kph,
            reverse_progress_epsilon=reverse_progress_epsilon,
            energy_loss_epsilon=energy_loss_epsilon,
            wrong_way_progress_epsilon=wrong_way_progress_epsilon,
            max_episode_steps=max_episode_steps,
            stuck_step_limit=stuck_step_limit,
            wrong_way_step_limit=wrong_way_step_limit,
            joypad_mask=state.joypad_mask,
            left_stick_x=state.left_stick_x,
            left_stick_y=state.left_stick_y,
            right_stick_x=state.right_stick_x,
            right_stick_y=state.right_stick_y,
        )
        frame = np.asarray(observation, dtype=np.uint8)
        spec = self.observation_spec(preset)
        stacked_channels = spec.channels * frame_stack
        expected_shape = (spec.height, spec.width, stacked_channels)
        if tuple(int(value) for value in frame.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected repeated-step observation shape from native emulator: "
                f"expected {expected_shape!r}, got {tuple(frame.shape)!r}"
            )
        return BackendStepResult(
            observation=np.ascontiguousarray(frame),
            summary=summary,
            status=status,
            telemetry=telemetry,
        )

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

    def try_read_telemetry(self) -> FZeroXTelemetry | None:
        """Return the latest telemetry snapshot from the native host, if available."""
        return self._native.telemetry()

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
                None if self._baseline_state_path is None else str(self._baseline_state_path)
            ),
            "baseline_kind": self.baseline_kind,
            "renderer": self._renderer,
            "display_aspect_ratio": self.display_aspect_ratio,
            "native_fps": self.native_fps,
        }
