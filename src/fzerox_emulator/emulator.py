# src/fzerox_emulator/emulator.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from fzerox_emulator._native import Emulator as NativeEmulator
from fzerox_emulator._native import FZeroXTelemetry
from fzerox_emulator.arrays import ObservationFrame, RgbFrame
from fzerox_emulator.base import (
    BackendStepResult,
    FrameStep,
    ObservationSpec,
    ObservationStackMode,
    ResetState,
    stacked_observation_channels,
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

    def game_rng_state(self) -> tuple[int, int, int, int]:
        """Return the live F-Zero X RNG globals from system RAM."""

        seed1, mask1, seed2, mask2 = self._native.game_rng_state()
        return int(seed1), int(mask1), int(seed2), int(mask2)

    def randomize_game_rng(self, seed: int) -> tuple[int, int, int, int]:
        """Patch the live F-Zero X RNG globals using a deterministic seed."""

        normalized_seed = seed & ((1 << 64) - 1)
        seed1, mask1, seed2, mask2 = self._native.randomize_game_rng(normalized_seed)
        return int(seed1), int(mask1), int(seed2), int(mask2)

    def step_repeat_raw(
        self,
        controller_state: ControllerState,
        *,
        action_repeat: int,
        preset: str,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        stuck_step_limit: int,
        wrong_way_timer_limit: int | None,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
    ) -> BackendStepResult:
        """Execute one repeated env step natively and return the final payload."""

        state = controller_state.clamped()
        observation, summary, status, telemetry = self._native.step_repeat_raw(
            action_repeat=action_repeat,
            preset=preset,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
            stuck_min_speed_kph=stuck_min_speed_kph,
            energy_loss_epsilon=energy_loss_epsilon,
            max_episode_steps=max_episode_steps,
            stuck_step_limit=stuck_step_limit,
            wrong_way_timer_limit=wrong_way_timer_limit,
            progress_frontier_stall_limit_frames=progress_frontier_stall_limit_frames,
            progress_frontier_epsilon=progress_frontier_epsilon,
            terminate_on_energy_depleted=terminate_on_energy_depleted,
            lean_timer_assist=lean_timer_assist,
            joypad_mask=state.joypad_mask,
            left_stick_x=state.left_stick_x,
            left_stick_y=state.left_stick_y,
            right_stick_x=state.right_stick_x,
            right_stick_y=state.right_stick_y,
        )
        frame = np.asarray(observation, dtype=np.uint8)
        spec = self.observation_spec(preset)
        stacked_channels = stacked_observation_channels(
            spec.channels,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
        )
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

    def step_repeat_watch_raw(
        self,
        controller_state: ControllerState,
        *,
        action_repeat: int,
        preset: str,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        stuck_step_limit: int,
        wrong_way_timer_limit: int | None,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
    ) -> BackendStepResult:
        """Execute one repeated watch step and return per-frame display images."""

        state = controller_state.clamped()
        observation, display_frames, summary, status, telemetry = (
            self._native.step_repeat_watch_raw(
                action_repeat=action_repeat,
                preset=preset,
                frame_stack=frame_stack,
                stack_mode=stack_mode,
                stuck_min_speed_kph=stuck_min_speed_kph,
                energy_loss_epsilon=energy_loss_epsilon,
                max_episode_steps=max_episode_steps,
                stuck_step_limit=stuck_step_limit,
                wrong_way_timer_limit=wrong_way_timer_limit,
                progress_frontier_stall_limit_frames=progress_frontier_stall_limit_frames,
                progress_frontier_epsilon=progress_frontier_epsilon,
                terminate_on_energy_depleted=terminate_on_energy_depleted,
                lean_timer_assist=lean_timer_assist,
                joypad_mask=state.joypad_mask,
                left_stick_x=state.left_stick_x,
                left_stick_y=state.left_stick_y,
                right_stick_x=state.right_stick_x,
                right_stick_y=state.right_stick_y,
            )
        )
        frame = np.asarray(observation, dtype=np.uint8)
        spec = self.observation_spec(preset)
        stacked_channels = stacked_observation_channels(
            spec.channels,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
        )
        expected_observation_shape = (spec.height, spec.width, stacked_channels)
        if tuple(int(value) for value in frame.shape) != expected_observation_shape:
            raise RuntimeError(
                "Unexpected repeated-step observation shape from native emulator: "
                f"expected {expected_observation_shape!r}, got {tuple(frame.shape)!r}"
            )

        expected_display_shape = (spec.display_height, spec.display_width, 3)
        validated_display_frames = tuple(
            _validated_display_frame(display_frame, expected_shape=expected_display_shape)
            for display_frame in display_frames
        )
        if len(validated_display_frames) != action_repeat:
            raise RuntimeError(
                "Unexpected display frame count from native watch step: "
                f"expected {action_repeat}, got {len(validated_display_frames)}"
            )
        return BackendStepResult(
            observation=np.ascontiguousarray(frame),
            summary=summary,
            status=status,
            telemetry=telemetry,
            display_frames=validated_display_frames,
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

    def load_baseline(self, path: Path) -> None:
        """Replace the active reset baseline with a savestate file."""

        resolved_path = path.resolve()
        self._native.load_baseline(str(resolved_path))
        self._baseline_state_path = resolved_path

    def load_baseline_bytes(self, state: bytes, *, source_path: Path | None = None) -> None:
        """Replace the active reset baseline from already-loaded savestate bytes."""

        self._native.load_baseline_bytes(state)
        if source_path is not None:
            self._baseline_state_path = source_path.resolve()

    def capture_current_as_baseline(self, path: Path | None = None) -> None:
        """Promote the current state to the active reset baseline."""

        resolved_path = None if path is None else str(path.resolve())
        self._native.capture_current_as_baseline(resolved_path)
        if path is not None:
            self._baseline_state_path = path.resolve()

    def render(self) -> RgbFrame:
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
    ) -> RgbFrame:
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

    def render_observation(
        self,
        *,
        preset: str,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
    ) -> ObservationFrame:
        """Return one native stacked observation tensor for the requested preset."""

        spec = self.observation_spec(preset)
        frame = np.asarray(
            self._native.frame_observation(preset, frame_stack, stack_mode),
            dtype=np.uint8,
        )
        stacked_channels = stacked_observation_channels(
            spec.channels,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
        )
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


def _validated_display_frame(frame: object, *, expected_shape: tuple[int, int, int]) -> RgbFrame:
    display_frame = np.asarray(frame, dtype=np.uint8)
    if tuple(int(value) for value in display_frame.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected display frame shape from native watch step: "
            f"expected {expected_shape!r}, got {tuple(display_frame.shape)!r}"
        )
    return np.ascontiguousarray(display_frame)
