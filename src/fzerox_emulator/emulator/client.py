# src/fzerox_emulator/emulator/client.py
"""Concrete Python wrapper around the native Rust libretro host."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from fzerox_emulator._native import Emulator as NativeEmulator
from fzerox_emulator._native import FZeroXTelemetry
from fzerox_emulator.base.observations import (
    ObservationImageRecipe,
    ObservationResizeFilter,
    ObservationSpec,
    ObservationStackMode,
)
from fzerox_emulator.base.results import (
    BackendMultiObservationStepResult,
    BackendStepResult,
    FrameStep,
    ResetState,
)
from fzerox_emulator.control import ControllerState, RaceControlState
from fzerox_emulator.control.spin import SpinRequest
from fzerox_emulator.emulator.observations import ObservationRenderingMixin
from fzerox_emulator.emulator.race_start import RaceStartMixin
from fzerox_emulator.repeat import (
    RepeatStepConfig,
    run_repeat_multi_observation_step,
    run_repeat_step,
    run_repeat_watch_step,
)

_DEFAULT_RENDERER = "gliden64"


class Emulator(RaceStartMixin, ObservationRenderingMixin):
    """High-level emulator object exposed to training, watch, and tooling code."""

    def __init__(
        self,
        *,
        core_path: Path,
        rom_path: Path,
        runtime_dir: Path | None = None,
        baseline_state_path: Path | None = None,
        renderer: str = _DEFAULT_RENDERER,
    ) -> None:
        self._core_path = core_path.resolve()
        self._rom_path = rom_path.resolve()
        self._runtime_dir = runtime_dir.resolve() if runtime_dir is not None else None
        self._baseline_state_path = (
            baseline_state_path.resolve() if baseline_state_path is not None else None
        )
        self._renderer = renderer
        if not self._core_path.is_file():
            raise FileNotFoundError(f"Libretro core not found: {self._core_path}")
        if not self._rom_path.is_file():
            raise FileNotFoundError(f"ROM not found: {self._rom_path}")
        self._native = NativeEmulator(
            str(self._core_path),
            str(self._rom_path),
            None if self._runtime_dir is None else str(self._runtime_dir),
            (None if self._baseline_state_path is None else str(self._baseline_state_path)),
            self._renderer,
        )
        self._observation_specs: dict[
            tuple[str | None, int | None, int | None],
            ObservationSpec,
        ] = {}

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
    def renderer(self) -> str:
        return self._renderer

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        height, width, channels = self._native.frame_shape
        return int(height), int(width), int(channels)

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

    def read_system_ram(self, offset: int, length: int) -> bytes:
        """Read raw N64 system RAM bytes for reverse-engineering probes."""

        return bytes(self._native.read_system_ram(offset, length))

    def write_system_ram(self, offset: int, data: bytes) -> None:
        """Write raw N64 system RAM bytes for controlled reverse-engineering probes."""

        self._native.write_system_ram(offset, data)

    def step_repeat_raw(
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
        resize_filter: ObservationResizeFilter = "nearest",
        minimap_resize_filter: ObservationResizeFilter = "nearest",
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
        spin_request: SpinRequest = "none",
    ) -> BackendStepResult:
        """Execute one repeated env step natively and return the final payload."""

        config = RepeatStepConfig(
            action_repeat=action_repeat,
            stuck_min_speed_kph=stuck_min_speed_kph,
            energy_loss_epsilon=energy_loss_epsilon,
            max_episode_steps=max_episode_steps,
            progress_frontier_stall_limit_frames=progress_frontier_stall_limit_frames,
            progress_frontier_epsilon=progress_frontier_epsilon,
            terminate_on_energy_depleted=terminate_on_energy_depleted,
            lean_timer_assist=lean_timer_assist,
            spin_request=spin_request,
        )
        recipe = ObservationImageRecipe(
            preset=preset,
            height=height,
            width=width,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
            minimap_layer=minimap_layer,
            resize_filter=resize_filter,
            minimap_resize_filter=minimap_resize_filter,
        )
        return run_repeat_step(
            self._native,
            control_state,
            config=config,
            recipe=recipe,
        )

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
        resize_filter: ObservationResizeFilter = "nearest",
        minimap_resize_filter: ObservationResizeFilter = "nearest",
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
        spin_request: SpinRequest = "none",
    ) -> BackendStepResult:
        """Execute one repeated watch step and return batched display images."""

        config = RepeatStepConfig(
            action_repeat=action_repeat,
            stuck_min_speed_kph=stuck_min_speed_kph,
            energy_loss_epsilon=energy_loss_epsilon,
            max_episode_steps=max_episode_steps,
            progress_frontier_stall_limit_frames=progress_frontier_stall_limit_frames,
            progress_frontier_epsilon=progress_frontier_epsilon,
            terminate_on_energy_depleted=terminate_on_energy_depleted,
            lean_timer_assist=lean_timer_assist,
            spin_request=spin_request,
        )
        recipe = ObservationImageRecipe(
            preset=preset,
            height=height,
            width=width,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
            minimap_layer=minimap_layer,
            resize_filter=resize_filter,
            minimap_resize_filter=minimap_resize_filter,
        )
        return run_repeat_watch_step(
            self._native,
            control_state,
            config=config,
            recipe=recipe,
        )

    def step_repeat_multi_observation_raw(
        self,
        control_state: RaceControlState,
        *,
        action_repeat: int,
        observation_recipes: Sequence[ObservationImageRecipe],
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
        spin_request: SpinRequest = "none",
    ) -> BackendMultiObservationStepResult:
        """Execute one repeated env step and return multiple observation views."""

        config = RepeatStepConfig(
            action_repeat=action_repeat,
            stuck_min_speed_kph=stuck_min_speed_kph,
            energy_loss_epsilon=energy_loss_epsilon,
            max_episode_steps=max_episode_steps,
            progress_frontier_stall_limit_frames=progress_frontier_stall_limit_frames,
            progress_frontier_epsilon=progress_frontier_epsilon,
            terminate_on_energy_depleted=terminate_on_energy_depleted,
            lean_timer_assist=lean_timer_assist,
            spin_request=spin_request,
        )
        return run_repeat_multi_observation_step(
            self._native,
            control_state,
            config=config,
            recipes=observation_recipes,
        )

    def set_controller_state(self, controller_state: ControllerState) -> None:
        """Set the held controller state used for subsequent frame stepping."""

        self._native.set_controller_state(
            joypad_mask=controller_state.joypad_mask,
            left_stick_x=controller_state.left_stick_x,
            left_stick_y=controller_state.left_stick_y,
            right_stick_x=controller_state.right_stick_x,
            right_stick_y=controller_state.right_stick_y,
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
