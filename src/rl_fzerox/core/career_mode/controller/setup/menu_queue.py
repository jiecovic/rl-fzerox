# src/rl_fzerox/core/career_mode/controller/setup/menu_queue.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rl_fzerox.core.career_mode.navigation import (
    MENU_TIMING,
    MenuInput,
    RawMenuStep,
    raw_step,
)


@dataclass(slots=True)
class CareerMenuStepQueue:
    """Own queued controller button presses and per-phase advance retries."""

    pending_steps: deque[RawMenuStep] = field(default_factory=deque)
    advance_presses_in_phase: int = 0

    def __bool__(self) -> bool:
        return bool(self.pending_steps)

    @property
    def pending_count(self) -> int:
        return len(self.pending_steps)

    def peek(self) -> RawMenuStep | None:
        if not self.pending_steps:
            return None
        return self.pending_steps[0]

    def pop(self) -> RawMenuStep:
        return self.pending_steps.popleft()

    def append(self, step: RawMenuStep) -> None:
        self.pending_steps.append(step)

    def extend(self, steps: tuple[RawMenuStep, ...]) -> None:
        self.pending_steps.extend(steps)

    def clear(self) -> None:
        self.pending_steps.clear()

    def reset_advance_presses(self) -> None:
        self.advance_presses_in_phase = 0

    def queue_tap(
        self,
        menu_input: MenuInput,
        *,
        hold_frames: int,
        settle_frames: int,
        phase: str,
    ) -> RawMenuStep:
        self.pending_steps.append(
            raw_step(MenuInput.NEUTRAL, settle_frames, phase=f"{phase}:settle")
        )
        return raw_step(menu_input, hold_frames, phase=phase)

    def queue_menu_steps(self, steps: tuple[RawMenuStep, ...]) -> RawMenuStep:
        if not steps:
            return raw_step(MenuInput.NEUTRAL, 1, phase="menu_steps:empty")
        self.pending_steps.extend(steps[1:])
        return steps[0]

    def start_until_phase(self, phase: str) -> RawMenuStep:
        return self.tap_until_phase(
            MenuInput.START,
            hold_frames=MENU_TIMING.start_hold_frames,
            settle_frames=MENU_TIMING.start_settle_frames,
            phase=phase,
            label="start",
        )

    def accept_until_phase(self, phase: str) -> RawMenuStep:
        return self.tap_until_phase(
            MenuInput.ACCEPT,
            hold_frames=MENU_TIMING.start_hold_frames,
            settle_frames=MENU_TIMING.start_settle_frames,
            phase=phase,
            label="accept",
        )

    def tap_until_phase(
        self,
        menu_input: MenuInput,
        *,
        hold_frames: int,
        settle_frames: int,
        phase: str,
        label: str,
    ) -> RawMenuStep:
        self.advance_presses_in_phase += 1
        if self.advance_presses_in_phase > MENU_TIMING.max_advance_presses_per_phase:
            raise RuntimeError(
                f"Career Mode menu phase {phase!r} did not reach the expected screen"
            )
        return self.queue_tap(
            menu_input,
            hold_frames=hold_frames,
            settle_frames=settle_frames,
            phase=f"{phase}:{label}:{self.advance_presses_in_phase}",
        )
