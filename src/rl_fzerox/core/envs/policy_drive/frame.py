# src/rl_fzerox/core/envs/policy_drive/frame.py
"""Policy-drive frame data.

Watch/manual driving can request rendered frames and audio outside Gym training.
These containers describe what one live policy-drive step produced.
"""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator.arrays import (
    AudioFrameCounts,
    ControllerMaskBatch,
    DisplayFrames,
    Pcm16Samples,
)
from rl_fzerox.core.envs.engine.stepping import WatchEnvStep
from rl_fzerox.core.envs.observations import ObservationValue

_TERMINAL_REASONS = frozenset({"finished", "retired", "crashed"})
_EXCLUDED_INFO_KEYS = frozenset(
    {
        "terminated",
        "truncated",
        "truncation_reason",
    }
)


@dataclass(frozen=True, slots=True)
class PolicyDriveStep:
    """Policy-facing live-race step without Gym done/truncation fields."""

    observation: ObservationValue
    reward: float
    info: dict[str, object]
    display_frames: DisplayFrames
    display_controller_masks: ControllerMaskBatch = ()
    audio_samples: Pcm16Samples = ()
    audio_frame_counts: AudioFrameCounts = ()


@dataclass(frozen=True, slots=True)
class PolicyDriveFrame:
    """One policy-owned race frame without Gym done/truncation semantics."""

    observation: ObservationValue
    reward: float
    info: dict[str, object]
    display_frames: DisplayFrames
    display_controller_masks: ControllerMaskBatch
    audio_samples: Pcm16Samples = ()
    audio_frame_counts: AudioFrameCounts = ()


def policy_drive_info(info: dict[str, object]) -> dict[str, object]:
    """Return live policy-drive info without Gym done/truncation state."""

    normalized = {key: value for key, value in info.items() if key not in _EXCLUDED_INFO_KEYS}
    reason = normalized.get("termination_reason")
    if reason not in _TERMINAL_REASONS:
        normalized.pop("termination_reason", None)
    return normalized


def policy_drive_step(step: WatchEnvStep) -> PolicyDriveStep:
    """Adapt a low-level repeated-step result to the policy-drive contract."""

    return PolicyDriveStep(
        observation=step.observation,
        reward=step.reward,
        info=policy_drive_info(step.info),
        display_frames=step.display_frames,
        display_controller_masks=step.display_controller_masks,
        audio_samples=step.audio_samples,
        audio_frame_counts=step.audio_frame_counts,
    )
