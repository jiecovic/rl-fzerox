# src/rl_fzerox/core/envs/engine/stepping/result.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator.arrays import ControllerMaskBatch, DisplayFrames
from rl_fzerox.core.envs.observations import ObservationValue

_POLICY_DRIVE_TERMINAL_REASONS = frozenset({"finished", "retired", "crashed"})
_POLICY_DRIVE_EXCLUDED_INFO_KEYS = frozenset(
    {
        "terminated",
        "truncated",
        "truncation_reason",
    }
)


@dataclass(frozen=True)
class PolicyDriveStep:
    """Policy-facing live-race step without Gym episode lifecycle fields."""

    observation: ObservationValue
    reward: float
    info: dict[str, object]
    display_frames: DisplayFrames
    display_controller_masks: ControllerMaskBatch = ()


@dataclass(frozen=True)
class WatchEnvStep:
    """Gym step result plus watch-only display frames from repeated inner frames."""

    observation: ObservationValue
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]
    display_frames: DisplayFrames
    display_controller_masks: ControllerMaskBatch = ()

    def gym_result(self) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def policy_drive_result(self) -> PolicyDriveStep:
        return PolicyDriveStep(
            observation=self.observation,
            reward=self.reward,
            info=policy_drive_info(self.info),
            display_frames=self.display_frames,
            display_controller_masks=self.display_controller_masks,
        )


def policy_drive_info(info: dict[str, object]) -> dict[str, object]:
    """Return live policy-drive info without env episode lifecycle state."""

    normalized = {
        key: value for key, value in info.items() if key not in _POLICY_DRIVE_EXCLUDED_INFO_KEYS
    }
    reason = normalized.get("termination_reason")
    if reason not in _POLICY_DRIVE_TERMINAL_REASONS:
        normalized.pop("termination_reason", None)
    return normalized
