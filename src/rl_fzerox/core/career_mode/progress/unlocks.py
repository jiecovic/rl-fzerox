# src/rl_fzerox/core/career_mode/progress/unlocks.py
"""Rule-derived unlock path for portable save games."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_fzerox.core.domain.race_difficulty import RaceDifficultyName
from rl_fzerox.core.manager.models import (
    ManagedSaveAttempt,
    ManagedSaveUnlockProgress,
    ManagedSaveUnlockTarget,
    SaveUnlockInspectionStatus,
    SaveUnlockTargetStatus,
)
from rl_fzerox.core.runtime_spec.vehicle_catalog import vehicle_ids_by_menu_slot
from rl_fzerox.core.save_game import FZeroXUnlockState, read_fzerox_unlock_state


@dataclass(frozen=True, slots=True)
class UnlockRuleTarget:
    """One game-rule target that can be attempted by a trained policy."""

    sequence_index: int
    kind: str
    label: str
    difficulty: RaceDifficultyName
    cup_id: str
    course_id: str | None = None

    def to_progress_target(
        self,
        *,
        status: SaveUnlockTargetStatus,
    ) -> ManagedSaveUnlockTarget:
        return ManagedSaveUnlockTarget(
            sequence_index=self.sequence_index,
            kind=self.kind,
            status=status,
            label=self.label,
            difficulty=self.difficulty,
            cup_id=self.cup_id,
            course_id=self.course_id,
        )


@dataclass(frozen=True, slots=True)
class UnlockRulePath:
    """Ordered fixed-cup GP work derived from game unlock rules."""

    kind: str
    difficulties: tuple[RaceDifficultyName, ...]
    cup_ids: tuple[str, ...]

    def targets(self) -> tuple[UnlockRuleTarget, ...]:
        targets: list[UnlockRuleTarget] = []
        for difficulty in self.difficulties:
            for cup_id in self.cup_ids:
                targets.append(
                    UnlockRuleTarget(
                        sequence_index=len(targets),
                        kind=self.kind,
                        difficulty=difficulty,
                        cup_id=cup_id,
                        label=f"Clear {difficulty.title()} {cup_id.title()} Cup",
                    )
                )
        return tuple(targets)


DEFAULT_UNLOCK_RULE_PATH = UnlockRulePath(
    kind="clear_gp_cup",
    difficulties=("novice", "standard", "expert", "master"),
    cup_ids=("jack", "queen", "king", "joker"),
)


def default_unlock_targets() -> tuple[UnlockRuleTarget, ...]:
    """Return the fixed-cup GP targets derived from game rules."""

    return DEFAULT_UNLOCK_RULE_PATH.targets()


def build_unlock_progress(
    save_path: Path,
    *,
    attempts: tuple[ManagedSaveAttempt, ...] = (),
    rule_targets: tuple[UnlockRuleTarget, ...] | None = None,
) -> ManagedSaveUnlockProgress:
    """Build current unlock progress for one save file."""

    del attempts
    unlock_state = read_fzerox_unlock_state(save_path)
    inspection_status: SaveUnlockInspectionStatus = (
        "inspected" if unlock_state is not None else "not_inspected"
    )
    progress_targets = tuple(
        target.to_progress_target(
            status=_target_status_from_save(target, unlock_state=unlock_state)
        )
        for target in (rule_targets or default_unlock_targets())
    )
    next_target = next(
        (target for target in progress_targets if target.status == "pending"),
        None,
    )
    unlocked_vehicle_ids = _unlocked_vehicle_ids(unlock_state)
    return ManagedSaveUnlockProgress(
        inspection_status=inspection_status,
        completed_count=sum(1 for target in progress_targets if target.status == "succeeded"),
        total_count=len(progress_targets),
        unlocked_vehicle_count=len(unlocked_vehicle_ids),
        unlocked_vehicle_ids=unlocked_vehicle_ids,
        next_target=next_target,
        targets=progress_targets,
    )


def _target_status_from_save(
    target: UnlockRuleTarget,
    *,
    unlock_state: FZeroXUnlockState | None,
) -> SaveUnlockTargetStatus:
    if unlock_state is None:
        return "pending" if target.sequence_index == 0 else "locked"
    if target.kind != "clear_gp_cup":
        return "pending"
    if not _target_available(target, unlock_state=unlock_state):
        return "locked"
    return (
        "succeeded"
        if unlock_state.gp_cup_cleared(
            difficulty=target.difficulty,
            cup_id=target.cup_id,
        )
        else "pending"
    )


def _target_available(
    target: UnlockRuleTarget,
    *,
    unlock_state: FZeroXUnlockState,
) -> bool:
    if target.difficulty == "master" and not _master_difficulty_available(unlock_state):
        return False
    if target.cup_id == "joker" and not _joker_cup_available(unlock_state):
        return False
    return True


def _joker_cup_available(unlock_state: FZeroXUnlockState) -> bool:
    return all(
        unlock_state.gp_cup_cleared(difficulty="standard", cup_id=cup_id)
        for cup_id in ("jack", "queen", "king")
    )


def _master_difficulty_available(unlock_state: FZeroXUnlockState) -> bool:
    return all(
        unlock_state.gp_cup_cleared(difficulty="expert", cup_id=cup_id)
        for cup_id in ("jack", "queen", "king", "joker")
    )


def _unlocked_vehicle_ids(unlock_state: FZeroXUnlockState | None) -> tuple[str, ...]:
    vehicle_ids = vehicle_ids_by_menu_slot()
    if unlock_state is None:
        return vehicle_ids[:6]
    return vehicle_ids[: unlock_state.unlocked_vehicle_count]
