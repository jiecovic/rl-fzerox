from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .action_adapters import ActionAdapterName
else:
    ActionAdapterName: TypeAlias = str


@dataclass(frozen=True, slots=True)
class LegacyActionAdapterCatalog:
    """Explicit historical adapter names kept for config/runtime compatibility."""

    continuous_steer_drive: ActionAdapterName = "continuous_steer_drive"
    hybrid_steer_drive_boost_lean: ActionAdapterName = "hybrid_steer_drive_boost_lean"
    hybrid_steer_drive_boost_lean_primitive: ActionAdapterName = (
        "hybrid_steer_drive_boost_lean_primitive"
    )
    hybrid_steer_drive_air_brake_boost_lean_pitch: ActionAdapterName = (
        "hybrid_steer_drive_air_brake_boost_lean_pitch"
    )
    hybrid_steer_gas_boost_lean: ActionAdapterName = "hybrid_steer_gas_boost_lean"
    hybrid_steer_gas_air_brake_boost_lean: ActionAdapterName = (
        "hybrid_steer_gas_air_brake_boost_lean"
    )
    hybrid_steer_gas_air_brake_boost_lean_pitch: ActionAdapterName = (
        "hybrid_steer_gas_air_brake_boost_lean_pitch"
    )
    hybrid_steer_drive_lean: ActionAdapterName = "hybrid_steer_drive_lean"
    steer_gas_air_brake_boost_lean: ActionAdapterName = "steer_gas_air_brake_boost_lean"
    steer_drive: ActionAdapterName = "steer_drive"
    steer_drive_boost: ActionAdapterName = "steer_drive_boost"
    steer_drive_boost_lean: ActionAdapterName = "steer_drive_boost_lean"

    @property
    def default(self) -> ActionAdapterName:
        return self.steer_drive_boost_lean

    @property
    def sac(self) -> frozenset[ActionAdapterName]:
        return frozenset((self.continuous_steer_drive,))

    @property
    def hybrid(self) -> frozenset[ActionAdapterName]:
        return frozenset(
            (
                self.hybrid_steer_drive_lean,
                self.hybrid_steer_drive_boost_lean,
                self.hybrid_steer_drive_boost_lean_primitive,
                self.hybrid_steer_drive_air_brake_boost_lean_pitch,
                self.hybrid_steer_gas_boost_lean,
                self.hybrid_steer_gas_air_brake_boost_lean,
                self.hybrid_steer_gas_air_brake_boost_lean_pitch,
            )
        )

    @property
    def continuous_drive(self) -> frozenset[ActionAdapterName]:
        return frozenset(
            (
                self.continuous_steer_drive,
                self.hybrid_steer_drive_lean,
                self.hybrid_steer_drive_boost_lean,
                self.hybrid_steer_drive_boost_lean_primitive,
                self.hybrid_steer_drive_air_brake_boost_lean_pitch,
            )
        )


LEGACY_ACTION_ADAPTERS = LegacyActionAdapterCatalog()

__all__ = ["LEGACY_ACTION_ADAPTERS", "LegacyActionAdapterCatalog"]
