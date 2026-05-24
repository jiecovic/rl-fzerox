# src/rl_fzerox/core/manager/run_spec/sections/vehicle.py
"""Vehicle selection and engine-setting section of the manager config."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

from rl_fzerox.core.manager.run_spec.common import (
    EngineSettingMode,
    VehicleSelectionMode,
)
from rl_fzerox.core.runtime_spec.vehicle_catalog import known_vehicle_ids


class ManagedVehicleConfig(BaseModel):
    """Vehicle selection knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    selection_mode: VehicleSelectionMode = "pool"
    selected_vehicle_ids: tuple[str, ...] = Field(default_factory=lambda: ("blue_falcon",))
    engine_mode: EngineSettingMode = "fixed"
    engine_setting_raw_value: NonNegativeInt = Field(default=50, le=100)
    engine_setting_min_raw_value: NonNegativeInt = Field(default=20, le=100)
    engine_setting_max_raw_value: NonNegativeInt = Field(default=80, le=100)

    @model_validator(mode="after")
    def _validate_vehicle_config(self) -> ManagedVehicleConfig:
        if not self.selected_vehicle_ids:
            raise ValueError("vehicle.selected_vehicle_ids must not be empty")
        if len(set(self.selected_vehicle_ids)) != len(self.selected_vehicle_ids):
            raise ValueError("vehicle.selected_vehicle_ids must not contain duplicates")
        unknown_ids = sorted(set(self.selected_vehicle_ids) - set(known_vehicle_ids()))
        if unknown_ids:
            known = ", ".join(known_vehicle_ids())
            joined = ", ".join(unknown_ids)
            raise ValueError(
                f"vehicle.selected_vehicle_ids contains unknown vehicles: {joined}; known: {known}"
            )
        if self.selection_mode == "fixed" and len(self.selected_vehicle_ids) != 1:
            raise ValueError("vehicle.selection_mode=fixed requires exactly one selected vehicle")
        if self.engine_setting_min_raw_value > self.engine_setting_max_raw_value:
            raise ValueError(
                "vehicle.engine_setting_min_raw_value must be <= "
                "vehicle.engine_setting_max_raw_value"
            )
        return self
