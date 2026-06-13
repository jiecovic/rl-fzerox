# src/rl_fzerox/core/manager/run_spec/sections/vehicle.py
"""Vehicle selection and engine-setting section of the manager config."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from rl_fzerox.core.engine_tuning.types import ENGINE_TUNER_DEFAULTS
from rl_fzerox.core.manager.run_spec.common import (
    EngineSettingMode,
    EngineTunerBackend,
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
    adaptive_engine_tuner_backend: EngineTunerBackend = ENGINE_TUNER_DEFAULTS.backend
    adaptive_engine_stat_decay: float = Field(
        default=ENGINE_TUNER_DEFAULTS.stat_decay,
        gt=0.0,
        lt=1.0,
    )
    adaptive_engine_ensemble_members: int = Field(
        default=ENGINE_TUNER_DEFAULTS.mlp_ensemble_members,
        ge=1,
        le=32,
    )
    adaptive_engine_mlp_hidden_dim: int = Field(
        default=ENGINE_TUNER_DEFAULTS.mlp_hidden_dim,
        ge=4,
        le=512,
    )
    adaptive_engine_mlp_training_steps: int = Field(
        default=ENGINE_TUNER_DEFAULTS.mlp_training_steps,
        ge=1,
        le=2048,
    )
    adaptive_engine_mlp_learning_rate: float = Field(
        default=ENGINE_TUNER_DEFAULTS.mlp_learning_rate,
        gt=0.0,
        le=1.0,
    )
    adaptive_engine_mlp_bootstrap_keep_probability: float = Field(
        default=ENGINE_TUNER_DEFAULTS.mlp_bootstrap_keep_probability,
        gt=0.0,
        le=1.0,
    )
    adaptive_engine_mlp_warmup_successes: int = Field(
        default=ENGINE_TUNER_DEFAULTS.mlp_warmup_successes,
        ge=1,
        le=4096,
    )
    adaptive_engine_uniform_exploration: float = Field(
        default=ENGINE_TUNER_DEFAULTS.uniform_exploration,
        ge=0.0,
        le=1.0,
    )
    adaptive_engine_greedy_plateau_seconds: float = Field(
        default=ENGINE_TUNER_DEFAULTS.greedy_plateau_tolerance_seconds,
        ge=0.0,
        le=30.0,
    )

    @model_serializer(mode="wrap")
    def _serialize_vehicle(self, handler: SerializerFunctionWrapHandler) -> object:
        data = handler(self)
        if isinstance(data, dict) and self.adaptive_engine_tuner_backend != "gaussian_process":
            data.pop("adaptive_engine_stat_decay", None)
        if isinstance(data, dict) and self.adaptive_engine_tuner_backend != "mlp_ensemble":
            data.pop("adaptive_engine_ensemble_members", None)
            data.pop("adaptive_engine_mlp_hidden_dim", None)
            data.pop("adaptive_engine_mlp_training_steps", None)
            data.pop("adaptive_engine_mlp_learning_rate", None)
            data.pop("adaptive_engine_mlp_bootstrap_keep_probability", None)
            data.pop("adaptive_engine_mlp_warmup_successes", None)
        return data

    @model_validator(mode="before")
    @classmethod
    def _default_adaptive_engine_range(cls, data: object) -> object:
        if not isinstance(data, dict) or data.get("engine_mode") != "adaptive_tuner":
            return data
        next_data = dict(data)
        next_data.setdefault("engine_setting_min_raw_value", 0)
        next_data.setdefault("engine_setting_max_raw_value", 100)
        return next_data

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
