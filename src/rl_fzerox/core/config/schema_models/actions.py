# src/rl_fzerox/core/config/schema_models/actions.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    ValidationInfo,
    field_validator,
    model_validator,
)

from rl_fzerox.core.config.action_branches import (
    ActionBranchCompilation,
    ActionBranchesConfig,
    compile_action_branches,
)
from rl_fzerox.core.config.schema_models.common import (
    ActionMaskOverrides,
    ContinuousAirBrakeMode,
    ContinuousDriveMode,
)
from rl_fzerox.core.domain.action_adapters import DEFAULT_ACTION_ADAPTER_NAME, ActionAdapterName
from rl_fzerox.core.domain.action_values import (
    ActionMaskValue,
    compile_action_mask_values,
)
from rl_fzerox.core.domain.lean import DEFAULT_LEAN_MODE, LeanMode


@dataclass(frozen=True, slots=True)
class ActionRuntimeDefaults:
    """Internal action timing defaults that are not exposed in YAML."""

    boost_decision_interval_frames: int = 1
    boost_request_lockout_frames: int = 5


ACTION_RUNTIME_DEFAULTS = ActionRuntimeDefaults()


class ActionMaskConfig(BaseModel):
    """Optional branch-wise action-mask restrictions for MultiDiscrete actions."""

    model_config = ConfigDict(extra="forbid")

    steer: tuple[ActionMaskValue, ...] | None = None
    drive: tuple[ActionMaskValue, ...] | None = None
    gas: tuple[ActionMaskValue, ...] | None = None
    air_brake: tuple[ActionMaskValue, ...] | None = None
    boost: tuple[ActionMaskValue, ...] | None = None
    lean: tuple[ActionMaskValue, ...] | None = None
    pitch: tuple[ActionMaskValue, ...] | None = None

    @field_validator("steer", "drive", "gas", "air_brake", "boost", "lean", "pitch")
    @classmethod
    def _validate_non_empty_mask_branch(
        cls,
        value: tuple[ActionMaskValue, ...] | None,
        info: ValidationInfo,
    ) -> tuple[ActionMaskValue, ...] | None:
        if value is not None and len(value) == 0:
            raise ValueError("Action mask branches must not be empty")
        if value is not None:
            compile_action_mask_values(str(info.field_name), value)
        return value

    def branch_overrides(self) -> dict[str, tuple[int, ...]]:
        """Return the explicitly configured branch restrictions only."""

        overrides: dict[str, tuple[int, ...]] = {}
        for branch_name in ("steer", "drive", "gas", "air_brake", "boost", "lean", "pitch"):
            values = getattr(self, branch_name)
            if values is not None:
                overrides[branch_name] = compile_action_mask_values(branch_name, values)
        return overrides


@dataclass(frozen=True, slots=True)
class ActionRuntimeConfig:
    """Concrete action adapter settings consumed by env/runtime code."""

    name: ActionAdapterName
    steer_buckets: int
    steer_response_power: float
    continuous_drive_mode: ContinuousDriveMode
    continuous_drive_deadzone: float
    continuous_drive_full_threshold: float
    continuous_drive_min_level: float
    continuous_air_brake_mode: ContinuousAirBrakeMode
    continuous_lean_deadzone: float
    lean_mode: LeanMode
    boost_unmask_max_speed_kph: float | None
    boost_decision_interval_frames: int
    boost_request_lockout_frames: int
    lean_unmask_min_speed_kph: float | None
    mask_overrides: ActionMaskOverrides | None

    @classmethod
    def from_config(cls, config: ActionConfig) -> ActionRuntimeConfig:
        return cls(
            name=config.name,
            steer_buckets=int(config.steer_buckets),
            steer_response_power=float(config.steer_response_power),
            continuous_drive_mode=config.continuous_drive_mode,
            continuous_drive_deadzone=float(config.continuous_drive_deadzone),
            continuous_drive_full_threshold=float(config.continuous_drive_full_threshold),
            continuous_drive_min_level=float(config.continuous_drive_min_level),
            continuous_air_brake_mode=config.continuous_air_brake_mode,
            continuous_lean_deadzone=float(config.continuous_lean_deadzone),
            lean_mode=config.lean_mode,
            boost_unmask_max_speed_kph=(
                None
                if config.boost_unmask_max_speed_kph is None
                else float(config.boost_unmask_max_speed_kph)
            ),
            boost_decision_interval_frames=ACTION_RUNTIME_DEFAULTS.boost_decision_interval_frames,
            boost_request_lockout_frames=ACTION_RUNTIME_DEFAULTS.boost_request_lockout_frames,
            lean_unmask_min_speed_kph=(
                None
                if config.lean_unmask_min_speed_kph is None
                else float(config.lean_unmask_min_speed_kph)
            ),
            mask_overrides=config.mask.branch_overrides() if config.mask is not None else None,
        )

    @classmethod
    def from_branch_config(
        cls,
        config: ActionConfig,
        compilation: ActionBranchCompilation,
    ) -> ActionRuntimeConfig:
        return cls(
            name=compilation.name,
            steer_buckets=int(config.steer_buckets),
            steer_response_power=float(
                config.steer_response_power
                if compilation.steer_response_power is None
                else compilation.steer_response_power
            ),
            continuous_drive_mode=(
                config.continuous_drive_mode
                if compilation.continuous_drive_mode is None
                else compilation.continuous_drive_mode
            ),
            continuous_drive_deadzone=float(
                config.continuous_drive_deadzone
                if compilation.continuous_drive_deadzone is None
                else compilation.continuous_drive_deadzone
            ),
            continuous_drive_full_threshold=float(
                config.continuous_drive_full_threshold
                if compilation.continuous_drive_full_threshold is None
                else compilation.continuous_drive_full_threshold
            ),
            continuous_drive_min_level=float(
                config.continuous_drive_min_level
                if compilation.continuous_drive_min_level is None
                else compilation.continuous_drive_min_level
            ),
            continuous_air_brake_mode=(
                config.continuous_air_brake_mode
                if compilation.continuous_air_brake_mode is None
                else compilation.continuous_air_brake_mode
            ),
            continuous_lean_deadzone=float(config.continuous_lean_deadzone),
            lean_mode=config.lean_mode if compilation.lean_mode is None else compilation.lean_mode,
            boost_unmask_max_speed_kph=compilation.boost_unmask_max_speed_kph,
            boost_decision_interval_frames=ACTION_RUNTIME_DEFAULTS.boost_decision_interval_frames,
            boost_request_lockout_frames=ACTION_RUNTIME_DEFAULTS.boost_request_lockout_frames,
            lean_unmask_min_speed_kph=compilation.lean_unmask_min_speed_kph,
            mask_overrides=compilation.mask_overrides,
        )


class ActionConfig(BaseModel):
    """Policy action adapter settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    name: ActionAdapterName = DEFAULT_ACTION_ADAPTER_NAME
    steer_buckets: int = Field(default=7, ge=3)
    steer_response_power: PositiveFloat = 1.0
    continuous_drive_mode: ContinuousDriveMode = "threshold"
    continuous_drive_deadzone: float = Field(default=0.2, ge=0.0, lt=1.0)
    continuous_drive_full_threshold: float = Field(default=1.0, gt=0.0, le=1.0)
    continuous_drive_min_level: float = Field(default=0.0, ge=0.0, lt=1.0)
    continuous_air_brake_mode: ContinuousAirBrakeMode = "always"
    continuous_lean_deadzone: float = Field(default=0.333333, ge=0.0, lt=1.0)
    lean_mode: LeanMode = DEFAULT_LEAN_MODE
    boost_unmask_max_speed_kph: NonNegativeFloat | None = None
    lean_unmask_min_speed_kph: NonNegativeFloat | None = None
    mask: ActionMaskConfig | None = None
    branches: ActionBranchesConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_action_fields(cls, data: object) -> object:
        # V4 LEGACY SHIM: legacy action config field names.
        # Early run manifests used `boost_unmask_min_speed_kph`, but the intended
        # boost gate is a max-speed cap: allow boost while slower, mask it when
        # already fast. Older air-brake booleans existed before
        # `continuous_air_brake_mode`.
        if not isinstance(data, Mapping):
            return data
        values: dict[str, object] = {str(key): value for key, value in data.items()}
        missing = object()
        legacy_gate = values.pop("boost_unmask_min_speed_kph", missing)
        if legacy_gate is not missing and "boost_unmask_max_speed_kph" not in values:
            values["boost_unmask_max_speed_kph"] = legacy_gate
        values.pop("boost_decision_interval_frames", missing)
        values.pop("boost_request_lockout_frames", missing)

        branches = values.get("branches")
        if isinstance(branches, Mapping):
            branch_values = {str(key): value for key, value in branches.items()}
            boost_branch = branch_values.get("boost")
            if isinstance(boost_branch, Mapping):
                boost_values = {str(key): value for key, value in boost_branch.items()}
                boost_values.pop("decision_interval_frames", missing)
                boost_values.pop("request_lockout_frames", missing)
                branch_values["boost"] = boost_values
                values["branches"] = branch_values

        legacy_air_brake_enabled = values.pop("continuous_air_brake_enabled", missing)
        legacy_disable_on_ground = values.pop(
            "continuous_air_brake_disable_on_ground",
            missing,
        )
        legacy_airborne_only = values.pop("continuous_air_brake_airborne_only", missing)
        if "continuous_air_brake_mode" not in values:
            if legacy_air_brake_enabled is False:
                values["continuous_air_brake_mode"] = "off"
            elif legacy_disable_on_ground is True or legacy_airborne_only is True:
                values["continuous_air_brake_mode"] = "disable_on_ground"

        return values

    def runtime(self) -> ActionRuntimeConfig:
        """Return the concrete adapter config consumed by env/runtime code.

        Branch-style YAML is the canonical user-facing config for new runs. Old
        adapter fields remain on this model only so v4 manifests can still load.
        """

        if self.branches is None:
            return ActionRuntimeConfig.from_config(self)
        compiled = compile_action_branches(self.branches)
        return ActionRuntimeConfig.from_branch_config(self, compiled)

    @field_validator("steer_buckets")
    @classmethod
    def _validate_odd_steer_buckets(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("steer_buckets must be odd so one bucket maps to straight")
        return value

    @model_validator(mode="after")
    def _validate_branch_runtime_config(self) -> ActionConfig:
        if self.continuous_drive_deadzone >= self.continuous_drive_full_threshold:
            raise ValueError(
                "continuous_drive_deadzone must be lower than continuous_drive_full_threshold"
            )
        if self.continuous_drive_min_level > 0.0 and self.continuous_drive_mode != "pwm":
            raise ValueError("continuous_drive_min_level requires continuous_drive_mode='pwm'")
        if self.branches is not None:
            compile_action_branches(self.branches)
        return self
