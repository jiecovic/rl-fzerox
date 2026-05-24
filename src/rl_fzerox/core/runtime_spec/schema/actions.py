# src/rl_fzerox/core/runtime_spec/schema/actions.py
"""Derived runtime action-schema models used by env, training, and manifests.

The manager-owned ``run_spec`` model is the canonical authoring surface. These
models describe the lower-level runtime shape after projection. Saved manifests
use the same shape, so this module stays strict and avoids carrying extra
authoring-era aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    ValidationInfo,
    field_validator,
    model_validator,
)

from rl_fzerox.core.domain.action_adapters import ActionAdapterName
from rl_fzerox.core.domain.action_values import (
    ActionMaskSpec,
    compile_action_mask_values,
)
from rl_fzerox.core.domain.lean import DEFAULT_LEAN_MODE, LeanMode
from rl_fzerox.core.runtime_spec.schema.common import (
    ActionMaskOverrides,
    ContinuousAirBrakeMode,
)

ConfiguredContinuousAxis: TypeAlias = Literal["steer", "drive", "air_brake", "pitch"]
ConfiguredDiscreteAxis: TypeAlias = Literal[
    "steer",
    "gas",
    "air_brake",
    "boost",
    "lean",
    "pitch",
]


@dataclass(frozen=True, slots=True)
class ConfiguredActionLayoutCatalog:
    """Owned axis catalog for the configured discrete/hybrid runtime layouts."""

    continuous_axes: frozenset[ConfiguredContinuousAxis] = frozenset(
        ("steer", "drive", "air_brake", "pitch")
    )
    discrete_axes: frozenset[ConfiguredDiscreteAxis] = frozenset(
        ("steer", "gas", "air_brake", "boost", "lean", "pitch")
    )
    default_discrete_axes: tuple[ConfiguredDiscreteAxis, ...] = (
        "steer",
        "gas",
        "boost",
        "lean",
    )


CONFIGURED_ACTION_LAYOUTS = ConfiguredActionLayoutCatalog()


@dataclass(frozen=True, slots=True)
class ActionRuntimeDefaults:
    """Internal action timing defaults that are not exposed in YAML."""

    boost_decision_interval_frames: int = 1
    boost_request_lockout_frames: int = 5


ACTION_RUNTIME_DEFAULTS = ActionRuntimeDefaults()


class ActionMaskConfig(BaseModel):
    """Optional branch-wise action-mask restrictions for MultiDiscrete actions."""

    model_config = ConfigDict(extra="forbid")

    steer: ActionMaskSpec | None = None
    gas: ActionMaskSpec | None = None
    air_brake: ActionMaskSpec | None = None
    boost: ActionMaskSpec | None = None
    lean: ActionMaskSpec | None = None
    pitch: ActionMaskSpec | None = None

    @field_validator("steer", "gas", "air_brake", "boost", "lean", "pitch")
    @classmethod
    def _validate_non_empty_mask_branch(
        cls,
        value: ActionMaskSpec | None,
        info: ValidationInfo,
    ) -> ActionMaskSpec | None:
        if value is not None and not isinstance(value, str) and len(value) == 0:
            raise ValueError("Action mask branches must not be empty")
        if value is not None:
            compile_action_mask_values(str(info.field_name), value)
        return value

    def branch_overrides(self) -> dict[str, tuple[int, ...]]:
        """Return the explicitly configured branch restrictions only."""

        overrides: dict[str, tuple[int, ...]] = {}
        for branch_name in ("steer", "gas", "air_brake", "boost", "lean", "pitch"):
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
    continuous_drive_deadzone: float
    continuous_drive_full_threshold: float
    continuous_drive_min_thrust: float
    continuous_air_brake_deadzone: float
    continuous_air_brake_full_threshold: float
    continuous_air_brake_min_duty: float
    force_full_throttle: bool
    mask_air_brake_on_ground: bool
    continuous_air_brake_mode: ContinuousAirBrakeMode
    continuous_lean_deadzone: float
    lean_mode: LeanMode
    boost_unmask_max_speed_kph: float | None
    boost_decision_interval_frames: int
    boost_request_lockout_frames: int
    lean_unmask_min_speed_kph: float | None
    lean_initial_lockout_frames: int
    pitch_deadzone: float
    pitch_buckets: int
    independent_lean_buttons: bool
    layout_continuous_axes: tuple[ConfiguredContinuousAxis, ...]
    layout_discrete_axes: tuple[ConfiguredDiscreteAxis, ...]
    mask_overrides: ActionMaskOverrides | None

    def continuous_drive_axis_index(self) -> int | None:
        """Return the continuous drive-axis index when this layout exposes one."""

        if self.name != "configured_hybrid":
            return None
        try:
            return self.layout_continuous_axes.index("drive")
        except ValueError:
            return None

    def continuous_air_brake_axis_index(self) -> int | None:
        """Return the continuous air-brake axis index when this layout exposes one."""

        if self.name != "configured_hybrid":
            return None
        try:
            return self.layout_continuous_axes.index("air_brake")
        except ValueError:
            return None

    def uses_continuous_drive(self) -> bool:
        """Return whether the action layout carries a live continuous throttle axis."""

        return self.continuous_drive_axis_index() is not None

    def uses_continuous_air_brake(self) -> bool:
        """Return whether the action layout carries a live continuous air-brake axis."""

        return self.continuous_air_brake_axis_index() is not None

    @classmethod
    def from_config(cls, config: ActionConfig) -> ActionRuntimeConfig:
        return cls(
            name=config.resolved_adapter_name(),
            steer_buckets=int(config.steer_buckets),
            steer_response_power=float(config.steer_response_power),
            continuous_drive_deadzone=float(config.continuous_drive_deadzone),
            continuous_drive_full_threshold=float(config.continuous_drive_full_threshold),
            continuous_drive_min_thrust=float(config.continuous_drive_min_thrust),
            continuous_air_brake_deadzone=float(config.continuous_air_brake_deadzone),
            continuous_air_brake_full_threshold=float(config.continuous_air_brake_full_threshold),
            continuous_air_brake_min_duty=float(config.continuous_air_brake_min_duty),
            force_full_throttle=bool(config.force_full_throttle),
            mask_air_brake_on_ground=bool(config.mask_air_brake_on_ground),
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
            lean_initial_lockout_frames=int(config.lean_initial_lockout_frames),
            pitch_deadzone=float(config.pitch_deadzone),
            pitch_buckets=int(config.pitch_buckets),
            independent_lean_buttons=bool(config.independent_lean_buttons),
            layout_continuous_axes=tuple(config.layout_continuous_axes),
            layout_discrete_axes=tuple(config.layout_discrete_axes),
            mask_overrides=config.resolved_mask_overrides(),
        )


class ActionConfig(BaseModel):
    """Concrete runtime action-layout settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    steer_buckets: int = Field(default=7, ge=3)
    steer_response_power: PositiveFloat = 1.0
    continuous_drive_deadzone: float = Field(default=0.05, ge=0.0, lt=1.0)
    continuous_drive_full_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    continuous_drive_min_thrust: float = Field(default=0.25, ge=0.0, le=1.0)
    continuous_air_brake_deadzone: float = Field(default=0.05, ge=0.0, lt=1.0)
    continuous_air_brake_full_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    continuous_air_brake_min_duty: float = Field(default=0.0, ge=0.0, le=1.0)
    force_full_throttle: bool = False
    mask_air_brake_on_ground: bool = True
    continuous_air_brake_mode: ContinuousAirBrakeMode = "always"
    continuous_lean_deadzone: float = Field(default=0.333333, ge=0.0, lt=1.0)
    lean_mode: LeanMode = DEFAULT_LEAN_MODE
    boost_unmask_max_speed_kph: NonNegativeFloat | None = None
    lean_unmask_min_speed_kph: NonNegativeFloat | None = None
    lean_initial_lockout_frames: NonNegativeInt = 0
    pitch_deadzone: float = Field(default=0.1, ge=0.0, lt=1.0)
    pitch_buckets: int = Field(default=5, ge=3)
    independent_lean_buttons: bool = False
    layout_continuous_axes: tuple[ConfiguredContinuousAxis, ...] = ()
    layout_discrete_axes: tuple[ConfiguredDiscreteAxis, ...] = (
        CONFIGURED_ACTION_LAYOUTS.default_discrete_axes
    )
    mask: ActionMaskConfig | None = None

    def runtime(self) -> ActionRuntimeConfig:
        """Return the concrete adapter config consumed by env/runtime code."""

        return ActionRuntimeConfig.from_config(self)

    def resolved_adapter_name(self) -> ActionAdapterName:
        """Return the adapter family implied by the configured action layout."""

        return "configured_hybrid" if self.layout_continuous_axes else "configured_discrete"

    def resolved_mask_overrides(self) -> ActionMaskOverrides | None:
        """Return one canonical set of branch overrides for runtime consumers."""

        if self.mask is not None:
            return self.mask.branch_overrides()
        return None

    @field_validator("steer_buckets", "pitch_buckets")
    @classmethod
    def _validate_odd_bucket_count(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("action bucket counts must be odd so one bucket maps to neutral")
        return value

    @field_validator("layout_continuous_axes", "layout_discrete_axes")
    @classmethod
    def _validate_unique_layout_axes(
        cls,
        value: tuple[object, ...],
    ) -> tuple[object, ...]:
        if len(set(value)) != len(value):
            raise ValueError("configured action layout axes must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _validate_branch_runtime_config(self) -> ActionConfig:
        if self.continuous_drive_deadzone >= self.continuous_drive_full_threshold:
            raise ValueError(
                "continuous_drive_deadzone must be lower than continuous_drive_full_threshold"
            )
        if self.continuous_air_brake_deadzone >= self.continuous_air_brake_full_threshold:
            raise ValueError(
                "continuous_air_brake_deadzone must be lower than "
                "continuous_air_brake_full_threshold"
            )
        invalid_continuous_axes = (
            set(self.layout_continuous_axes) - CONFIGURED_ACTION_LAYOUTS.continuous_axes
        )
        if invalid_continuous_axes:
            joined = ", ".join(sorted(invalid_continuous_axes))
            raise ValueError(f"Unknown continuous action layout axis: {joined}")
        invalid_discrete_axes = (
            set(self.layout_discrete_axes) - CONFIGURED_ACTION_LAYOUTS.discrete_axes
        )
        if invalid_discrete_axes:
            joined = ", ".join(sorted(invalid_discrete_axes))
            raise ValueError(f"Unknown discrete action layout axis: {joined}")
        if set(self.layout_continuous_axes) & set(self.layout_discrete_axes):
            raise ValueError("configured action axes cannot be both continuous and discrete")
        resolved_name = self.resolved_adapter_name()
        if resolved_name == "configured_hybrid" and not self.layout_continuous_axes:
            raise ValueError("configured_hybrid requires at least one continuous axis")
        if resolved_name == "configured_discrete":
            if self.layout_continuous_axes:
                raise ValueError("configured_discrete cannot define continuous axes")
            if not self.layout_discrete_axes:
                raise ValueError("configured_discrete requires at least one discrete axis")
        if self.independent_lean_buttons and "lean" not in self.layout_discrete_axes:
            raise ValueError("independent_lean_buttons requires a discrete lean axis")
        return self
