# src/rl_fzerox/core/config/vehicle_catalog.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EngineSettingAlias = Literal["max_acceleration", "balanced", "max_speed"]


@dataclass(frozen=True, slots=True)
class EngineSetting:
    """Resolved engine-slider setting used for ids and RAM writes."""

    id: str
    raw_value: int


@dataclass(frozen=True, slots=True)
class EngineSettingPreset:
    """Human-readable alias for the 0..100 engine slider."""

    id: EngineSettingAlias
    display_name: str
    raw_value: int


@dataclass(frozen=True, slots=True)
class VehicleInfo:
    """Stable stock vehicle identity needed before the emulator exists.

    The catalog intentionally stores only identity/menu-order fields. Physics
    stats such as body, boost, grip, weight, and derived runtime values belong
    to emulator RAM/decomp-backed telemetry, not config metadata.
    """

    id: str
    display_name: str
    character_index: int
    machine_select_slot: int


@dataclass(frozen=True, slots=True)
class VehicleCatalog:
    """ROM/decomp-backed stable ids for stock machines and engine aliases."""

    generated_engine_prefix: str = "engine_"
    engine_presets: tuple[EngineSettingPreset, ...] = (
        EngineSettingPreset("max_acceleration", "Max Acceleration", 0),
        EngineSettingPreset("balanced", "Balanced", 50),
        EngineSettingPreset("max_speed", "Max Speed", 100),
    )
    vehicles: tuple[VehicleInfo, ...] = field(
        default_factory=lambda: (
            VehicleInfo("blue_falcon", "Blue Falcon", 0, 0),
            VehicleInfo("golden_fox", "Golden Fox", 1, 1),
            VehicleInfo("wild_goose", "Wild Goose", 2, 2),
            VehicleInfo("fire_stingray", "Fire Stingray", 3, 3),
            VehicleInfo("white_cat", "White Cat", 4, 4),
            VehicleInfo("red_gazelle", "Red Gazelle", 5, 5),
            VehicleInfo("great_star", "Great Star", 6, 9),
            VehicleInfo("iron_tiger", "Iron Tiger", 7, 6),
            VehicleInfo("deep_claw", "Deep Claw", 8, 7),
            VehicleInfo("twin_noritta", "Twin Noritta", 9, 13),
            VehicleInfo("super_piranha", "Super Piranha", 10, 22),
            VehicleInfo("mighty_hurricane", "Mighty Hurricane", 11, 23),
            VehicleInfo("little_wyvern", "Little Wyvern", 12, 18),
            VehicleInfo("space_angler", "Space Angler", 13, 24),
            VehicleInfo("green_panther", "Green Panther", 14, 27),
            VehicleInfo("black_bull", "Black Bull", 15, 28),
            VehicleInfo("wild_boar", "Wild Boar", 16, 20),
            VehicleInfo("astro_robin", "Astro Robin", 17, 17),
            VehicleInfo("king_meteor", "King Meteor", 18, 21),
            VehicleInfo("queen_meteor", "Queen Meteor", 19, 15),
            VehicleInfo("wonder_wasp", "Wonder Wasp", 20, 14),
            VehicleInfo("hyper_speeder", "Hyper Speeder", 21, 26),
            VehicleInfo("death_anchor", "Death Anchor", 22, 19),
            VehicleInfo("crazy_bear", "Crazy Bear", 23, 8),
            VehicleInfo("night_thunder", "Night Thunder", 24, 12),
            VehicleInfo("big_fang", "Big Fang", 25, 10),
            VehicleInfo("mighty_typhoon", "Mighty Typhoon", 26, 25),
            VehicleInfo("mad_wolf", "Mad Wolf", 27, 11),
            VehicleInfo("sonic_phantom", "Sonic Phantom", 28, 29),
            VehicleInfo("blood_hawk", "Blood Hawk", 29, 16),
        )
    )


CATALOG = VehicleCatalog()


def vehicle_by_id(vehicle_id: str) -> VehicleInfo:
    """Return a stock vehicle identity by config id."""

    for vehicle in CATALOG.vehicles:
        if vehicle.id == vehicle_id:
            return vehicle
    known = ", ".join(known_vehicle_ids())
    raise ValueError(f"unknown vehicle {vehicle_id!r}; known: {known}")


def known_vehicle_ids() -> tuple[str, ...]:
    """Return all stock vehicle ids accepted by config."""

    return tuple(vehicle.id for vehicle in CATALOG.vehicles)


def resolve_engine_setting(raw_engine_setting: object, *, context: str) -> EngineSetting:
    """Resolve a named alias, generated id, or raw 0..100 slider value."""

    if isinstance(raw_engine_setting, str) and raw_engine_setting:
        resolved = try_resolve_engine_setting_id(raw_engine_setting, context=context)
        if resolved is not None:
            return resolved
        known = ", ".join(preset.id for preset in CATALOG.engine_presets)
        raise ValueError(
            f"{context} unknown engine_setting {raw_engine_setting!r}; "
            f"known aliases: {known}, or raw integers 0..100"
        )

    if isinstance(raw_engine_setting, bool) or not isinstance(raw_engine_setting, int):
        raise ValueError(
            "track_sampling.baseline.engine_setting must be a string id or raw integer"
        )
    _validate_engine_setting_raw_value(raw_engine_setting, context=context)
    for preset in CATALOG.engine_presets:
        if preset.raw_value == raw_engine_setting:
            return EngineSetting(id=preset.id, raw_value=raw_engine_setting)
    return EngineSetting(
        id=f"{CATALOG.generated_engine_prefix}{raw_engine_setting}",
        raw_value=raw_engine_setting,
    )


def try_resolve_engine_setting_id(
    engine_setting: str,
    *,
    context: str,
) -> EngineSetting | None:
    """Resolve a named preset or generated raw engine id, if valid."""

    for preset in CATALOG.engine_presets:
        if preset.id == engine_setting:
            return EngineSetting(id=preset.id, raw_value=preset.raw_value)

    raw_value = _raw_value_from_generated_engine_setting_id(engine_setting)
    if raw_value is None:
        return None
    _validate_engine_setting_raw_value(raw_value, context=context)
    return EngineSetting(id=engine_setting, raw_value=raw_value)


def engine_setting_display_name(engine_setting: str) -> str:
    """Return a short display label for named or generated engine settings."""

    for preset in CATALOG.engine_presets:
        if preset.id == engine_setting:
            return preset.display_name
    raw_value = _raw_value_from_generated_engine_setting_id(engine_setting)
    if raw_value is not None:
        return f"Engine {raw_value}"
    return engine_setting.replace("_", " ").title()


def _raw_value_from_generated_engine_setting_id(engine_setting: str) -> int | None:
    if not engine_setting.startswith(CATALOG.generated_engine_prefix):
        return None
    raw_value = engine_setting.removeprefix(CATALOG.generated_engine_prefix)
    if not raw_value.isdigit():
        return None
    return int(raw_value)


def _validate_engine_setting_raw_value(raw_value: int, *, context: str) -> None:
    if not 0 <= raw_value <= 100:
        raise ValueError(f"{context} raw engine_setting must be in [0, 100], got {raw_value}")
