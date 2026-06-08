# src/rl_fzerox/core/runtime_spec/vehicle_catalog.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EngineSetting:
    """Resolved 0..100 engine-slider setting used for RAM writes."""

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
class VehicleMenuGrid:
    """Physical machine-select grid layout used by the stock vehicle menu."""

    column_count: int = 6

    def row_and_column(self, machine_select_slot: int) -> tuple[int, int]:
        if machine_select_slot < 0:
            raise ValueError(f"machine_select_slot must be non-negative, got {machine_select_slot}")
        return divmod(machine_select_slot, self.column_count)


@dataclass(frozen=True, slots=True)
class VehicleCatalog:
    """ROM/decomp-backed stable ids for stock machines."""

    menu_grid: VehicleMenuGrid = field(default_factory=VehicleMenuGrid)
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


def vehicle_by_character_index(character_index: int) -> VehicleInfo | None:
    """Return a stock vehicle identity from the native character index."""

    for vehicle in CATALOG.vehicles:
        if vehicle.character_index == character_index:
            return vehicle
    return None


def known_vehicle_ids() -> tuple[str, ...]:
    """Return all stock vehicle ids accepted by config."""

    return tuple(vehicle.id for vehicle in CATALOG.vehicles)


def vehicle_ids_by_menu_slot() -> tuple[str, ...]:
    """Return stock vehicle ids in machine-select unlock row order."""

    return tuple(
        vehicle.id
        for vehicle in sorted(CATALOG.vehicles, key=lambda item: item.machine_select_slot)
    )


def vehicle_menu_row_and_column(machine_select_slot: int) -> tuple[int, int]:
    """Return the row/column of one machine in the stock vehicle-select grid."""

    return CATALOG.menu_grid.row_and_column(machine_select_slot)


def resolve_engine_setting(raw_engine_setting: object, *, context: str) -> EngineSetting:
    """Resolve one raw 0..100 engine-slider value."""

    if isinstance(raw_engine_setting, bool) or not isinstance(raw_engine_setting, int):
        raise ValueError(f"{context} engine_setting_raw_value must be a raw integer in [0, 100]")
    _validate_engine_setting_raw_value(raw_engine_setting, context=context)
    return EngineSetting(raw_value=raw_engine_setting)


def engine_setting_display_name_for_raw(raw_value: int) -> str:
    """Return a short display label for a native 0..100 engine-slider value."""

    return f"Engine {raw_value}"


def _validate_engine_setting_raw_value(raw_value: int, *, context: str) -> None:
    if not 0 <= raw_value <= 100:
        raise ValueError(f"{context} raw engine_setting must be in [0, 100], got {raw_value}")
