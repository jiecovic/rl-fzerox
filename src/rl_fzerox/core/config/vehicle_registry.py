# src/rl_fzerox/core/config/vehicle_registry.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True, slots=True)
class VehicleRegistryLayout:
    """Filesystem layout for optional vehicle metadata overrides."""

    root: str = "vehicles"


@dataclass(frozen=True, slots=True)
class EngineSetting:
    """Resolved engine-slider setting used for registry ids and RAM writes."""

    id: str
    raw_value: int | None


@dataclass(frozen=True, slots=True)
class BuiltinVehicle:
    """Stable built-in machine metadata needed before the emulator exists."""

    id: str
    display_name: str
    character_index: int
    machine_select_slot: int | None = None
    body_stat: int | None = None
    boost_stat: int | None = None
    grip_stat: int | None = None
    weight: int | None = None

    def config_data(self) -> dict[str, object]:
        machine: dict[str, object] = {"character_index": self.character_index}
        for key in (
            "machine_select_slot",
            "body_stat",
            "boost_stat",
            "grip_stat",
            "weight",
        ):
            value = getattr(self, key)
            if value is not None:
                machine[key] = value
        return {
            "id": self.id,
            "display_name": self.display_name,
            "engine_settings": DEFAULT_ENGINE_SETTINGS,
            "machine": machine,
        }


@dataclass(frozen=True, slots=True)
class VehicleRegistryDefaults:
    """Canonical stock engine presets used by every F-Zero X machine."""

    generated_engine_prefix: str = "engine_"
    engine_settings: Mapping[str, Mapping[str, object]] = field(
        default_factory=lambda: {
            "max_acceleration": {"display_name": "Max Acceleration", "raw_value": 0},
            "balanced": {"display_name": "Balanced", "raw_value": 50},
            "max_speed": {"display_name": "Max Speed", "raw_value": 100},
        }
    )
    unsupported_yaml_keys: frozenset[str] = frozenset(
        {
            "baseline_source_engine_setting",
            "baseline_source_vehicle",
        }
    )


LAYOUT = VehicleRegistryLayout()
DEFAULTS = VehicleRegistryDefaults()
DEFAULT_ENGINE_SETTINGS = DEFAULTS.engine_settings

# Character ids come from the game enum; machine_select_slot is the machine-select
# menu order. Those two orders diverge after the first few machines.
BUILTIN_VEHICLES: tuple[BuiltinVehicle, ...] = (
    BuiltinVehicle("blue_falcon", "Blue Falcon", 0, 0, 1, 2, 1, 1260),
    BuiltinVehicle("golden_fox", "Golden Fox", 1, 1),
    BuiltinVehicle("wild_goose", "Wild Goose", 2, 2),
    BuiltinVehicle("fire_stingray", "Fire Stingray", 3, 3),
    BuiltinVehicle("white_cat", "White Cat", 4, 4),
    BuiltinVehicle("red_gazelle", "Red Gazelle", 5, 5),
    BuiltinVehicle("great_star", "Great Star", 6, 9),
    BuiltinVehicle("iron_tiger", "Iron Tiger", 7, 6),
    BuiltinVehicle("deep_claw", "Deep Claw", 8, 7),
    BuiltinVehicle("twin_noritta", "Twin Noritta", 9, 13),
    BuiltinVehicle("super_piranha", "Super Piranha", 10, 22),
    BuiltinVehicle("mighty_hurricane", "Mighty Hurricane", 11, 23),
    BuiltinVehicle("little_wyvern", "Little Wyvern", 12, 18),
    BuiltinVehicle("space_angler", "Space Angler", 13, 24),
    BuiltinVehicle("green_panther", "Green Panther", 14, 27),
    BuiltinVehicle("black_bull", "Black Bull", 15, 28),
    BuiltinVehicle("wild_boar", "Wild Boar", 16, 20),
    BuiltinVehicle("astro_robin", "Astro Robin", 17, 17),
    BuiltinVehicle("king_meteor", "King Meteor", 18, 21),
    BuiltinVehicle("queen_meteor", "Queen Meteor", 19, 15),
    BuiltinVehicle("wonder_wasp", "Wonder Wasp", 20, 14),
    BuiltinVehicle("hyper_speeder", "Hyper Speeder", 21, 26),
    BuiltinVehicle("death_anchor", "Death Anchor", 22, 19),
    BuiltinVehicle("crazy_bear", "Crazy Bear", 23, 8),
    BuiltinVehicle("night_thunder", "Night Thunder", 24, 12),
    BuiltinVehicle("big_fang", "Big Fang", 25, 10),
    BuiltinVehicle("mighty_typhoon", "Mighty Typhoon", 26, 25),
    BuiltinVehicle("mad_wolf", "Mad Wolf", 27, 11),
    BuiltinVehicle("sonic_phantom", "Sonic Phantom", 28, 29),
    BuiltinVehicle("blood_hawk", "Blood Hawk", 29, 16),
)


def vehicle_config_by_id(vehicle_id: str, *, config_root: Path) -> dict[str, object]:
    """Return vehicle metadata from optional YAML or the built-in stock registry."""

    registry_root = (config_root / LAYOUT.root).resolve()
    direct_path = (registry_root / vehicle_id).with_suffix(".yaml").resolve()
    if direct_path.is_relative_to(registry_root) and direct_path.is_file():
        return _load_vehicle_config(direct_path, ref=vehicle_id)

    matches: list[dict[str, object]] = []
    if registry_root.is_dir():
        for path in sorted(registry_root.rglob("*.yaml")):
            vehicle = _load_vehicle_config(path, ref=path.stem)
            if vehicle.get("id") == vehicle_id:
                matches.append(vehicle)
    if not matches:
        builtin = _builtin_vehicle_config(vehicle_id)
        if builtin is not None:
            return builtin
        raise FileNotFoundError(f"Vehicle registry id not found: {vehicle_id!r}")
    if len(matches) > 1:
        raise ValueError(f"Vehicle id {vehicle_id!r} is ambiguous")
    return matches[0]


def known_vehicle_ids(*, config_root: Path) -> tuple[str, ...]:
    """Return built-in plus optional YAML vehicle ids."""

    ids = {vehicle.id for vehicle in BUILTIN_VEHICLES}
    registry_root = (config_root / LAYOUT.root).resolve()
    if registry_root.is_dir():
        for path in sorted(registry_root.rglob("*.yaml")):
            vehicle = _load_vehicle_config(path, ref=path.stem)
            vehicle_id = vehicle.get("id")
            if isinstance(vehicle_id, str) and vehicle_id:
                ids.add(vehicle_id)
    return tuple(sorted(ids))


def resolve_engine_setting(
    vehicle_config: dict[str, object],
    raw_engine_setting: object,
    *,
    context: str,
) -> EngineSetting:
    """Resolve a named or raw integer engine-slider setting."""

    engine_settings = _engine_settings(vehicle_config)
    if isinstance(raw_engine_setting, str) and raw_engine_setting:
        resolved = try_resolve_engine_setting_id(
            vehicle_config,
            raw_engine_setting,
            context=context,
        )
        if resolved is not None:
            return resolved
        known = ", ".join(sorted(str(key) for key in engine_settings))
        raise ValueError(f"{context} unknown engine_setting {raw_engine_setting!r}; known: {known}")

    if isinstance(raw_engine_setting, bool) or not isinstance(raw_engine_setting, int):
        raise ValueError(
            "track_sampling.baseline.engine_setting must be a string id or raw integer"
        )
    _validate_engine_setting_raw_value(raw_engine_setting, context=context)
    for setting_id, setting_data in engine_settings.items():
        if (
            isinstance(setting_id, str)
            and _engine_setting_raw_value(setting_data) == raw_engine_setting
        ):
            return EngineSetting(id=setting_id, raw_value=raw_engine_setting)
    return EngineSetting(
        id=_generated_engine_setting_id(raw_engine_setting),
        raw_value=raw_engine_setting,
    )


def try_resolve_engine_setting_id(
    vehicle_config: dict[str, object],
    engine_setting: str,
    *,
    context: str,
) -> EngineSetting | None:
    """Resolve a named preset or generated raw engine id, if valid."""

    engine_settings = _engine_settings(vehicle_config)
    if engine_setting in engine_settings:
        return EngineSetting(
            id=engine_setting,
            raw_value=_engine_setting_raw_value(engine_settings[engine_setting]),
        )
    raw_value = _raw_value_from_generated_engine_setting_id(engine_setting)
    if raw_value is None:
        return None
    _validate_engine_setting_raw_value(raw_value, context=context)
    return EngineSetting(id=engine_setting, raw_value=raw_value)


def engine_setting_display_name(
    vehicle_config: dict[str, object],
    engine_setting: str,
) -> str:
    """Return a short display label for named or generated engine settings."""

    setting_data = _engine_settings(vehicle_config).get(engine_setting)
    if isinstance(setting_data, Mapping):
        display_name = _optional_str(setting_data.get("display_name"))
        if display_name is not None:
            return display_name
    raw_value = _raw_value_from_generated_engine_setting_id(engine_setting)
    if raw_value is not None:
        return f"Engine {raw_value}"
    return engine_setting.replace("_", " ").title()


def vehicle_machine_config(vehicle_config: dict[str, object]) -> dict[str, object] | None:
    """Return the RAM/menu machine metadata for a vehicle config."""

    machine = vehicle_config.get("machine")
    if not isinstance(machine, Mapping):
        return None
    return {str(key): value for key, value in machine.items() if isinstance(key, str)}


def _builtin_vehicle_config(vehicle_id: str) -> dict[str, object] | None:
    for vehicle in BUILTIN_VEHICLES:
        if vehicle.id == vehicle_id:
            return vehicle.config_data()
    return None


def _load_vehicle_config(path: Path, *, ref: str) -> dict[str, object]:
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"Vehicle registry entry {ref!r} must resolve to a mapping")
    vehicle = loaded.get("vehicle")
    if not isinstance(vehicle, dict):
        raise ValueError(f"Vehicle registry entry {ref!r} does not define a vehicle section")
    normalized = {str(key): value for key, value in vehicle.items() if isinstance(key, str)}
    unsupported_keys = sorted(DEFAULTS.unsupported_yaml_keys.intersection(normalized))
    if unsupported_keys:
        formatted_keys = ", ".join(unsupported_keys)
        raise ValueError(
            f"Vehicle registry entry {ref!r} uses unsupported baseline source keys: "
            f"{formatted_keys}. Add an exact source state for that vehicle or a neutral "
            "baseline generation path instead."
        )
    normalized.setdefault("engine_settings", DEFAULT_ENGINE_SETTINGS)
    return normalized


def _engine_settings(vehicle_config: dict[str, object]) -> Mapping[object, object]:
    engine_settings = vehicle_config.get("engine_settings", DEFAULT_ENGINE_SETTINGS)
    if not isinstance(engine_settings, Mapping):
        raise TypeError("vehicle engine_settings must be a mapping")
    return engine_settings


def _engine_setting_raw_value(setting_data: object) -> int | None:
    if not isinstance(setting_data, Mapping):
        return None
    raw_value = setting_data.get("raw_value")
    if isinstance(raw_value, bool):
        return None
    return raw_value if isinstance(raw_value, int) else None


def _generated_engine_setting_id(raw_value: int) -> str:
    return f"{DEFAULTS.generated_engine_prefix}{raw_value}"


def _raw_value_from_generated_engine_setting_id(engine_setting: str) -> int | None:
    prefix = DEFAULTS.generated_engine_prefix
    if not engine_setting.startswith(prefix):
        return None
    raw_value = engine_setting.removeprefix(prefix)
    if not raw_value.isdigit():
        return None
    return int(raw_value)


def _validate_engine_setting_raw_value(raw_value: int, *, context: str) -> None:
    if not 0 <= raw_value <= 100:
        raise ValueError(f"{context} raw engine_setting must be in [0, 100], got {raw_value}")


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
