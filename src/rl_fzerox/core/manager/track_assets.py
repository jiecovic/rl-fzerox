# src/rl_fzerox/core/manager/track_assets.py
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

_TRACK_ASSET_ROOT = Path(__file__).resolve().parents[4] / "local" / "minimap-assets"

_COURSE_MINIMAP_FILES = MappingProxyType(
    {
        "mute_city": "X_Minimap_Mute_City_Figure_Eight.png",
        "silence": "X_Minimap_Silence_High_Speed.png",
        "sand_ocean": "X_Minimap_Sand_Ocean_Pipe.png",
        "devils_forest": "X_Minimap_Devil's_Forest_Corkscrew.png",
        "big_blue": "X_Minimap_Big_Blue_Cylinder.png",
        "port_town": "X_Minimap_Port_Town_High_Jump.png",
        "sector_alpha": "X_Minimap_Sector_α_Inverted_Loop.png",
        "red_canyon": "X_Minimap_Red_Canyon_Multi_Jump.png",
        "devils_forest_2": "X_Minimap_Devil's_Forest_2_Up_and_Down.png",
        "mute_city_2": "X_Minimap_Mute_City_2_Technique.png",
        "big_blue_2": "X_Minimap_Big_Blue_2_Quick_Turn.png",
        "white_land": "X_Minimap_White_Land_Dangerous_Steps.png",
        "fire_field": "X_Minimap_Fire_Field_Zig-Zag_Jump.png",
        "silence_2": "X_Minimap_Silence_2_Wavy_Road.png",
        "sector_beta": "X_Minimap_Sector_β_Double_Somersault.png",
        "red_canyon_2": "X_Minimap_Red_Canyon_2_Slim_Line.png",
        "white_land_2": "X_Minimap_White_Land_2_Half_Pipe.png",
        "mute_city_3": "X_Minimap_Mute_City_3_Jumps_of_Doom.png",
        "rainbow_road": "X_Minimap_Rainbow_Road_Psychedelic_Experience.png",
        "devils_forest_3": "X_Minimap_Devil's_Forest_3_Mirror_Road.png",
        "space_plant": "X_Minimap_Space_Plant_Cylinder_&_High_Jump.png",
        "sand_ocean_2": "X_Minimap_Sand_Ocean_2_Wave_Panic.png",
        "port_town_2": "X_Minimap_Port_Town_2_Snake_Road.png",
        "big_hand": "X_Minimap_Big_Hand_Deadly_Curves.png",
    }
)

_CUP_BANNER_FILES = MappingProxyType(
    {
        "jack": "FZX_Jack_Cup.png",
        "queen": "FZX_Queen_Cup.png",
        "king": "FZX_King_Cup.png",
        "joker": "FZX_Joker_Cup.png",
        "x": "FZX_X_Cup.png",
    }
)


def course_minimap_path(course_id: str) -> Path | None:
    """Return the local minimap asset for one built-in course, if available."""

    return _asset_path(_COURSE_MINIMAP_FILES.get(course_id))


def cup_banner_path(cup_id: str) -> Path | None:
    """Return the local cup banner asset, if available."""

    return _asset_path(_CUP_BANNER_FILES.get(cup_id))


def _asset_path(filename: str | None) -> Path | None:
    if filename is None:
        return None
    asset_path = _TRACK_ASSET_ROOT / filename
    return asset_path if asset_path.is_file() else None
