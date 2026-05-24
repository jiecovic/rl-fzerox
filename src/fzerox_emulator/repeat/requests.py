# src/fzerox_emulator/repeat/requests.py
from __future__ import annotations

from typing import TYPE_CHECKING

from fzerox_emulator.base import ObservationImageRecipe

if TYPE_CHECKING:
    from fzerox_emulator._native import ObservationImageRequestDict


def native_observation_recipe(recipe: ObservationImageRecipe) -> ObservationImageRequestDict:
    preset, height, width = recipe.normalized_resolution()
    payload: ObservationImageRequestDict = {
        "preset": "" if preset is None else preset,
        "frame_stack": recipe.frame_stack,
        "stack_mode": recipe.stack_mode,
        "minimap_layer": recipe.minimap_layer,
        "resize_filter": recipe.resize_filter,
        "minimap_resize_filter": recipe.minimap_resize_filter,
    }
    if height is not None:
        payload["height"] = height
    if width is not None:
        payload["width"] = width
    return payload
