# src/fzerox_emulator/repeat/requests.py
from __future__ import annotations

from typing import TYPE_CHECKING

from fzerox_emulator.base import ObservationImageRecipe

if TYPE_CHECKING:
    from fzerox_emulator._native import ObservationImageRequestDict


def native_observation_recipe(recipe: ObservationImageRecipe) -> ObservationImageRequestDict:
    payload: ObservationImageRequestDict = {
        "preset": "" if recipe.preset is None else recipe.preset,
        "frame_stack": recipe.frame_stack,
        "stack_mode": recipe.stack_mode,
        "minimap_layer": recipe.minimap_layer,
        "resize_filter": recipe.resize_filter,
        "minimap_resize_filter": recipe.minimap_resize_filter,
    }
    if recipe.height is not None:
        payload["height"] = recipe.height
    if recipe.width is not None:
        payload["width"] = recipe.width
    return payload
