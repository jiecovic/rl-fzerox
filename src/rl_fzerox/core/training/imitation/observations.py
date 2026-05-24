# src/rl_fzerox/core/training/imitation/observations.py
"""Teacher/student observation-view planning helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fzerox_emulator import ObservationImageRecipe
from rl_fzerox.core.domain.observation_components import StateComponentsSettings
from rl_fzerox.core.domain.observation_image import PresetResolutionChoice
from rl_fzerox.core.runtime_spec.renderers import DEFAULT_RENDERER, RendererName
from rl_fzerox.core.runtime_spec.schema import EnvConfig, ObservationConfig


@dataclass(frozen=True, slots=True)
class ObservationViewSpec:
    """One policy-facing observation view for imitation collection/training."""

    image_recipe: ObservationImageRecipe
    mode: str
    state_components: StateComponentsSettings | None
    independent_lean_buttons: bool = False

    @classmethod
    def from_observation_config(
        cls,
        config: ObservationConfig,
        *,
        renderer: RendererName = DEFAULT_RENDERER,
        independent_lean_buttons: bool = False,
    ) -> ObservationViewSpec:
        return cls(
            image_recipe=observation_image_recipe(config, renderer=renderer),
            mode=config.mode,
            state_components=config.state_components_data(),
            independent_lean_buttons=independent_lean_buttons,
        )

    @classmethod
    def from_env_config(
        cls,
        config: EnvConfig,
        *,
        renderer: RendererName = DEFAULT_RENDERER,
    ) -> ObservationViewSpec:
        return cls.from_observation_config(
            config.observation,
            renderer=renderer,
            independent_lean_buttons=config.action.independent_lean_buttons,
        )


@dataclass(frozen=True, slots=True)
class ObservationRenderPlan:
    """Deduplicated native image-render plan for one ordered set of views."""

    image_recipes: tuple[ObservationImageRecipe, ...]
    view_recipe_indices: tuple[int, ...]

    def recipe_index_for_view(self, view_index: int) -> int:
        return self.view_recipe_indices[view_index]


def observation_image_recipe(
    config: ObservationConfig,
    *,
    renderer: RendererName = DEFAULT_RENDERER,
) -> ObservationImageRecipe:
    """Project one runtime observation config into a native image recipe."""

    if isinstance(config.resolution, PresetResolutionChoice):
        return ObservationImageRecipe(
            preset=config.resolution.preset,
            frame_stack=int(config.frame_stack),
            stack_mode=config.stack_mode,
            minimap_layer=config.minimap_layer,
            resize_filter=config.resize_filter,
            minimap_resize_filter=config.minimap_resize_filter,
        )
    height, width = config.image_geometry(renderer=renderer)
    return ObservationImageRecipe(
        height=height,
        width=width,
        frame_stack=int(config.frame_stack),
        stack_mode=config.stack_mode,
        minimap_layer=config.minimap_layer,
        resize_filter=config.resize_filter,
        minimap_resize_filter=config.minimap_resize_filter,
    )


def plan_observation_renders(view_specs: Sequence[ObservationViewSpec]) -> ObservationRenderPlan:
    """Return one deduplicated native render plan for ordered observation views."""

    image_recipes: list[ObservationImageRecipe] = []
    recipe_indices_by_value: dict[ObservationImageRecipe, int] = {}
    view_recipe_indices: list[int] = []

    for view_spec in view_specs:
        recipe_index = recipe_indices_by_value.get(view_spec.image_recipe)
        if recipe_index is None:
            recipe_index = len(image_recipes)
            recipe_indices_by_value[view_spec.image_recipe] = recipe_index
            image_recipes.append(view_spec.image_recipe)
        view_recipe_indices.append(recipe_index)

    return ObservationRenderPlan(
        image_recipes=tuple(image_recipes),
        view_recipe_indices=tuple(view_recipe_indices),
    )
