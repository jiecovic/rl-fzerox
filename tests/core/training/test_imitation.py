# tests/core/training/test_imitation.py
from __future__ import annotations

import numpy as np

from fzerox_emulator import ObservationImageRecipe
from rl_fzerox.core.runtime_spec.schema import EnvConfig, ObservationConfig
from rl_fzerox.core.training.imitation import (
    BehaviorCloningBatch,
    BehaviorCloningSample,
    CanonicalControlIntent,
    ObservationViewSpec,
    observation_image_recipe,
    plan_observation_renders,
)


def test_observation_image_recipe_preserves_preset_runtime_settings() -> None:
    recipe = observation_image_recipe(
        ObservationConfig(
            resolution={"mode": "preset", "preset": "crop_84x84"},
            frame_stack=3,
            stack_mode="gray",
            minimap_layer=True,
            resize_filter="bilinear",
            minimap_resize_filter="nearest",
        )
    )

    assert recipe == ObservationImageRecipe(
        preset="crop_84x84",
        frame_stack=3,
        stack_mode="gray",
        minimap_layer=True,
        resize_filter="bilinear",
        minimap_resize_filter="nearest",
    )


def test_observation_view_spec_from_env_config_carries_state_metadata() -> None:
    view_spec = ObservationViewSpec.from_env_config(
        EnvConfig.model_validate(
            {
                "action": {"independent_lean_buttons": True},
                "observation": {
                    "mode": "image_state",
                    "resolution": {"mode": "custom", "height": 72, "width": 96},
                    "frame_stack": 2,
                    "stack_mode": "rgb",
                    "state_components": [
                        "vehicle_state",
                        {"control_history": {"length": 3, "controls": ["steer", "gas"]}},
                    ],
                },
            }
        )
    )

    assert view_spec.independent_lean_buttons is True
    assert view_spec.image_recipe == ObservationImageRecipe(
        height=72,
        width=96,
        frame_stack=2,
        stack_mode="rgb",
    )
    assert view_spec.state_components is not None
    assert tuple(component.name for component in view_spec.state_components) == (
        "vehicle_state",
        "control_history",
    )


def test_plan_observation_renders_deduplicates_identical_image_recipes() -> None:
    shared_recipe = ObservationImageRecipe(preset="crop_84x84", frame_stack=4)
    unique_recipe = ObservationImageRecipe(height=72, width=96, frame_stack=2)
    render_plan = plan_observation_renders(
        (
            ObservationViewSpec(
                image_recipe=shared_recipe,
                mode="image",
                state_components=None,
            ),
            ObservationViewSpec(
                image_recipe=shared_recipe,
                mode="image_state",
                state_components=(),
            ),
            ObservationViewSpec(
                image_recipe=unique_recipe,
                mode="image",
                state_components=None,
            ),
        )
    )

    assert render_plan.image_recipes == (shared_recipe, unique_recipe)
    assert render_plan.view_recipe_indices == (0, 0, 1)
    assert render_plan.recipe_index_for_view(2) == 1


def test_canonical_control_intent_clamps_to_expected_ranges() -> None:
    clamped = CanonicalControlIntent(
        steer=2.0,
        gas=-1.0,
        air_brake=1.5,
        boost=0.5,
        lean=-2.0,
        pitch=3.0,
    ).clamped()

    assert clamped == CanonicalControlIntent(
        steer=1.0,
        gas=0.0,
        air_brake=1.0,
        boost=0.5,
        lean=-1.0,
        pitch=1.0,
    )


def test_behavior_cloning_batch_reports_sample_count() -> None:
    sample = BehaviorCloningSample(
        student_observation=np.zeros((84, 84, 3), dtype=np.uint8),
        teacher_action=np.array([1, 2, 3], dtype=np.int64),
    )
    batch = BehaviorCloningBatch(samples=(sample, sample))

    assert len(batch) == 2
