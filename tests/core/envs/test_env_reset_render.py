# tests/core/envs/test_env_reset_render.py

import pytest
from gymnasium.spaces import Box

from fzerox_emulator.arrays import ObservationFrame
from rl_fzerox.core.domain.observation_image import PresetResolutionChoice
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.observations import ObservationStackMode
from rl_fzerox.core.envs.observations.state import state_feature_names
from rl_fzerox.core.runtime_spec.schema import (
    EnvConfig,
    ObservationConfig,
)
from tests.core.envs.helpers import (
    image_obs as _image_obs,
)
from tests.support.fakes import SyntheticBackend


def test_env_reset_passes_preset_to_render_observation() -> None:
    class ObservationPresetBackend(SyntheticBackend):
        def __init__(self) -> None:
            super().__init__()
            self.render_observation_calls: list[
                tuple[str | None, int | None, int | None, int, str, bool]
            ] = []

        def render_observation(
            self,
            *,
            preset: str | None = None,
            height: int | None = None,
            width: int | None = None,
            frame_stack: int,
            stack_mode: ObservationStackMode = "rgb",
            minimap_layer: bool = False,
            resize_filter: object = "nearest",
            minimap_resize_filter: object = "nearest",
        ) -> ObservationFrame:
            _ = (resize_filter, minimap_resize_filter)
            self.render_observation_calls.append(
                (preset, height, width, frame_stack, stack_mode, minimap_layer)
            )
            return super().render_observation(
                preset=preset,
                height=height,
                width=width,
                frame_stack=frame_stack,
                stack_mode=stack_mode,
                minimap_layer=minimap_layer,
                resize_filter=resize_filter,
                minimap_resize_filter=minimap_resize_filter,
            )

    backend = ObservationPresetBackend()

    env = FZeroXEnv(backend=backend, config=EnvConfig(action_repeat=1))

    obs, info = env.reset(seed=13)
    obs = _image_obs(obs)

    assert obs.shape == (84, 84, 12)
    assert info["observation_frame_shape"] == (84, 84, 3)
    assert backend.render_observation_calls == [("crop_84x84", None, None, 4, "rgb", False)]


def test_env_reset_passes_custom_resolution_to_render_observation() -> None:
    class CustomResolutionBackend(SyntheticBackend):
        def __init__(self) -> None:
            super().__init__()
            self.render_observation_calls: list[tuple[str | None, int | None, int | None]] = []

        def render_observation(
            self,
            *,
            preset: str | None = None,
            height: int | None = None,
            width: int | None = None,
            frame_stack: int,
            stack_mode: ObservationStackMode = "rgb",
            minimap_layer: bool = False,
            resize_filter: object = "nearest",
            minimap_resize_filter: object = "nearest",
        ) -> ObservationFrame:
            _ = (frame_stack, stack_mode, minimap_layer, resize_filter, minimap_resize_filter)
            self.render_observation_calls.append((preset, height, width))
            return super().render_observation(
                preset=preset,
                height=height,
                width=width,
                frame_stack=frame_stack,
                stack_mode=stack_mode,
                minimap_layer=minimap_layer,
                resize_filter=resize_filter,
                minimap_resize_filter=minimap_resize_filter,
            )

    backend = CustomResolutionBackend()
    config = EnvConfig.model_validate(
        {
            "action_repeat": 1,
            "observation": {
                "mode": "image",
                "resolution": {"mode": "custom", "height": 72, "width": 96},
                "frame_stack": 4,
                "stack_mode": "rgb",
                "minimap_layer": False,
                "resize_filter": "nearest",
                "minimap_resize_filter": "nearest",
            },
        }
    )

    env = FZeroXEnv(backend=backend, config=config)

    obs, info = env.reset(seed=13)
    obs = _image_obs(obs)

    assert obs.shape == (72, 96, 12)
    assert info["observation_frame_shape"] == (72, 96, 3)
    assert backend.render_observation_calls == [(None, 72, 96)]


def test_env_reset_uses_rgb_stack_shape() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            observation=ObservationConfig(
                resolution=PresetResolutionChoice(preset="crop_84x84"),
                frame_stack=4,
                stack_mode="rgb",
            ),
        ),
    )

    obs, info = env.reset(seed=13)
    obs = _image_obs(obs)

    assert obs.shape == (84, 84, 12)
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (84, 84, 12)
    assert info["observation_stack"] == 4
    assert info["observation_stack_mode"] == "rgb"


def test_env_reset_uses_optional_minimap_layer_shape() -> None:
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            observation=ObservationConfig(
                resolution=PresetResolutionChoice(preset="crop_84x84"),
                frame_stack=4,
                stack_mode="rgb",
                minimap_layer=True,
            ),
        ),
    )

    obs, info = env.reset(seed=13)
    obs = _image_obs(obs)

    assert obs.shape == (84, 84, 13)
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (84, 84, 13)
    assert info["observation_minimap_layer"] is True


def test_env_config_accepts_split_lean_history_feature_selection() -> None:
    config = EnvConfig.model_validate(
        {
            "action": {
                "lean_output_mode": "four_way_categorical",
            },
            "observation": {
                "mode": "image_state",
                "state_components": [
                    {
                        "name": "control_history",
                        "length": 1,
                        "controls": ["lean"],
                        "included_features": [
                            "control_history.prev_lean_left_1",
                            "control_history.prev_lean_right_1",
                        ],
                    }
                ],
            },
        }
    )
    components = config.observation.state_components
    assert components is not None

    assert state_feature_names(
        state_components=tuple(component.data() for component in components),
        split_lean_history=config.action.runtime().split_lean_history,
    ) == (
        "control_history.prev_lean_left_1",
        "control_history.prev_lean_right_1",
    )


def test_env_config_rejects_split_lean_history_for_three_way_lean() -> None:
    with pytest.raises(ValueError, match="control_history.prev_lean_left_1"):
        EnvConfig.model_validate(
            {
                "observation": {
                    "mode": "image_state",
                    "state_components": [
                        {
                            "name": "control_history",
                            "length": 1,
                            "controls": ["lean"],
                            "included_features": ["control_history.prev_lean_left_1"],
                        }
                    ],
                },
            }
        )


def test_env_render_uses_cropped_aspect_corrected_display_size() -> None:
    backend = SyntheticBackend(width=640, height=240)
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(observation=ObservationConfig(frame_stack=4)),
    )

    env.reset(seed=1)
    frame = env.render()

    assert frame.shape == (444, 592, 3)
