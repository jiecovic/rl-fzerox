# src/rl_fzerox/core/envs/engine/observation.py
from __future__ import annotations

from dataclasses import dataclass

from gymnasium import spaces

from fzerox_emulator import EmulatorBackend, FZeroXTelemetry, ObservationSpec
from fzerox_emulator.arrays import ObservationFrame
from rl_fzerox.core.domain.observation_components import (
    ActionHistoryControlName,
    StateComponentsSettings,
)
from rl_fzerox.core.envs.engine.controls import ControlStateTracker
from rl_fzerox.core.envs.engine.info import set_observation_info
from rl_fzerox.core.envs.observations import (
    ObservationValue,
    action_history_settings_for_observation,
    build_observation,
    build_observation_space,
)
from rl_fzerox.core.runtime_spec.renderers import RendererName
from rl_fzerox.core.runtime_spec.schema import EnvConfig


@dataclass(slots=True)
class EngineObservationBuilder:
    """Own image/state observation setup for one env engine."""

    backend: EmulatorBackend
    config: EnvConfig
    renderer: RendererName
    spec: ObservationSpec
    state_components: StateComponentsSettings | None
    independent_lean_buttons: bool
    action_history_len: int | None
    action_history_controls: tuple[ActionHistoryControlName, ...]
    space: spaces.Space

    @classmethod
    def from_engine_config(
        cls,
        *,
        backend: EmulatorBackend,
        config: EnvConfig,
        renderer: RendererName,
    ) -> EngineObservationBuilder:
        spec = backend.observation_spec(
            **config.observation.native_resolution_kwargs(renderer=renderer)
        )
        state_components = config.observation.state_components_data()
        independent_lean_buttons = config.action.independent_lean_buttons
        action_history_len, action_history_controls = action_history_settings_for_observation(
            state_components=state_components,
        )
        space = build_observation_space(
            spec,
            frame_stack=config.observation.frame_stack,
            stack_mode=config.observation.stack_mode,
            minimap_layer=config.observation.minimap_layer,
            mode=config.observation.mode,
            state_components=state_components,
            independent_lean_buttons=independent_lean_buttons,
        )
        return cls(
            backend=backend,
            config=config,
            renderer=renderer,
            spec=spec,
            state_components=state_components,
            independent_lean_buttons=independent_lean_buttons,
            action_history_len=action_history_len,
            action_history_controls=action_history_controls,
            space=space,
        )

    def render_image(self) -> ObservationFrame:
        return self.backend.render_observation(
            frame_stack=self.config.observation.frame_stack,
            stack_mode=self.config.observation.stack_mode,
            minimap_layer=self.config.observation.minimap_layer,
            resize_filter=self.config.observation.resize_filter,
            minimap_resize_filter=self.config.observation.minimap_resize_filter,
            **self.config.observation.native_resolution_kwargs(renderer=self.renderer),
        )

    def build_observation(
        self,
        *,
        image: ObservationFrame,
        telemetry: FZeroXTelemetry | None,
        control_state: ControlStateTracker,
    ) -> ObservationValue:
        """Build the policy observation from the rendered image plus control context."""

        return build_observation(
            image=image,
            telemetry=telemetry,
            mode=self.config.observation.mode,
            action_history=control_state.action_history_fields(),
            state_components=self.state_components,
            independent_lean_buttons=self.independent_lean_buttons,
        )

    def set_info(
        self,
        info: dict[str, object],
        *,
        image_shape: tuple[int, ...],
    ) -> None:
        set_observation_info(
            info,
            observation_shape=image_shape,
            observation_spec=self.spec,
            frame_stack=self.config.observation.frame_stack,
            observation_stack_mode=self.config.observation.stack_mode,
            observation_minimap_layer=self.config.observation.minimap_layer,
            observation_resize_filter=self.config.observation.resize_filter,
            observation_minimap_resize_filter=self.config.observation.minimap_resize_filter,
            observation_mode=self.config.observation.mode,
            action_history_len=self.action_history_len,
            action_history_controls=self.action_history_controls,
            observation_state_components=self.state_components,
            independent_lean_buttons=self.independent_lean_buttons,
        )
