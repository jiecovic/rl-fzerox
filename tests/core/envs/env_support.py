# tests/core/envs/env_support.py


from rl_fzerox.core.runtime_spec.schema import (
    ObservationStateComponentConfig,
)


def _state_components(*components: object) -> tuple[ObservationStateComponentConfig, ...]:
    return tuple(
        ObservationStateComponentConfig.model_validate(component) for component in components
    )
