# src/rl_fzerox/core/envs/engine/stepping/__init__.py
"""Step assembly facade shared by Gym and policy-drive runtimes.

Backend step results are normalized here into Gym-style observations, rewards,
terminal flags, info dictionaries, and optional watch frames. Reset selection
and policy action decoding stay outside this package.
"""

from rl_fzerox.core.envs.engine.stepping.assembly import (
    EngineStepAssembler,
    EnvStepAssembly,
    EnvStepRequest,
    set_episode_boost_pad_info,
)
from rl_fzerox.core.envs.engine.stepping.result import WatchEnvStep

__all__ = [
    "EngineStepAssembler",
    "EnvStepAssembly",
    "EnvStepRequest",
    "WatchEnvStep",
    "set_episode_boost_pad_info",
]
