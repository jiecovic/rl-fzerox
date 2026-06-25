# src/rl_fzerox/core/envs/gym_runtime/__init__.py
"""Gym-runtime facade used by the public `FZeroXEnv` wrapper.

The runtime owns Gym reset/step/render calls and delegates backend mechanics to
engine subpackages. Keeping this package narrow makes the public env wrapper a
thin API boundary.
"""

from rl_fzerox.core.envs.gym_runtime.runtime import FZeroXEnvRuntime

__all__ = ["FZeroXEnvRuntime"]
