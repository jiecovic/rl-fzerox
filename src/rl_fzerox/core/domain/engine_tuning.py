# src/rl_fzerox/core/domain/engine_tuning.py
"""Engine-tuning literal names shared across manager and runtime code."""

from __future__ import annotations

from typing import Literal, TypeAlias

# Bandit is the maintained run-manager backend. GP and MLP remain loadable for
# old configs and future experiments, but are not exposed for new managed runs
# until their telemetry and reset-distribution contracts are redesigned.
EngineTunerBackend: TypeAlias = Literal["bandit", "gaussian_process", "mlp_ensemble"]
EngineTunerObjective: TypeAlias = Literal["finish_time", "safe_finish_time", "finish_rate"]
