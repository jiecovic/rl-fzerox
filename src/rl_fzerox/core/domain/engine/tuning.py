# src/rl_fzerox/core/domain/engine/tuning.py
"""Engine-tuning literal names shared across manager and runtime code."""

from __future__ import annotations

from typing import Literal

# Bandit is the maintained run-manager backend. GP and MLP remain loadable for
# old configs and future experiments, but are not exposed for new managed runs
# until their telemetry and reset-distribution contracts are redesigned.
type EngineTunerBackend = Literal["bandit", "gaussian_process", "mlp_ensemble"]
type EngineTunerObjective = Literal["finish_time", "safe_finish_time", "finish_rate"]
