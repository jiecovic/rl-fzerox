# src/rl_fzerox/core/policy/auxiliary_state/actor_regularization/__init__.py
"""Facade for actor-side auxiliary regularization helpers and mixins.

Regularization is split into distribution readers, pure tensor losses, metric
formatting, and the policy mixin that combines them during action evaluation.
"""

from __future__ import annotations

from rl_fzerox.core.policy.auxiliary_state.actor_regularization.distributions import (
    _AxisDistributionStats,
    _categorical_lean_expected_signed_values,
    _split_lean_expected_signed_values,
)
from rl_fzerox.core.policy.auxiliary_state.actor_regularization.losses import (
    _signed_balance_loss,
    _SignedBalanceLoss,
    _std_cap_loss,
)
from rl_fzerox.core.policy.auxiliary_state.actor_regularization.mixin import (
    _ActorRegularizationMixin,
)

__all__ = [
    "_ActorRegularizationMixin",
    "_AxisDistributionStats",
    "_SignedBalanceLoss",
    "_categorical_lean_expected_signed_values",
    "_signed_balance_loss",
    "_split_lean_expected_signed_values",
    "_std_cap_loss",
]
