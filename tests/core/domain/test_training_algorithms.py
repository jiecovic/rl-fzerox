# tests/core/domain/test_training_algorithms.py
from __future__ import annotations

from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS


def test_training_algorithm_capability_groups_are_derived_from_supported_algorithms() -> None:
    all_algorithms = frozenset(
        (
            TRAINING_ALGORITHMS.maskable_hybrid_action_ppo,
            TRAINING_ALGORITHMS.maskable_hybrid_recurrent_ppo,
        )
    )

    assert TRAINING_ALGORITHMS.default == "maskable_hybrid_action_ppo"
    assert TRAINING_ALGORITHMS.maskable == all_algorithms
    assert TRAINING_ALGORITHMS.hybrid == all_algorithms
    assert TRAINING_ALGORITHMS.sb3x == all_algorithms
    assert TRAINING_ALGORITHMS.full_model_policy == all_algorithms
    assert TRAINING_ALGORITHMS.recurrent == frozenset(
        (TRAINING_ALGORITHMS.maskable_hybrid_recurrent_ppo,)
    )
