# tests/core/manager/manager_training_support.py
from __future__ import annotations

from typing import Literal


def _manager_config_data_with_control_history_features(
    included_features: tuple[str, ...],
    *,
    lean_output_mode: Literal["three_way", "four_way_categorical", "independent_buttons"],
) -> dict[str, object]:
    return {
        "action": {"lean_output_mode": lean_output_mode},
        "observation": {
            "state_components": [
                {
                    "name": "control_history",
                    "length": 1,
                    "controls": ["lean"],
                    "included_features": list(included_features),
                }
            ],
            "state_feature_dropouts": [],
        },
    }
