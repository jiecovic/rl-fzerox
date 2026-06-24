# tests/core/domain/test_action_values.py
from __future__ import annotations

import pytest

from rl_fzerox.core.domain.action_adapters import ActionAdapterName as LegacyActionAdapterName
from rl_fzerox.core.domain.action_values import (
    compile_action_mask_values as legacy_compile_action_mask_values,
)
from rl_fzerox.core.domain.actions import (
    ACTION_BRANCH_SPECS,
    DISCRETE_ACTION_BRANCH_VALUES,
    ActionAdapterName,
    compile_action_mask_values,
)


def test_legacy_action_values_facade_reexports_public_helpers() -> None:
    assert LegacyActionAdapterName is ActionAdapterName
    assert legacy_compile_action_mask_values is compile_action_mask_values


def test_action_branch_specs_capture_named_and_fixed_width_branches() -> None:
    assert ACTION_BRANCH_SPECS["steer"].index_count is None
    assert ACTION_BRANCH_SPECS["spin"].index_count == 3
    assert ACTION_BRANCH_SPECS["boost"].named_values == ("idle", "engaged")
    assert DISCRETE_ACTION_BRANCH_VALUES["pitch"] == (
        "down_full",
        "down",
        "neutral",
        "up",
        "up_full",
    )


def test_compile_action_mask_values_expands_unrestricted_preset() -> None:
    assert compile_action_mask_values("pitch", "unrestricted") == (0, 1, 2, 3, 4)
    assert compile_action_mask_values("spin", "unrestricted") == (0, 1, 2)


def test_compile_action_mask_values_accepts_named_branch_values() -> None:
    assert compile_action_mask_values("lean", ("left", "right")) == (1, 2)
    assert compile_action_mask_values("boost", ("idle", "engaged")) == (0, 1)


def test_compile_action_mask_values_accepts_numeric_indices() -> None:
    assert compile_action_mask_values("pitch", (0, 2, 4)) == (0, 2, 4)
    assert compile_action_mask_values("custom_branch", (0, 3)) == (0, 3)


@pytest.mark.parametrize(
    ("branch_name", "values", "message"),
    (
        ("boost", (True,), "Boolean mask value"),
        ("boost", ("left",), "Unknown mask value"),
        ("boost", (0, 0), "contains duplicates"),
        ("boost", (2,), "out of range"),
        ("spin", (3,), "out of range"),
        ("steer", "unrestricted", "not supported"),
        ("custom_branch", ("idle",), "Named mask values are not supported"),
        ("custom_branch", "unrestricted", "not supported"),
    ),
)
def test_compile_action_mask_values_rejects_invalid_values(
    branch_name: str,
    values: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_action_mask_values(branch_name, values)  # pyright: ignore[reportArgumentType]
