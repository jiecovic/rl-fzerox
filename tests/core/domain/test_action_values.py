# tests/core/domain/test_action_values.py
from __future__ import annotations

import pytest

from rl_fzerox.core.domain.action_values import compile_action_mask_values


def test_compile_action_mask_values_expands_unrestricted_preset() -> None:
    assert compile_action_mask_values("pitch", "unrestricted") == (0, 1, 2, 3, 4)


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
        ("custom_branch", ("idle",), "Named mask values are not supported"),
    ),
)
def test_compile_action_mask_values_rejects_invalid_values(
    branch_name: str,
    values: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_action_mask_values(branch_name, values)  # pyright: ignore[reportArgumentType]
