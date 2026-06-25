# src/rl_fzerox/ui/watch/view/panels/content/policy.py
"""Policy-output side-panel section for the Watch UI."""

from __future__ import annotations

from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.ui.watch.view.panels.core.format import _format_policy_action
from rl_fzerox.ui.watch.view.panels.core.lines import panel_line as _panel_line
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelSection


def policy_output_section(
    *,
    policy_action: ActionValue | None,
    policy_deterministic: bool | None,
    watch_device: str,
) -> PanelSection:
    return PanelSection(
        title="Policy Output",
        lines=[
            _panel_line(
                "Mode",
                _format_policy_deterministic(policy_deterministic),
                PALETTE.text_primary if policy_deterministic is not None else PALETTE.text_muted,
            ),
            _panel_line("Device", watch_device, PALETTE.text_primary),
            _panel_line(
                "Action",
                _format_policy_action(policy_action),
                PALETTE.text_primary,
            ),
        ],
    )


def _format_policy_deterministic(value: bool | None) -> str:
    if value is None:
        return "-"
    return "deterministic" if value else "stochastic"
