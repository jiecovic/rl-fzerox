# src/rl_fzerox/ui/watch/view/panels/content/state_vector.py
"""State-vector side-panel facade and feature-set extraction helpers."""

from __future__ import annotations

from rl_fzerox.ui.watch.view.panels.content.state_vector_panel import policy_state_sections


def zeroed_state_features(info: dict[str, object]) -> frozenset[str]:
    raw_features = info.get("observation_zeroed_state_features")
    if isinstance(raw_features, tuple | list):
        return frozenset(str(feature) for feature in raw_features)
    return frozenset()


def watch_zeroed_state_features(info: dict[str, object]) -> frozenset[str]:
    raw_features = info.get("watch_zeroed_state_features")
    if isinstance(raw_features, tuple | list):
        return frozenset(str(feature) for feature in raw_features)
    return frozenset()


__all__ = ("policy_state_sections", "watch_zeroed_state_features", "zeroed_state_features")
