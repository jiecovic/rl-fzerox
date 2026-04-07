# tests/core/policy/test_extractors.py
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

pytest.importorskip("stable_baselines3")

from rl_fzerox.core.policy import (
    FZeroXCnnExtractor,
    FZeroXCnnWideExtractor,
    resolve_extractor_class,
)


def test_fzerox_cnn_extractor_accepts_channels_last_observations() -> None:
    extractor = FZeroXCnnExtractor(
        spaces.Box(low=0, high=255, shape=(120, 160, 12), dtype=np.uint8),
        features_dim=512,
    )

    observations = torch.zeros((2, 120, 160, 12), dtype=torch.float32)
    features = extractor(observations)

    assert tuple(features.shape) == (2, 512)


def test_fzerox_cnn_extractor_accepts_channels_first_observations() -> None:
    extractor = FZeroXCnnExtractor(
        spaces.Box(low=0, high=255, shape=(12, 120, 160), dtype=np.uint8),
        features_dim=512,
    )

    observations = torch.zeros((2, 12, 120, 160), dtype=torch.float32)
    features = extractor(observations)

    assert tuple(features.shape) == (2, 512)


def test_fzerox_cnn_wide_extractor_accepts_channels_last_observations() -> None:
    extractor = FZeroXCnnWideExtractor(
        spaces.Box(low=0, high=255, shape=(120, 160, 12), dtype=np.uint8),
        features_dim=512,
    )

    observations = torch.zeros((2, 120, 160, 12), dtype=torch.float32)
    features = extractor(observations)

    assert tuple(features.shape) == (2, 512)


def test_resolve_extractor_class_supports_both_known_extractors() -> None:
    assert resolve_extractor_class("fzerox_cnn") is FZeroXCnnExtractor
    assert resolve_extractor_class("fzerox_cnn_wide") is FZeroXCnnWideExtractor
