# src/rl_fzerox/core/training/session/observation_augmentation.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from rl_fzerox.core.config.schema import EnvConfig, TrainConfig
from rl_fzerox.core.envs.observations import mask_observation_state, state_feature_names
from rl_fzerox.core.seed import derive_seed


@dataclass(frozen=True, slots=True)
class _TrainingAugmentationSeedDomains:
    """Domain separators for train-only observation augmentation RNG streams."""

    state_feature_dropout: int = 0x34C2_2E5B_7B67_021D


_TRAINING_AUGMENTATION_SEED_DOMAINS = _TrainingAugmentationSeedDomains()


@dataclass(frozen=True, slots=True)
class _StateFeatureDropoutGroup:
    feature_indices: tuple[int, ...]
    dropout_prob: float


def maybe_wrap_training_observation_augmentation(
    env: gym.Env,
    *,
    env_config: EnvConfig,
    train_config: TrainConfig,
) -> gym.Env:
    """Apply train-only observation augmentations without changing env spaces."""

    dropout_groups = _state_feature_dropout_groups(env_config=env_config, train_config=train_config)
    if not dropout_groups:
        return env

    return EpisodeStateFeatureDropoutWrapper(
        env,
        dropout_groups=dropout_groups,
    )


class EpisodeStateFeatureDropoutWrapper(gym.ObservationWrapper):
    """Randomly zero configured state-feature groups for whole training episodes."""

    def __init__(
        self,
        env: gym.Env,
        *,
        dropout_groups: Sequence[_StateFeatureDropoutGroup],
    ) -> None:
        super().__init__(env)
        self._dropout_groups = tuple(dropout_groups)
        self._rng = np.random.default_rng()
        self._dropped_feature_indices: tuple[int, ...] = ()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ):
        observation, info = self.env.reset(seed=seed, options=options)
        derived_seed = derive_seed(
            seed,
            _TRAINING_AUGMENTATION_SEED_DOMAINS.state_feature_dropout,
        )
        if derived_seed is not None:
            self._rng = np.random.default_rng(derived_seed)
        dropped_indices: set[int] = set()
        for group in self._dropout_groups:
            if group.dropout_prob >= 1.0:
                dropped_indices.update(group.feature_indices)
                continue
            if self._rng.random() < group.dropout_prob:
                dropped_indices.update(group.feature_indices)
        self._dropped_feature_indices = tuple(sorted(dropped_indices))
        return self.observation(observation), info

    def observation(self, observation: object) -> object:
        if not self._dropped_feature_indices:
            return observation
        if not isinstance(observation, dict):
            return observation
        return mask_observation_state(
            observation,
            feature_indices=self._dropped_feature_indices,
        )


def _state_feature_dropout_groups(
    *,
    env_config: EnvConfig,
    train_config: TrainConfig,
) -> tuple[_StateFeatureDropoutGroup, ...]:
    index_by_feature = {
        feature_name: index
        for index, feature_name in enumerate(_observation_state_feature_names(env_config))
    }
    merged_groups: dict[tuple[str, ...], float] = {}
    ordered_feature_names = tuple(index_by_feature)
    for group in train_config.state_feature_dropout_groups:
        active_names = tuple(
            feature_name
            for feature_name in ordered_feature_names
            if feature_name in group.feature_names
        )
        if not active_names or group.dropout_prob <= 0.0:
            continue
        merged_groups[active_names] = max(
            merged_groups.get(active_names, 0.0),
            float(group.dropout_prob),
        )
    return tuple(
        _StateFeatureDropoutGroup(
            feature_indices=tuple(index_by_feature[feature_name] for feature_name in feature_names),
            dropout_prob=dropout_prob,
        )
        for feature_names, dropout_prob in merged_groups.items()
    )


def _observation_state_feature_names(config: EnvConfig) -> tuple[str, ...]:
    observation_config = config.observation
    state_components = observation_config.state_components_data()
    if state_components is None:
        return ()
    return state_feature_names(
        state_components=state_components,
        independent_lean_buttons=config.action.independent_lean_buttons,
    )
