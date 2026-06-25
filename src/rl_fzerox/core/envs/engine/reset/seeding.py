# src/rl_fzerox/core/envs/engine/reset/seeding.py
"""Deterministic seed derivation for reset-time subsystems.

Track sampling, reward episode state, action dropout, GP lives, and game RNG all
derive from separate seed domains to avoid accidental coupling.
"""
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.seed import derive_seed


@dataclass(frozen=True, slots=True)
class EngineSeedDomains:
    """Domain separators for independent per-env RNG streams."""

    reset_rng: int = 0xD6E8_2BC9_2A5F_1873
    reward_milestone_phase: int = 0xA409_3822_299F_31D0
    track_sampling: int = 0x35E7_40D8_FF53_42B1
    action_episode_mask: int = 0x7403_BCE1_0F68_964A
    gp_lives_jitter: int = 0x4430_6A1E_13E1_F7E2


ENGINE_SEED_DOMAINS = EngineSeedDomains()


@dataclass(slots=True)
class EngineResetSeeds:
    """Track reset seed state and derive independent episode sub-seeds."""

    seed_base: int | None = None
    reset_count: int = 0

    def remember_reset_seed(self, seed: int | None) -> None:
        if seed is not None:
            self.seed_base = seed

    def advance_reset_count(self) -> None:
        self.reset_count += 1

    def reset_rng_seed(self, seed: int | None) -> int | None:
        return self._derive(seed, ENGINE_SEED_DOMAINS.reset_rng)

    def reward_episode_seed(self, seed: int | None) -> int | None:
        return self._derive(seed, ENGINE_SEED_DOMAINS.reward_milestone_phase)

    def track_sampling_seed(self, seed: int | None) -> int | None:
        return self._derive(seed, ENGINE_SEED_DOMAINS.track_sampling)

    def action_episode_mask_seed(self, seed: int | None) -> int | None:
        return self._derive(seed, ENGINE_SEED_DOMAINS.action_episode_mask)

    def gp_lives_jitter_seed(self, seed: int | None) -> int | None:
        return self._derive(seed, ENGINE_SEED_DOMAINS.gp_lives_jitter)

    def _derive(self, seed: int | None, domain: int) -> int | None:
        seed_base = seed if seed is not None else self.seed_base
        if seed_base is None:
            return None
        return derive_seed(seed_base, domain, self.reset_count)
