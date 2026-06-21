# src/rl_fzerox/core/career_mode/execution/attempt_timing.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CareerAttemptMenuJitter:
    """Deterministic player-like timing variance for replay attempts.

    The Career Runner should not patch game RNG RAM during an emulator lifetime.
    It varies opponent setup the way a real reset can: by waiting a reproducible
    number of neutral frames before the normal menu FSM starts pressing buttons.
    """

    max_neutral_frames: int = 90

    def frames_for(self, *, base_seed: int | None, attempt_id: str | None) -> int:
        if base_seed is None or attempt_id is None or self.max_neutral_frames <= 0:
            return 0
        digest = hashlib.blake2s(
            f"career_attempt_menu_jitter|{base_seed}|{attempt_id}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) % (self.max_neutral_frames + 1)


CAREER_ATTEMPT_MENU_JITTER = CareerAttemptMenuJitter()


def career_attempt_menu_jitter_frames(
    *,
    base_seed: int | None,
    attempt_id: str | None,
    jitter: CareerAttemptMenuJitter = CAREER_ATTEMPT_MENU_JITTER,
) -> int:
    return jitter.frames_for(base_seed=base_seed, attempt_id=attempt_id)
