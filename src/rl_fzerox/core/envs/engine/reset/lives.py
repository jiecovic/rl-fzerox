# src/rl_fzerox/core/envs/engine/reset/lives.py
"""Reset-time GP spare-machine randomization."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from fzerox_emulator import EmulatorBackend, FZeroXTelemetry
from rl_fzerox.core.domain.race import RaceDifficultyName, race_difficulty_raw_value


@dataclass(frozen=True, slots=True)
class GpLivesRamLayout:
    """US rev0 RAM layout for `gPlayerLives` from the F-Zero X decomp."""

    player_lives_offset: int = 0x000E_5ED8
    player_count: int = 4
    bytes_per_count: int = 2


GP_LIVES_RAM = GpLivesRamLayout()
DEFAULT_GP_LIVES_BY_DIFFICULTY_RAW: tuple[int, ...] = (5, 4, 3, 2)


def randomize_gp_lives_on_reset(
    *,
    backend: EmulatorBackend,
    telemetry: FZeroXTelemetry | None,
    target_gp_difficulty: RaceDifficultyName | None = None,
    jitter_min: int,
    jitter_max: int,
    seed: int | None,
    info: dict[str, object],
) -> None:
    """Patch GP lives to the game default plus signed jitter for one reset."""

    if telemetry is None or not telemetry.in_race_mode:
        info["gp_lives_randomized"] = False
        info["gp_lives_randomization_skip_reason"] = "not_in_race"
        return
    if telemetry.game_mode_name != "gp_race":
        info["gp_lives_randomized"] = False
        info["gp_lives_randomization_skip_reason"] = "not_gp_race"
        return
    if seed is None:
        info["gp_lives_randomized"] = False
        info["gp_lives_randomization_skip_reason"] = "missing_seed"
        return

    difficulty_raw = (
        race_difficulty_raw_value(target_gp_difficulty)
        if target_gp_difficulty is not None
        else int(telemetry.difficulty_raw)
    )
    if difficulty_raw < 0 or difficulty_raw >= len(DEFAULT_GP_LIVES_BY_DIFFICULTY_RAW):
        info["gp_lives_randomized"] = False
        info["gp_lives_randomization_skip_reason"] = "unsupported_difficulty"
        info["gp_lives_difficulty_raw"] = difficulty_raw
        return

    base_lives = DEFAULT_GP_LIVES_BY_DIFFICULTY_RAW[difficulty_raw]
    jitter = Random(seed).randint(jitter_min, jitter_max)
    lives = max(0, base_lives + jitter)
    patch_gp_lives(backend=backend, lives=lives)
    info.update(
        {
            "gp_lives_randomized": True,
            "gp_lives_seed": seed,
            "gp_lives_difficulty_raw": difficulty_raw,
            "gp_lives_base": base_lives,
            "gp_lives_jitter": jitter,
            "gp_lives": lives,
        }
    )


def patch_gp_lives(*, backend: EmulatorBackend, lives: int) -> None:
    """Patch all player GP spare-machine counters in live system RAM."""

    count_bytes = max(0, int(lives)).to_bytes(
        GP_LIVES_RAM.bytes_per_count,
        byteorder="little",
        signed=True,
    )
    backend.write_system_ram(
        GP_LIVES_RAM.player_lives_offset,
        count_bytes * GP_LIVES_RAM.player_count,
    )
