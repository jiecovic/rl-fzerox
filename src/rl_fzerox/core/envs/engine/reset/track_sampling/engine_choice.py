# src/rl_fzerox/core/envs/engine/reset/track_sampling/engine_choice.py
from __future__ import annotations

from random import Random

from rl_fzerox.core.engine_tuning import (
    EngineTuningChoice,
    EngineTuningResetSampler,
    EngineTuningSelectionMode,
)
from rl_fzerox.core.engine_tuning.contexts import engine_tuning_context_for_entry
from rl_fzerox.core.runtime_spec.schema import (
    AdaptiveEngineTuningConfig,
    TrackSamplingEntryConfig,
)
from rl_fzerox.core.runtime_spec.vehicle_catalog import EngineSetting, resolve_engine_setting


def resolve_entry_engine_setting(
    entry: TrackSamplingEntryConfig,
    *,
    seed: int | None,
    engine_choice: EngineTuningChoice | None,
) -> EngineSetting | None:
    raw_value = (
        engine_choice.engine_setting_raw_value
        if engine_choice is not None
        else target_engine_setting_raw_value(entry, seed=seed)
    )
    if raw_value is None:
        return None
    return resolve_engine_setting(
        raw_value,
        context=f"track sampling entry {entry.id!r}",
    )


def choose_engine_tuning(
    entry: TrackSamplingEntryConfig,
    *,
    config: AdaptiveEngineTuningConfig,
    sampler: EngineTuningResetSampler | None,
    seed: int | None,
    selection: EngineTuningSelectionMode,
) -> EngineTuningChoice | None:
    if not config.enabled or sampler is None:
        return None
    return sampler.choose(
        engine_tuning_context_for_entry(entry),
        selection=selection,
        seed=seed,
    )


def target_engine_setting_raw_value(
    entry: TrackSamplingEntryConfig,
    *,
    seed: int | None,
) -> int | None:
    minimum = entry.engine_setting_min_raw_value
    maximum = entry.engine_setting_max_raw_value
    if minimum is None and maximum is None:
        return entry.engine_setting_raw_value
    if minimum is None or maximum is None:
        raise ValueError(f"track sampling entry {entry.id!r} must define both engine range bounds")
    if minimum > maximum:
        raise ValueError(
            f"track sampling entry {entry.id!r} has engine range min > max: {minimum} > {maximum}"
        )
    if minimum == maximum:
        return int(minimum)
    rng = Random(seed) if seed is not None else None
    return (
        rng.randint(int(minimum), int(maximum))
        if rng is not None
        else Random().randint(int(minimum), int(maximum))
    )
