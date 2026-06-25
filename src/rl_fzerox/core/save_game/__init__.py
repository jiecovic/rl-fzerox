# src/rl_fzerox/core/save_game/__init__.py
"""Public facade for F-Zero X save-game inspection helpers.

This package keeps low-level save-RAM byte diffing, exploratory memory probes,
and known unlock decoding separate from Career Mode orchestration. Callers can
use this facade for stable data contracts while implementation modules keep the
format-specific details local.
"""

from rl_fzerox.core.save_game.probes import (
    MEMORY_PROBE_FORMATS,
    MEMORY_PROBE_REGIONS,
    MemoryProbeDefinition,
    MemoryProbeFormat,
    MemoryProbeReading,
    MemoryProbeRegion,
    MemoryProbeReport,
    MemoryProbeValue,
    collect_memory_probe_report,
    parse_memory_probe_definition,
    read_memory_probes,
)
from rl_fzerox.core.save_game.sram import (
    SaveRamBitDiff,
    SaveRamBitDiffReport,
    SaveRamDiff,
    SaveRamRangeDiff,
    SaveRamSummary,
    diff_save_ram,
    diff_save_ram_bits,
    summarize_save_ram,
)
from rl_fzerox.core.save_game.unlocks import (
    FZeroXSaveDecodeError,
    FZeroXUnlockState,
    FZeroXVehicleUnlockLayout,
    GpCupProgress,
    decode_fzerox_unlock_state,
    read_fzerox_unlock_state,
)

__all__ = [
    "FZeroXSaveDecodeError",
    "FZeroXUnlockState",
    "FZeroXVehicleUnlockLayout",
    "GpCupProgress",
    "MEMORY_PROBE_FORMATS",
    "MEMORY_PROBE_REGIONS",
    "MemoryProbeDefinition",
    "MemoryProbeFormat",
    "MemoryProbeReading",
    "MemoryProbeRegion",
    "MemoryProbeReport",
    "MemoryProbeValue",
    "SaveRamBitDiff",
    "SaveRamBitDiffReport",
    "SaveRamDiff",
    "SaveRamRangeDiff",
    "SaveRamSummary",
    "collect_memory_probe_report",
    "decode_fzerox_unlock_state",
    "diff_save_ram",
    "diff_save_ram_bits",
    "parse_memory_probe_definition",
    "read_fzerox_unlock_state",
    "read_memory_probes",
    "summarize_save_ram",
]
