# src/rl_fzerox/core/runtime_spec/roms.py
"""Resolve the local F-Zero X ROM used by runtime launches.

The emulator host remains the final authority for ROM compatibility. This
module only finds candidate ROM files in ``local/roms`` and applies the same
native header check before manager code writes a concrete runtime config.
"""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.runtime_spec.paths import project_root_dir


class FZeroXRomResolutionError(RuntimeError):
    """Raised when no compatible local F-Zero X ROM can be selected."""


def fzerox_rom_dir(root: Path | None = None) -> Path:
    """Return the repository-local ROM directory."""

    base = project_root_dir() if root is None else Path(root)
    return (base / "local" / "roms").resolve()


def fzerox_default_rom_path(root: Path | None = None) -> Path:
    """Return the documented example path for the US F-Zero X ROM."""

    return (fzerox_rom_dir(root) / "fzerox_usa.n64").resolve()


def resolve_fzerox_rom_path(root: Path | None = None) -> Path:
    """Return the first compatible US F-Zero X ROM found in ``local/roms``."""

    rom_dir = fzerox_rom_dir(root)
    rejected: list[str] = []
    for candidate in fzerox_rom_candidates(root):
        try:
            _validate_supported_rom_path(candidate)
        except (ImportError, OSError, ValueError) as exc:
            rejected.append(f"{candidate.name}: {exc}")
            continue
        return candidate.resolve()
    raise FZeroXRomResolutionError(_rom_resolution_error(rom_dir=rom_dir, rejected=rejected))


def find_fzerox_rom_path(root: Path | None = None) -> Path | None:
    """Return a compatible ROM path when one is available, otherwise ``None``."""

    try:
        return resolve_fzerox_rom_path(root)
    except FZeroXRomResolutionError:
        return None


def fzerox_rom_candidates(root: Path | None = None) -> tuple[Path, ...]:
    """Return deterministic N64 ROM candidates, preferring the documented name."""

    rom_dir = fzerox_rom_dir(root)
    if not rom_dir.is_dir():
        return ()
    supported_suffixes = {".n64", ".z64", ".v64"}
    candidates = sorted(
        (
            path.resolve()
            for path in rom_dir.iterdir()
            if path.is_file() and path.suffix.lower() in supported_suffixes
        ),
        key=lambda path: path.name.casefold(),
    )
    default_path = fzerox_default_rom_path(root)
    if default_path in candidates:
        return (default_path, *(path for path in candidates if path != default_path))
    return tuple(candidates)


def _validate_supported_rom_path(path: Path) -> None:
    from fzerox_emulator import validate_supported_rom_path

    validate_supported_rom_path(str(path))


def _rom_resolution_error(*, rom_dir: Path, rejected: list[str]) -> str:
    message = (
        f"No compatible US F-Zero X ROM found in {rom_dir}. "
        "Place a .n64, .z64, or .v64 ROM there; the filename can be arbitrary."
    )
    if rejected:
        return f"{message} Rejected candidates: {'; '.join(rejected)}"
    return message
