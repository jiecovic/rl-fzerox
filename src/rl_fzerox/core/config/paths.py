# src/rl_fzerox/core/config/paths.py
from __future__ import annotations

from pathlib import Path


def project_root_dir() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parents[4]


def config_root_dir() -> Path:
    """Return the repository config root used for Hydra composition."""

    return project_root_dir() / "conf"


def resolve_config_path_value(raw_path: str, *, config_dir: Path) -> Path:
    """Resolve a config path using repo-root rules for repo configs.

    Paths inside the repository config tree resolve relative to the repository
    root so local configs can use stable `local/...` paths without `../../`.
    External config files still resolve relative to their own directory.
    """

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    return (config_path_resolution_base_dir(config_dir) / path).resolve()


def config_path_resolution_base_dir(config_dir: Path) -> Path:
    """Return the base directory used to resolve relative config paths.

    Repository-owned configs deliberately resolve from the project root so
    checked-in examples can use stable `local/...` paths. Ad hoc external
    configs still resolve relative to the config file itself.
    """

    resolved_config_dir = config_dir.resolve()
    if resolved_config_dir.is_relative_to(config_root_dir().resolve()):
        return project_root_dir().resolve()
    return resolved_config_dir
