# src/rl_fzerox/core/config/paths.py
from __future__ import annotations

from pathlib import Path

_REPO_ROOT_RELATIVE_PREFIXES = frozenset({"checkpoints", "conf", "local"})


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
    External config files still resolve relative to their own directory unless
    they explicitly use a repository-root prefix such as `local/...`.
    """

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    return (_config_path_resolution_base_dir(config_dir, path=path) / path).resolve()


def config_path_resolution_base_dir(config_dir: Path) -> Path:
    """Return the base directory used to resolve relative config paths.

    Repository-owned configs deliberately resolve from the project root so
    checked-in examples can use stable `local/...` paths. Ad hoc external
    configs still resolve relative to the config file itself.
    """

    return _config_path_resolution_base_dir(config_dir, path=None)


def resolve_config_data_paths(
    config_data: dict[str, object],
    *,
    config_dir: Path,
    path_fields: dict[str, tuple[str, ...]],
) -> None:
    """Resolve configured path fields in-place using project path rules."""

    for section_name, field_names in path_fields.items():
        section = config_data.get(section_name)
        if not isinstance(section, dict):
            continue

        for field_name in field_names:
            raw_value = section.get(field_name)
            if not isinstance(raw_value, str):
                continue

            section[field_name] = str(resolve_config_path_value(raw_value, config_dir=config_dir))


def _config_path_resolution_base_dir(config_dir: Path, *, path: Path | None) -> Path:
    resolved_config_dir = config_dir.resolve()
    if path is not None and _is_repo_root_relative_path(path):
        return project_root_dir().resolve()
    if resolved_config_dir.is_relative_to(config_root_dir().resolve()):
        return project_root_dir().resolve()
    return resolved_config_dir


def _is_repo_root_relative_path(path: Path) -> bool:
    parts = path.parts
    return bool(parts) and parts[0] in _REPO_ROOT_RELATIVE_PREFIXES
