# src/rl_fzerox/core/policy/__init__.py
from rl_fzerox.core.policy.extractors import (
    FZeroXCnnExtractor,
    FZeroXCnnWideExtractor,
)

EXTRACTOR_CLASSES = {
    "fzerox_cnn": FZeroXCnnExtractor,
    "fzerox_cnn_wide": FZeroXCnnWideExtractor,
}


def resolve_extractor_class(name: str):
    try:
        return EXTRACTOR_CLASSES[name]
    except KeyError as exc:
        known = ", ".join(sorted(EXTRACTOR_CLASSES))
        raise ValueError(f"Unknown policy extractor {name!r}. Expected one of: {known}") from exc

__all__ = [
    "EXTRACTOR_CLASSES",
    "FZeroXCnnExtractor",
    "FZeroXCnnWideExtractor",
    "resolve_extractor_class",
]
