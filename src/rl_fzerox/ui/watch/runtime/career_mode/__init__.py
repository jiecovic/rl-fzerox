# src/rl_fzerox/ui/watch/runtime/career_mode/__init__.py
"""Career Mode worker entrypoint facade for Watch runtime process startup."""

from rl_fzerox.ui.watch.runtime.career_mode.worker import run_career_mode_worker

__all__ = ["run_career_mode_worker"]
