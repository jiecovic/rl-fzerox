# src/rl_fzerox/apps/watch_cli/__init__.py
"""Watch CLI facade for run-manager session resolution."""

from rl_fzerox.apps.watch_cli.args import parse_args
from rl_fzerox.apps.watch_cli.resolve import resolve_watch_app_config

__all__ = ["parse_args", "resolve_watch_app_config"]
