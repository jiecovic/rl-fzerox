# src/rl_fzerox/apps/run_manager/api/__init__.py
"""Run-manager API facade."""

from rl_fzerox.apps.run_manager.api.routes import create_manager_api_app

__all__ = ["create_manager_api_app"]
