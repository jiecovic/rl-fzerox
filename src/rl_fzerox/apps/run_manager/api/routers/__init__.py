# src/rl_fzerox/apps/run_manager/api/routers/__init__.py
from __future__ import annotations

from rl_fzerox.apps.run_manager.api.routers.drafts import create_drafts_router
from rl_fzerox.apps.run_manager.api.routers.lineages import create_lineages_router
from rl_fzerox.apps.run_manager.api.routers.metrics import create_metrics_router
from rl_fzerox.apps.run_manager.api.routers.runs import create_runs_router
from rl_fzerox.apps.run_manager.api.routers.save_games import create_save_games_router
from rl_fzerox.apps.run_manager.api.routers.streams import create_streams_router
from rl_fzerox.apps.run_manager.api.routers.system import create_system_router
from rl_fzerox.apps.run_manager.api.routers.transfer import create_transfer_router

__all__ = [
    "create_drafts_router",
    "create_lineages_router",
    "create_metrics_router",
    "create_runs_router",
    "create_save_games_router",
    "create_streams_router",
    "create_system_router",
    "create_transfer_router",
]
