# src/rl_fzerox/apps/run_manager/api/execution.py
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import ParamSpec, TypeVar

from starlette.concurrency import run_in_threadpool

_P = ParamSpec("_P")
_T = TypeVar("_T")


async def run_sync(function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    return await run_in_threadpool(partial(function, *args, **kwargs))
