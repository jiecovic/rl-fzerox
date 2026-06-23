# src/rl_fzerox/apps/run_manager/api/execution.py
from __future__ import annotations

import asyncio
import contextvars
import os
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")
_MAX_WORKERS = min(8, max(2, (os.cpu_count() or 1)))
_MAX_QUEUED_CALLS = _MAX_WORKERS * 4
_EXECUTOR = ThreadPoolExecutor(
    max_workers=_MAX_WORKERS,
    thread_name_prefix="run-manager-api",
)
_SEMAPHORES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


def _run_sync_semaphore(loop: asyncio.AbstractEventLoop) -> asyncio.Semaphore:
    semaphore = _SEMAPHORES.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(_MAX_QUEUED_CALLS)
        _SEMAPHORES[loop] = semaphore
    return semaphore


async def run_sync(function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    context = contextvars.copy_context()
    call = partial(context.run, function, *args, **kwargs)
    loop = asyncio.get_running_loop()
    async with _run_sync_semaphore(loop):
        return await loop.run_in_executor(_EXECUTOR, call)
