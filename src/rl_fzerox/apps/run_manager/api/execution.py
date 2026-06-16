# src/rl_fzerox/apps/run_manager/api/execution.py
from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from threading import Thread
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


async def run_sync(function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    result: list[_T] = []
    errors: list[BaseException] = []
    call = partial(function, *args, **kwargs)

    def worker() -> None:
        try:
            result.append(call())
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=worker)
    thread.start()
    while thread.is_alive():
        await asyncio.sleep(0.001)
    thread.join()
    if errors:
        raise errors[0]
    return result[0]
