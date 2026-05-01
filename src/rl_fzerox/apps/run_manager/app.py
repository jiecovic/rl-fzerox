# src/rl_fzerox/apps/run_manager/app.py
from __future__ import annotations

import argparse
import errno
import os
import signal
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from rl_fzerox.apps.run_manager.api import create_manager_api_app
from rl_fzerox.core.manager import ManagerStore


@dataclass(frozen=True, slots=True)
class RunManagerLauncherDefaults:
    api_port: int = 8765
    web_port: int = 5174
    web_root: Path = Path(__file__).with_name("web")


@dataclass(frozen=True, slots=True)
class BoundApiSocket:
    port: int
    socket: socket.socket


DEFAULTS = RunManagerLauncherDefaults()


def main(argv: list[str] | None = None) -> None:
    """Launch the React run manager and its local FastAPI/SQLite API."""

    args = _parse_args(argv)
    store = ManagerStore()
    store.initialize()

    api_binding = _bind_api_socket(args.api_port)
    api_server = uvicorn.Server(
        uvicorn.Config(
            create_manager_api_app(store),
            access_log=False,
            log_level="warning",
        )
    )
    api_thread = threading.Thread(
        target=_run_api_server,
        args=(api_server, api_binding.socket),
        daemon=True,
    )
    api_thread.start()

    if args.api_only:
        print(f"Run manager API listening on http://127.0.0.1:{api_binding.port}")
        try:
            _join_api_server(api_thread)
        except KeyboardInterrupt:
            _stop_api_server(api_server, api_thread)
        return

    _ensure_frontend_dependencies(DEFAULTS.web_root)
    command = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.web_port),
    ]
    environment = os.environ.copy()
    environment["VITE_API_PROXY_TARGET"] = f"http://127.0.0.1:{api_binding.port}"

    print(f"Run manager UI:  http://localhost:{args.web_port}")
    print(f"Run manager API: http://127.0.0.1:{api_binding.port}")
    process = subprocess.Popen(command, cwd=DEFAULTS.web_root, env=environment)
    try:
        process.wait()
    except KeyboardInterrupt:
        _terminate_process(process)
    finally:
        _stop_api_server(api_server, api_thread)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local F-Zero X run manager")
    parser.add_argument("--api-port", type=int, default=DEFAULTS.api_port)
    parser.add_argument("--web-port", type=int, default=DEFAULTS.web_port)
    parser.add_argument("--api-only", action="store_true")
    return parser.parse_args(argv)


def _bind_api_socket(requested_port: int) -> BoundApiSocket:
    if requested_port == 0:
        bound_socket = _open_api_socket(requested_port)
        return BoundApiSocket(port=int(bound_socket.getsockname()[1]), socket=bound_socket)

    try:
        return BoundApiSocket(port=requested_port, socket=_open_api_socket(requested_port))
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise

    for candidate_port in range(requested_port + 1, requested_port + 21):
        try:
            bound_socket = _open_api_socket(candidate_port)
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                continue
            raise
        print(
            f"Run manager API port {requested_port} is busy; using {candidate_port} instead.",
            file=sys.stderr,
        )
        return BoundApiSocket(port=candidate_port, socket=bound_socket)

    raise SystemExit(
        f"Run manager API port {requested_port} is busy and no nearby free port was found."
    )


def _open_api_socket(port: int) -> socket.socket:
    bound_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bound_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        bound_socket.bind(("127.0.0.1", port))
        bound_socket.listen(socket.SOMAXCONN)
    except OSError:
        bound_socket.close()
        raise
    return bound_socket


def _run_api_server(api_server: uvicorn.Server, bound_socket: socket.socket) -> None:
    api_server.run(sockets=[bound_socket])


def _join_api_server(api_thread: threading.Thread) -> None:
    while api_thread.is_alive():
        api_thread.join(timeout=0.5)


def _stop_api_server(api_server: uvicorn.Server, api_thread: threading.Thread) -> None:
    api_server.should_exit = True
    api_thread.join(timeout=5.0)
    if api_thread.is_alive():
        api_server.force_exit = True
        api_thread.join(timeout=2.0)


def _ensure_frontend_dependencies(web_root: Path) -> None:
    if (web_root / "node_modules").is_dir():
        return
    print(
        "Run-manager frontend dependencies are missing. Run: just run-manager-install",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5.0)
