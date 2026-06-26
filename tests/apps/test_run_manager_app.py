# tests/apps/test_run_manager_app.py
from __future__ import annotations

import socket

import pytest

from rl_fzerox.apps.run_manager.app import (
    DEFAULTS,
    _parse_args,
    _port_is_free,
    _resolve_web_port,
    _web_dev_command,
    _web_dev_environment,
)


def test_run_manager_web_root_is_top_level_frontend() -> None:
    assert DEFAULTS.web_root.parent.name == "web"
    assert DEFAULTS.web_root.name == "run-manager"
    assert (DEFAULTS.web_root / "package.json").is_file()


def test_run_manager_web_host_defaults_to_loopback() -> None:
    args = _parse_args([])

    assert args.web_host == DEFAULTS.web_host
    assert _web_dev_command(host=args.web_host, port=args.web_port) == [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "localhost",
        "--port",
        "5174",
        "--strictPort",
    ]


def test_run_manager_web_host_can_be_explicitly_exposed() -> None:
    args = _parse_args(["--web-host", "0.0.0.0", "--web-port", "6000"])

    assert _web_dev_command(host=args.web_host, port=args.web_port) == [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "0.0.0.0",
        "--port",
        "6000",
        "--strictPort",
    ]


def test_run_manager_web_environment_sets_api_proxy_and_filters_node_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NODE_OPTIONS", raising=False)

    environment = _web_dev_environment(api_port=8891)

    assert environment["VITE_API_PROXY_TARGET"] == "http://127.0.0.1:8891"
    assert environment["NODE_OPTIONS"] == "--no-deprecation"


def test_run_manager_web_environment_preserves_existing_node_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=4096")

    environment = _web_dev_environment(api_port=8891)

    assert environment["NODE_OPTIONS"] == "--max-old-space-size=4096 --no-deprecation"


def test_run_manager_web_port_resolver_skips_reachable_localhost_listener() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(socket.SOMAXCONN)
        port = int(listener.getsockname()[1])

        resolved = _resolve_web_port(port, host="localhost")

        assert _port_is_free(port, host="localhost") is False
        assert resolved.reassigned is True
        assert resolved.port > port
    finally:
        listener.close()
