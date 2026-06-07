# tests/apps/test_run_manager_app.py
from __future__ import annotations

from rl_fzerox.apps.run_manager.app import DEFAULTS, _parse_args, _web_dev_command


def test_run_manager_web_host_defaults_to_loopback() -> None:
    args = _parse_args([])

    assert args.web_host == DEFAULTS.web_host
    assert _web_dev_command(host=args.web_host, port=args.web_port) == [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "127.0.0.1",
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
