# src/rl_fzerox/apps/evaluation_worker/__main__.py
"""Command-line worker for manager-owned course evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_fzerox.core.manager import ManagerStore


def main() -> None:
    args = _parse_args()
    store = ManagerStore(args.db_path)
    evaluation = store.get_evaluation(args.evaluation_id)
    if evaluation is None:
        raise RuntimeError(f"evaluation not found: {args.evaluation_id}")
    try:
        from rl_fzerox.core.evaluation.managed import run_managed_evaluation

        run_managed_evaluation(evaluation)
    except Exception as exc:
        store.mark_evaluation_failed(args.evaluation_id, error_message=str(exc))
        raise
    store.mark_evaluation_completed(args.evaluation_id)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--evaluation-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
