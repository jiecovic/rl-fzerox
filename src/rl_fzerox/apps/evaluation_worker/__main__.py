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
    cancel_path = store.evaluation_cancel_request_path(args.evaluation_id)
    try:
        from rl_fzerox.core.evaluation.managed import run_managed_evaluation

        result = run_managed_evaluation(
            evaluation,
            device=args.device,
            worker_count=args.worker_count,
            should_cancel=cancel_path.is_file,
        )
    except Exception as exc:
        current = store.get_evaluation(args.evaluation_id)
        if current is not None and current.status in {"cancelling", "cancelled"}:
            store.mark_evaluation_cancelled(args.evaluation_id)
            return
        store.mark_evaluation_failed(args.evaluation_id, error_message=str(exc))
        raise
    if result.status == "cancelled":
        store.mark_evaluation_cancelled(args.evaluation_id)
        return
    current = store.get_evaluation(args.evaluation_id)
    if current is not None and current.status == "cancelled":
        return
    store.mark_evaluation_completed(args.evaluation_id)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--worker-count", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
