from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.kernel_allocator_rl.trace import load_trace_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DQN-style policy for the RL allocator.")
    parser.add_argument("--trace", required=True, help="Path to a CSV trace file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    trace_path = Path(args.trace)
    trace = load_trace_csv(trace_path)

    if args.dry_run:
        print(
            "dry-run training using trace: "
            f"{trace_path} ({len(trace)} events, epochs={args.epochs})",
        )
        return 0

    raise SystemExit(
        "Full DQN training is not exercised in this environment. "
        "Use --dry-run to validate the pipeline entrypoint.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
