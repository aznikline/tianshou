from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.kernel_allocator_rl.train_grpo import main as grpo_main


def main(argv: list[str] | None = None) -> int:
    print("train_dqn.py is deprecated; forwarding to the GRPO training entrypoint.")
    return grpo_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
