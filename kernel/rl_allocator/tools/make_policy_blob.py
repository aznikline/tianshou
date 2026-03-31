from __future__ import annotations

import argparse
import struct
from pathlib import Path


HEADER = struct.Struct("<4sIII")
MAGIC = b"RLP1"


def checksum(table: bytes) -> int:
    return sum(table) & 0xFFFFFFFF


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or rewrite an RL allocator policy blob.")
    parser.add_argument("input", help="Input binary file.")
    parser.add_argument("--output", help="Optional rewritten output path.")
    parser.add_argument("--version", type=int, help="Override policy version in output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw = Path(args.input).read_bytes()
    if len(raw) < HEADER.size:
        raise SystemExit("input is too small to contain an RL policy header")

    magic, version, entry_count, stored_checksum = HEADER.unpack_from(raw, 0)
    payload = raw[HEADER.size :]
    actual_checksum = checksum(payload)
    if magic != MAGIC:
        raise SystemExit(f"unexpected magic {magic!r}")
    if len(payload) != entry_count:
        raise SystemExit(
            f"payload length mismatch: header={entry_count}, payload={len(payload)}",
        )

    print(
        f"magic={magic.decode()} version={version} entries={entry_count} "
        f"checksum={stored_checksum} valid={stored_checksum == actual_checksum}",
    )

    if args.output:
        out_version = version if args.version is None else args.version
        header = HEADER.pack(MAGIC, out_version, entry_count, actual_checksum)
        Path(args.output).write_bytes(header + payload)
        print(f"wrote policy blob to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
