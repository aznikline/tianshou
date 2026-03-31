from __future__ import annotations

import struct
from pathlib import Path


POLICY_MAGIC = b"RLP1"
POLICY_HEADER = struct.Struct("<4sIII")


def export_policy_table(path: str | Path, table: list[int], version: int) -> None:
    payload = bytes(table)
    checksum = sum(payload) & 0xFFFFFFFF
    header = POLICY_HEADER.pack(POLICY_MAGIC, version, len(table), checksum)
    Path(path).write_bytes(header + payload)
