from pathlib import Path

from examples.kernel_allocator_rl.policy_export import export_policy_table


def test_export_policy_table_writes_magic_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "policy.bin"

    export_policy_table(path, [0, 1, 2, 3], version=1)
    raw = path.read_bytes()

    assert raw[:4] == b"RLP1"
    assert len(raw) > 16
