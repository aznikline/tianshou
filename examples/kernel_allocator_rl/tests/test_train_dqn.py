from pathlib import Path
import subprocess
import sys


def test_training_script_supports_dry_run(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.csv"
    trace_path.write_text(
        "ts,cpu,op,ptr_id,size,flags\n"
        "1,0,alloc,a0,64,0\n"
        "2,0,free,a0,0,0\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "examples/kernel_allocator_rl/train_dqn.py",
            "--trace",
            str(trace_path),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "dry-run" in result.stdout.lower()
    assert "grpo" in result.stdout.lower()
