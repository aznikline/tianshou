from pathlib import Path

from examples.kernel_allocator_rl.config import BucketConfig
from examples.kernel_allocator_rl.trace import TraceEvent, load_trace_csv


def test_bucket_config_provides_expected_request_edges() -> None:
    cfg = BucketConfig.default()
    assert cfg.request_size_edges[:4] == (16, 32, 64, 128)
    assert cfg.request_size_edges[-1] == 4096


def test_load_trace_csv_parses_alloc_and_free_rows(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.csv"
    trace_path.write_text(
        "ts,cpu,op,ptr_id,size,flags\n"
        "1,0,alloc,a0,64,0\n"
        "2,0,free,a0,0,0\n",
        encoding="utf-8",
    )

    rows = load_trace_csv(trace_path)

    assert rows == [
        TraceEvent(ts=1, cpu=0, op="alloc", ptr_id="a0", size=64, flags=0),
        TraceEvent(ts=2, cpu=0, op="free", ptr_id="a0", size=0, flags=0),
    ]
