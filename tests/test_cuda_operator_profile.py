"""CPU checks for profiling attribution, multi-kernel exports and raw units."""
import csv
from pathlib import Path
import sqlite3

import pytest

from benchmarks.cuda_operator_profile_report import analyze_nsys, read_ncu


def test_ncu_keeps_both_split_k_stages_and_converts_units(tmp_path: Path) -> None:
    path = tmp_path / "multi.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows([
            ["Kernel Name", "gpu__time_duration.sum", "dram__bytes_op_read.sum.per_second"],
            ["", "ns", "Mbyte/s"],
            ["void decode_mma_kernel<true>()", "12,500", "380,000"],
            ["void finish_kernel<true>()", "2,000", "1,000"],
        ])
    rows = read_ncu(path)
    assert [r["kind"] for r in rows] == ["decode_mma_kernel", "finish_kernel"]
    assert [r["us"] for r in rows] == [12.5, 2.0]
    assert rows[0]["dram_gbs"] == 380
    assert rows[0]["raw_units"]["gpu__time_duration.sum"] == "ns"


def make_trace(path: Path, kernel_name: str) -> None:
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER, correlationId INTEGER, globalPid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER, correlationId INTEGER, globalTid INTEGER);
            INSERT INTO NVTX_EVENTS VALUES (1000, 9000, 'op_capture/example/resident/eager', NULL, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100, 900, 2, 8, 0);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2000, 4000, 1, 9, 0);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (6000, 7000, 1, 9, 0);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1500, 1800, 3, 9, 1);
        """)
        db.executemany("INSERT INTO StringIds VALUES (?,?)", [(1, kernel_name), (2, "reference_library_kernel"), (3, "cudaLaunchKernel")])


def test_nsys_excludes_initialization_and_preserves_idle(tmp_path: Path) -> None:
    path = tmp_path / "trace.sqlite"
    make_trace(path, "void gemm_kernel<true>()")
    row, = analyze_nsys(path)
    assert row["kernel_calls"] == 2
    assert row["operations"] == 96
    assert row["gpu_span_ms"] == .005
    assert row["between_kernel_idle_ms"] == .002
    assert row["active_pct_in_gpu_span"] == 60
    assert row["runtime_api_calls"] == {"cudaLaunchKernel": 1}


def test_nsys_rejects_library_kernel_inside_measurement(tmp_path: Path) -> None:
    path = tmp_path / "trace.sqlite"
    make_trace(path, "cublas_reference_kernel")
    with pytest.raises(ValueError, match="non-custom kernel"):
        analyze_nsys(path)


def test_nsys_correlation_keeps_gpu_event_outside_cpu_range(tmp_path: Path) -> None:
    path = tmp_path / "trace.sqlite"
    make_trace(path, "void gemm_kernel<true>()")
    with sqlite3.connect(path) as db:
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=8000,end=10000 WHERE start=6000")
    row, = analyze_nsys(path)
    assert row["kernel_calls"] == 2
    assert row["kernels"][-1]["end"] == 10000
