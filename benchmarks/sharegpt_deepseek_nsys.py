"""Attribute the ShareGPT DeepSeek capture to engine steps and copy/API costs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from benchmarks.cuda_operator_profile_report import union_ns


def analyze(sqlite_path: Path, events_path: Path | None) -> dict:
    events = json.loads(events_path.read_text()) if events_path else {}
    step_kinds = []
    for kind, count in events.get("step_kinds", {}).items():
        step_kinds.extend([kind] * int(count))
    with closing(sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        names = dict(db.execute("SELECT id, value FROM StringIds"))
        kernels = [
            (begin, end, names[name])
            for _, begin, end, name in db.execute(
                "SELECT globalPid, start, end, shortName "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL")
        ]
        has_graph_trace = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='CUPTI_ACTIVITY_KIND_GRAPH_TRACE'").fetchone()
        if has_graph_trace:
            kernels.extend(
                (begin, end, "CUDA Graph (replay)")
                for begin, end in db.execute(
                    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
            )
        copies = [
            (begin, end, size, kind)
            for begin, end, size, kind in db.execute(
                "SELECT start, end, bytes, copyKind "
                "FROM CUPTI_ACTIVITY_KIND_MEMCPY")
        ]
        apis = defaultdict(list)
        for tid, begin, end, name_id in db.execute(
                "SELECT globalTid, start, end, nameId "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            apis[tid].append((begin, end, names[name_id]))

        scopes = list(db.execute(
            "SELECT text, start, end, globalTid FROM NVTX_EVENTS "
            "WHERE end IS NOT NULL AND text = 'nsys_capture' ORDER BY start"))
        if len(scopes) != 1:
            raise ValueError(f"expected one nsys_capture range, got {len(scopes)}")
        _, capture_start, capture_stop, capture_tid = scopes[0]
        step_scopes = list(db.execute(
            "SELECT text, start, end, globalTid FROM NVTX_EVENTS "
            "WHERE end IS NOT NULL AND text LIKE 'LLMEngine.step[%' "
            "AND globalTid=? AND start>=? AND end<=? ORDER BY start",
            (capture_tid, capture_start, capture_stop)))
        if not step_scopes:
            raise ValueError("no engine step ranges in capture")
        if step_kinds and len(step_kinds) != len(step_scopes):
            raise ValueError(
                f"step kinds {len(step_kinds)} != step ranges {len(step_scopes)}")

        per_step = []
        for index, (_, begin, end, tid) in enumerate(step_scopes):
            step_kernels = [
                (a, b, name) for a, b, name in kernels if begin <= a and b <= end]
            step_copies = [
                (a, b, size, kind) for a, b, size, kind in copies
                if begin <= a and b <= end]
            step_apis = [
                (a, b, name) for a, b, name in apis.get(tid, ())
                if begin <= a and b <= end]
            kernel_time = Counter()
            kernel_count = Counter()
            for a, b, name in step_kernels:
                kernel_time[name] += b - a
                kernel_count[name] += 1
            copy_stats = Counter()
            copy_bytes = Counter()
            for _, _, size, kind in step_copies:
                copy_stats[kind] += 1
                copy_bytes[kind] += size
            api_stats = Counter()
            api_time = Counter()
            for a, b, name in step_apis:
                api_stats[name] += 1
                api_time[name] += b - a
            phase = step_kinds[index] if index < len(step_kinds) else "unknown"
            per_step.append({
                "index": index,
                "phase": phase,
                "wall_ms": (end - begin) / 1e6,
                "gpu_active_ms": union_ns([(a, b) for a, b, _ in step_kernels]) / 1e6,
                "kernel_sum_ms": sum(kernel_time.values()) / 1e6,
                "kernel_count": sum(kernel_count.values()),
                "top_kernels": [
                    {"name": name, "ms": duration / 1e6, "count": kernel_count[name]}
                    for name, duration in kernel_time.most_common(12)
                ],
                "copies": {
                    str(kind): {"count": copy_stats[kind], "bytes": copy_bytes[kind]}
                    for kind in copy_stats
                },
                "apis": [
                    {"name": name, "count": api_stats[name], "ms": api_time[name] / 1e6}
                    for name, _ in api_stats.most_common(12)
                ],
            })

    phases = {}
    for step in per_step:
        phase = step["phase"]
        row = phases.setdefault(phase, {
            "steps": 0, "wall_ms": 0.0, "gpu_active_ms": 0.0,
            "kernel_sum_ms": 0.0, "kernel_count": 0,
            "d2h_count": 0, "d2h_bytes": 0, "h2d_count": 0, "h2d_bytes": 0,
        })
        row["steps"] += 1
        row["wall_ms"] += step["wall_ms"]
        row["gpu_active_ms"] += step["gpu_active_ms"]
        row["kernel_sum_ms"] += step["kernel_sum_ms"]
        row["kernel_count"] += step["kernel_count"]
        row["d2h_count"] += step["copies"].get("2", {}).get("count", 0)
        row["d2h_bytes"] += step["copies"].get("2", {}).get("bytes", 0)
        row["h2d_count"] += step["copies"].get("1", {}).get("count", 0)
        row["h2d_bytes"] += step["copies"].get("1", {}).get("bytes", 0)
    return {
        "workload": events,
        "capture": {
            "wall_ms": (capture_stop - capture_start) / 1e6,
            "gpu_active_ms": union_ns([
                (a, b) for a, b, _ in kernels
                if capture_start <= a and b <= capture_stop]) / 1e6,
            "steps": len(per_step),
        },
        "phases": phases,
        "steps": per_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = analyze(args.sqlite, args.events)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
