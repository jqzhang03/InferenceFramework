"""Attribute CPU ranges, CUDA calls and GPU activity in DeepSeek bottleneck traces."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from benchmarks.cuda_operator_profile_report import union_ns


def analyze(path: Path) -> dict:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        names = dict(db.execute("SELECT id, value FROM StringIds"))
        kernels, copies = defaultdict(list), defaultdict(list)
        for pid, correlation, begin, end, name in db.execute("""
            SELECT globalPid, correlationId, start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL
        """):
            kernels[pid, correlation].append((begin, end, names[name]))
        for pid, correlation, begin, end, size, kind in db.execute("""
            SELECT globalPid, correlationId, start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY
        """):
            copies[pid, correlation].append((begin, end, size, kind))
        scopes = list(db.execute("""
            SELECT text, start, end, globalTid FROM NVTX_EVENTS
            WHERE end IS NOT NULL AND (text LIKE 'sample/%' OR text LIKE 'step/%' OR text LIKE 'component/%')
            ORDER BY start
        """))
        result = {}
        for name, start, stop, tid in scopes:
            if not name.startswith("sample/"):
                continue
            pid = tid & ~0xFFFFFF
            steps = []
            for label, begin, end, step_tid in scopes:
                if not label.startswith("step/") or step_tid != tid or begin < start or end > stop:
                    continue
                ranges = [(a, b, c.split("/")[-1]) for c, a, b, t in scopes
                          if t == tid and c.startswith("component/") and begin <= a and b <= end]
                expected = {"attention": 27, "dense_or_shared_ffn": 27, "routed_experts": 26, "lm_head": 1}
                if Counter(c for _, _, c in ranges) != expected:
                    raise ValueError(f"incomplete component ranges in {name}/{label}")
                if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
                    raise ValueError(f"overlapping component ranges in {name}/{label}")
                starts = [a for a, _, _ in ranges]
                cpu_ns = Counter()
                for a, b, c in ranges:
                    cpu_ns[c] += b - a
                cpu_ns["other"] = end - begin - sum(cpu_ns.values())
                stages = defaultdict(list)
                api_ns, api_counts = defaultdict(Counter), defaultdict(Counter)
                d2h, d2h_ns = defaultdict(Counter), Counter()
                calls = db.execute("""
                    SELECT start, end, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME
                    WHERE globalTid=? AND start>=? AND end<=?
                """, (tid, begin, end))
                seen = set()
                for a, b, correlation, api_id in calls:
                    index = bisect_right(starts, a) - 1
                    component = ranges[index][2] if index >= 0 and b <= ranges[index][1] else "other"
                    api = names[api_id]
                    api_counts[component][api] += 1
                    api_ns[component][api] += b - a
                    if correlation in seen:
                        continue
                    seen.add(correlation)
                    stages[component].extend(kernels.get((pid, correlation), ()))
                    for t0, t1, size, kind in copies.get((pid, correlation), ()):
                        if kind == 2:
                            d2h[component][size] += 1
                            d2h_ns[component] += t1 - t0
                components = {}
                for stage, duration in cpu_ns.items():
                    gpu_ns = Counter()
                    for a, b, kernel in stages[stage]:
                        gpu_ns[kernel] += b - a
                    components[stage] = dict(
                        cpu_window_ms=duration / 1e6,
                        gpu_kernel_sum_ms=sum(gpu_ns.values()) / 1e6,
                        gpu_kernel_ms={k: v / 1e6 for k, v in gpu_ns.items()},
                        kernel_count=len(stages[stage]), d2h_sizes=dict(d2h[stage]),
                        d2h_gpu_ms=d2h_ns[stage] / 1e6, api_counts=api_counts[stage],
                        api_ms={k: v / 1e6 for k, v in api_ns[stage].items()},
                    )
                activity = union_ns([(a, b) for stage in stages.values() for a, b, _ in stage])
                steps.append(dict(phase="prefill" if label == "step/0" else "decode",
                                  wall_ms=(end - begin) / 1e6, gpu_active_ms=activity / 1e6,
                                  components=components))
            if len(steps) != 4:
                raise ValueError(f"incomplete phase trace for {name}")
            result[name.removeprefix("sample/")] = steps
    if len(result) != 5:
        raise ValueError(f"expected five traced workload/variant combinations, got {list(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.sqlite), indent=2))


if __name__ == "__main__":
    main()
