"""Measure decode lookahead with matched unprofiled runs and Nsight traces."""

import argparse
import json
import sqlite3
from bisect import bisect_right
from contextlib import closing, nullcontext
from pathlib import Path
from statistics import median

from benchmarks.persistent_sampling_study import run_case


CASES = (
    ("decode_b1", 1, False, 128),
    ("decode_b8", 8, False, 128),
    ("decode_b32", 32, False, 128),
    ("boundary_b8", 8, False, 240),
    ("mixed_b8", 8, True, 128),
)


def _intersections(spans: list[tuple[int, int]], start: int, end: int) -> int:
    """Return activity time within the interval using disjoint, sorted spans."""
    index = max(0, bisect_right(spans, (start, end)) - 1)
    occupied = 0
    while index < len(spans) and spans[index][0] < end:
        a, b = spans[index]
        occupied += max(0, min(b, end) - max(a, start))
        index += 1
    return occupied


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _owning_range(spans: list[tuple[int, int]], start: int, end: int) -> int | None:
    index = bisect_right(spans, (start, float("inf"))) - 1
    return index if index >= 0 and end <= spans[index][1] else None


def _case_trace(db: sqlite3.Connection, bounds: tuple[int, int, int]) -> dict:
    begin, end, tid = bounds

    def scopes(label: str) -> list[tuple[int, int]]:
        return db.execute("""SELECT start,end FROM NVTX_EVENTS
            WHERE start>=? AND end<=? AND globalTid=? AND text=? ORDER BY start""",
                          (begin, end, tid, label)).fetchall()

    steps = scopes("LLM.step")
    runs = scopes("ModelRunner.run [decode]")
    models = scopes("ModelRunner.run_model [decode]")
    previews = scopes("ModelRunner.prepare_decode_lookahead")
    if not runs or len(runs) != len(models):
        raise ValueError("Expected one model scope per decode run")
    api = {correlation: (a, b) for a, b, correlation in db.execute("""
        SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE start>=? AND end<=? AND globalTid=?""", bounds)}
    kernels = db.execute("""SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start>=? AND end<=? ORDER BY start""", (begin, end)).fetchall()
    copies = db.execute("""SELECT start,end,correlationId,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE start>=? AND end<=?""", (begin, end)).fetchall()
    events = [(a, b) for a, b, _ in kernels]
    kernel_activity = _merge(events)
    events.extend((a, b) for a, b, _, _ in copies)
    if db.execute("""SELECT name FROM sqlite_master
                   WHERE name='CUPTI_ACTIVITY_KIND_MEMSET'""").fetchone():
        events.extend(db.execute("""SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET
            WHERE start>=? AND end<=?""", (begin, end)).fetchall())
    activity = _merge(events)

    first_kernel = {}
    for start, _, correlation in kernels:
        launch = api.get(correlation)
        if launch is not None:
            index = _owning_range(models, *launch)
            if index is not None:
                first_kernel.setdefault(index, start)

    readback_end = {}
    h2d_count = 0
    for _, stop, correlation, kind in copies:
        launch = api.get(correlation)
        if launch is not None:
            index = _owning_range(runs, *launch)
            if index is not None:
                if kind == 2:
                    readback_end[index] = max(stop, readback_end.get(index, 0))
                elif kind == 1:
                    h2d_count += 1
    if len(first_kernel) != len(runs) or len(readback_end) != len(runs):
        raise ValueError("A decode run is missing correlated kernels or token readback")

    step_indices = [_owning_range(steps, *run) for run in runs]
    if None in step_indices:
        raise ValueError("A decode run is missing its enclosing engine step")
    gaps = []
    for index in range(1, len(runs)):
        if step_indices[index - 1] == step_indices[index]:
            a, b = readback_end[index - 1], first_kernel[index]
            if b < a:
                raise ValueError("The next decode started before the current readback completed")
            gaps.append((b - a - _intersections(activity, a, b)) / 1000)

    overlaps = []
    for start, stop in previews:
        index = _owning_range(runs, start, stop)
        if index is None:
            raise ValueError("A preview is missing its enclosing decode run")
        overlaps.append(_intersections(kernel_activity, start, min(stop, readback_end[index])) / 1000)
    return dict(decode_forwards=len(runs), staged_inputs_used=len(runs) - len(scopes("ModelRunner.prepare_decode")),
                decode_h2d_count=h2d_count,
                consecutive_rounds=len(gaps), gpu_idle_median_us=median(gaps), gpu_idle_total_ms=sum(gaps) / 1000,
                preview_count=len(previews), preview_overlaps=sum(value > 0 for value in overlaps),
                preview_cpu_total_ms=sum(stop - start for start, stop in previews) / 1e6,
                preview_kernel_overlap_ms=sum(overlaps) / 1000)


def trace_metrics(path: Path) -> dict:
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        capture = db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='nsys_capture'").fetchone()
        cases = db.execute("""SELECT text,start,end,globalTid FROM NVTX_EVENTS WHERE start>=?
            AND end<=? AND text LIKE 'decode_case/%'""", capture).fetchall()
        return {name.split("/")[1]: _case_trace(db, (begin, end, tid)) for name, begin, end, tid in cases}


def summarize(directory: Path) -> None:
    variants = ("before", "after")
    traces = {name: trace_metrics(directory / f"{name}.sqlite") for name in variants}
    timings = {}
    for name in variants:
        timings[name] = {}
        for path in sorted(directory.glob(f"{name}.*.timing.json")):
            for row in json.loads(path.read_text(encoding="utf-8")):
                timings[name][path.name.split(".")[1], row["case"], row["seed"]] = row
        for row in json.loads((directory / f"{name}.capture.json").read_text(encoding="utf-8")):
            timings[name]["capture", row["case"], row["seed"]] = row
    if timings["before"].keys() != timings["after"].keys():
        raise ValueError("Before and after workloads must match")
    results = {}
    for case, *_ in CASES:
        pairs = []
        for key, before in timings["before"].items():
            if key[1] != case:
                continue
            after = timings["after"][key]
            fields = ("tokens", "token_sha256", "decode_graph_replays", "mixed_graph_replays", "preemptions")
            if any(before[field] != after[field] for field in fields):
                raise ValueError(f"Output or schedule changed for {key}")
            if key[0] != "capture":
                pairs.append((before, after))
        timing = {}
        for metric in ("wall_s", "cpu_s"):
            changes = [(b[metric] / a[metric] - 1) * 100 for a, b in pairs]
            timing[metric] = dict(paired_delta_pct=median(changes), paired_min_pct=min(changes),
                                  paired_max_pct=max(changes), faster_pairs=sum(delta < 0 for delta in changes))
        results[case] = dict(pairs=len(pairs), timing=timing,
                             trace={name: metrics[case] for name, metrics in traces.items()})
    (directory / "comparison.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(Path.home() / "huggingface/Qwen3-0.6B"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 99, 123, 202, 303])
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--summarize", action="store_true", help="Compare captures in the output directory")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output)
        return

    from benchmarks.nsys_profile import _wrap_method, nvtx_range
    from nanovllm import LLM

    engine = LLM(args.model, max_num_seqs=32, max_num_batched_tokens=4096,
                 max_model_len=1024, gpu_memory_utilization=0.55, execution_mode="mixed",
                 tpot_aware_scheduling=False, slo_aware_scheduling=False,
                 latency_aware_scheduling=False)
    runner = engine.model_runner
    rows = []
    scope = nvtx_range if args.annotate else lambda _: nullcontext()
    try:
        for _, batch, churn, length in CASES:
            run_case(engine, batch, churn, 11, input_length=length)
        if args.annotate:
            for obj, method, phase_arg in (
                (engine, "step", None),
                (runner, "run", 1),
                (runner, "run_model", 2),
                (runner, "prepare_decode", None),
                (runner, "prepare_decode_lookahead", None),
            ):
                _wrap_method(obj, method, f"{type(obj).__name__}.{method}", phase_arg)
        with scope("nsys_capture"):
            for seed in args.seeds:
                for case, batch, churn, length in CASES:
                    with scope(f"decode_case/{case}/{seed}"):
                        row = dict(case=case, seed=seed,
                                   **run_case(engine, batch, churn, seed, input_length=length))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
