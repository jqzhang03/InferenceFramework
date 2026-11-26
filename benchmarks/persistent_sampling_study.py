"""Compare sampling metadata preparation with unprofiled runs and Nsight traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from bisect import bisect_right
from contextlib import closing, nullcontext
from pathlib import Path
from statistics import median
from time import perf_counter, process_time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm import LLM


CASES = (("decode_b1", 1, False), ("decode_b8", 8, False),
         ("decode_b32", 32, False), ("mixed_b8", 8, True))


def run_case(engine: LLM, batch: int, churn: bool, seed: int, *, input_length: int = 128) -> dict:
    import torch
    from nanovllm import SamplingParams

    rng = random.Random(seed)
    num_requests = batch * 6 if churn else batch
    prompts = [[rng.randrange(100, 10000) for _ in range(input_length)]
               for _ in range(num_requests)]
    params = [SamplingParams(temperature=(0.55, 0.8, 1.0)[i % 3],
                             max_tokens=16 + i % 17 if churn else 64,
                             ignore_eos=True)
              for i in range(num_requests)]
    engine.reset_benchmark_metrics()
    torch.manual_seed(seed)
    start_wall, start_cpu = perf_counter(), process_time()
    outputs = {}
    active = {}
    next_request = 0
    while next_request < num_requests or active:
        while next_request < num_requests and len(active) < batch:
            seq_id = engine.add_request(prompts[next_request], params[next_request])
            active[seq_id] = next_request
            next_request += 1
        finished, *_ = engine.step()
        for seq_id, tokens in finished:
            outputs[active.pop(seq_id)] = tokens
    torch.cuda.synchronize()
    cpu_s, wall_s = process_time() - start_cpu, perf_counter() - start_wall
    ordered = [outputs[i] for i in range(num_requests)]
    if [len(tokens) for tokens in ordered] != [p.max_tokens for p in params]:
        raise RuntimeError("Output lengths differ from the workload")
    stats = engine.collect_metrics()
    return dict(wall_s=wall_s, cpu_s=cpu_s, tokens=sum(map(len, ordered)),
                token_sha256=hashlib.sha256(json.dumps(ordered).encode()).hexdigest(),
                preemptions=stats["num_preemptions"],
                decode_graph_replays=engine.model_runner.decode_cudagraph_replay_count,
                mixed_graph_replays=engine.model_runner.mixed_cudagraph_replay_count)


def trace_metrics(path: Path) -> dict:
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        capture = db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='nsys_capture'").fetchone()
        cases = db.execute("""SELECT start,end,globalTid,text FROM NVTX_EVENTS
                              WHERE text LIKE 'sampling_case/%' AND start>=? AND end<=?""",
                           tuple(capture)).fetchall()
        result = {}
        for case in cases:
            scopes = db.execute("""SELECT start,end FROM NVTX_EVENTS
                WHERE text='ModelRunner.prepare_sample' AND start>=? AND end<=? AND globalTid=?
                ORDER BY start""", (case["start"], case["end"], case["globalTid"])).fetchall()
            starts = [scope["start"] for scope in scopes]
            durations = [(scope["end"] - scope["start"]) / 1000 for scope in scopes]
            correlations = set()
            for api in db.execute("""SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
                WHERE start>=? AND end<=? AND globalTid=?""",
                                  (case["start"], case["end"], case["globalTid"])):
                index = bisect_right(starts, api["start"]) - 1
                if index >= 0 and api["end"] <= scopes[index]["end"]:
                    correlations.add(api["correlationId"])
            copies = [row["bytes"] for row in db.execute("""SELECT bytes,correlationId
                FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=? AND copyKind=1""",
                      (case["start"], case["end"])) if row["correlationId"] in correlations]
            kernels = sum(row["correlationId"] in correlations for row in db.execute("""
                SELECT correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?""",
                (case["start"], case["end"])))
            result[case["text"].split("/")[1]] = dict(
                calls=len(scopes), cpu_total_us=sum(durations), cpu_median_us=median(durations),
                h2d_copies=len(copies), h2d_bytes=sum(copies), gpu_kernels=kernels)
        return result


def summarize(directory: Path) -> None:
    variants = ("before", "after")
    traces = {variant: trace_metrics(directory / f"{variant}.sqlite") for variant in variants}
    timings = {}
    captures = {}
    for variant in variants:
        rows = {}
        for path in sorted(directory.glob(f"{variant}*.timing.json")):
            run = path.name.removeprefix(variant).removesuffix(".timing.json")
            for row in json.loads(path.read_text(encoding="utf-8")):
                rows[run, row["case"], row["seed"]] = row
        timings[variant] = rows
        captures[variant] = {row["case"]: row for row in json.loads(
            (directory / f"{variant}.capture.json").read_text(encoding="utf-8"))}
    if timings["before"].keys() != timings["after"].keys():
        raise ValueError("Before and after workloads must match")
    results = {}
    for case, _, _ in CASES:
        before, after = zip(*[(row, timings["after"][key]) for key, row in timings["before"].items()
                              if key[1] == case])
        keys = ("tokens", "token_sha256", "decode_graph_replays", "mixed_graph_replays", "preemptions")
        if (any(a[k] != b[k] for a, b in zip(before, after) for k in keys)
                or any(captures["before"][case][k] != captures["after"][case][k] for k in keys)):
            raise ValueError(f"Output or scheduling mismatch for {case}")
        timing = {}
        for metric in ("wall_s", "cpu_s"):
            changes = [(b[metric] / a[metric] - 1) * 100 for a, b in zip(before, after)]
            timing[metric] = dict(before_median=median(r[metric] for r in before),
                                  after_median=median(r[metric] for r in after),
                                  paired_delta_pct=median(changes),
                                  paired_min_pct=min(changes), paired_max_pct=max(changes))
        results[case] = dict(pairs=len(before), timing=timing,
                             trace={name: rows[case] for name, rows in traces.items()})
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
    rows = []
    scope = nvtx_range if args.annotate else lambda _: nullcontext()
    try:
        for _, batch, churn in CASES:
            run_case(engine, batch, churn, 11)
        if args.annotate:
            runner = engine.model_runner
            for obj, method, label, phase_arg in (
                (engine, "step", "LLMEngine.step", None),
                (runner, "run", "ModelRunner.run", 1),
                (runner, "prepare_sample", "ModelRunner.prepare_sample", None),
                (runner.sampler, "forward", "Sampler.forward", None),
            ):
                _wrap_method(obj, method, label, phase_arg)
        with scope("nsys_capture"):
            for seed in args.seeds:
                for case, batch, churn in CASES:
                    with scope(f"sampling_case/{case}/{seed}"):
                        row = dict(case=case, seed=seed, **run_case(engine, batch, churn, seed))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
