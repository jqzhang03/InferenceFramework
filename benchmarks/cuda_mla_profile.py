"""Profile paged DeepSeek-V2-Lite MLA with Nsight Systems and Nsight Compute."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
import random
import sqlite3
import subprocess
import sys

import torch

from benchmarks.cuda_gemm_profile import select_ncu
from benchmarks.cuda_mla_decode_study import graph_times, make_operator_inputs, time_graph
from benchmarks.cuda_operator_profile import run_logged
from benchmarks.cuda_operator_profile_report import union_ns
from benchmarks.deepseek_w4a16 import environment, write_json
from nanovllm.layers.attention_mla import mla_decode_attention
from nanovllm.layers.cuda_mla_decode import cuda_mla_decode_attention, load_cuda_mla_decode


MODULE = "benchmarks.cuda_mla_profile"
CASES = {"b8_l4096": (8, 4096), "b32_l128": (32, 128),
         "b32_l1024": (32, 1024), "b32_l4096": (32, 4096)}
BACKENDS = ("cuda", "triton")
VARIANTS = {"cuda": "original", "cuda_padding": "padded",
            "cuda_registers": "registers", "cuda_async": "async"}
REPLAYS = 8
NODES = 8  # The number of complete MLA calls captured per graph.
KERNEL_FILTER = "regex:^(mla_decode_cuda_kernel|mla_decode_combine|mla_decode_kernel)$"
SECTIONS = ("LaunchStats", "Occupancy", "SpeedOfLight", "SchedulerStats", "WarpStateStats",
            "MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Tables", "ComputeWorkloadAnalysis",
            "SourceCounters")
COUNTERS = (
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
)


def functions(case: str, seed: int) -> dict:
    torch.manual_seed(seed)
    args = make_operator_inputs(*CASES[case])
    ext = load_cuda_mla_decode(torch.device("cuda"), required=True)
    return {"cuda": partial(cuda_mla_decode_attention, *args, ext, variant="original"),
            "triton": partial(mla_decode_attention, *args)}


def independent_timing(root: Path) -> None:
    rows = []
    for case in CASES:
        for seed in (17, 42, 99):
            calls = functions(case, seed)
            want, got = calls["triton"](), calls["cuda"]()
            torch.testing.assert_close(got, want, atol=.002, rtol=.008)
            order = BACKENDS if seed == 42 else BACKENDS[::-1]
            times = {key: graph_times(calls[key]) for key in order}
            rows.append(dict(case=case, seed=seed, graph_us=times))
    write_json(root / "timing.json", dict(environment=environment(), timings=rows))


def trace() -> None:
    for case in CASES:
        for backend, call in functions(case, 42).items():
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(NODES):
                    call()
            graph.replay()
            torch.cuda.synchronize()
            with torch.cuda.nvtx.range(f"mla_capture/{case}/{backend}/graph"):
                for _ in range(REPLAYS):
                    graph.replay()
                torch.cuda.synchronize()
            with torch.cuda.nvtx.range(f"mla_capture/{case}/{backend}/eager"):
                for _ in range(NODES * REPLAYS):
                    call()
                torch.cuda.synchronize()


def probe(case: str, backend: str) -> None:
    if backend in VARIANTS:
        ext = load_cuda_mla_decode(torch.device("cuda"), required=True)
        if backend == "cuda_async":
            ext.prepare_async(torch.cuda.current_device())
        torch.manual_seed(42)
        call = partial(cuda_mla_decode_attention, *make_operator_inputs(*CASES[case]),
                       ext, variant=VARIANTS[backend])
    else:
        call = functions(case, 42)[backend]
    for _ in range(8):
        call()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    with torch.cuda.nvtx.range(f"mla_counter/{case}/{backend}"):
        call()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def analyze_timeline(path: Path) -> list[dict]:
    """Match kernels to launch APIs in each named window, including graph nodes."""
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        kernels = defaultdict(list)
        for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            kernels[row["correlationId"]].append(row)
        rows = []
        for r in db.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start"):
            label = r["text"] or strings.get(r["textId"], "")
            if not label.startswith("mla_capture/"):
                continue
            _, case, backend, execution = label.split("/")
            apis = list(db.execute(
                "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=? AND globalTid=?",
                (r["start"], r["end"], r["globalTid"])))
            pid = r["globalTid"] & ~0xffffff
            launch_names = Counter(strings[a["nameId"]] for a in apis)
            found = [k for a in apis for k in kernels.get(a["correlationId"], ())
                     if k["globalPid"] == pid]
            expected = REPLAYS * NODES * 3
            if len(found) != expected:
                raise ValueError(f"{label}: expected {expected} kernel nodes, got {len(found)}")
            by_kernel = defaultdict(list)
            for k in found:
                by_kernel[strings[k["shortName"]]].append((k["end"] - k["start"]) / 1000)
            intervals = [(k["start"], k["end"]) for k in found]
            span = max(b for _, b in intervals) - min(a for a, _ in intervals)
            operations = REPLAYS * NODES
            rows.append(dict(
                label=label, case=case, backend=backend, execution=execution,
                calls=operations, runtime_api_calls=launch_names,
                active_us_per_call=union_ns(intervals) / 1000 / operations,
                gpu_span_us_per_call=span / 1000 / operations,
                gap_us_per_call=(span - union_ns(intervals)) / 1000 / operations,
                kernels={name: dict(count=len(us), us_per_call=sum(us) / operations)
                         for name, us in by_kernel.items()},
            ))
    if len(rows) != len(CASES) * len(BACKENDS) * 2:
        raise ValueError("missing graph or eager windows")
    return rows


def collect_systems(root: Path, nsys: str) -> None:
    prefix = root / "mla_timeline"
    report, database = prefix.with_suffix(".nsys-rep"), prefix.with_suffix(".sqlite")
    if not report.exists():
        run_logged([nsys, "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
                    "--cpuctxsw=none", "--cuda-graph-trace=node", "-o", str(prefix),
                    sys.executable, "-m", MODULE, "--output", str(root), "--mode", "trace"], prefix)
    if not database.exists():
        run_logged([nsys, "export", "--type=sqlite", "-o", str(database), str(report)],
                   root / "export_systems")
    write_json(root / "timeline.json", analyze_timeline(database))


def capture_compute(root: Path, tool: str, case: str, backend: str, cache: str) -> None:
    folder = root / "ncu"
    folder.mkdir(exist_ok=True)
    prefix = folder / f"{case}_{backend}_{cache}"
    native = list(folder.glob(prefix.name + ".ncu-rep*"))
    if not native:
        command = [tool, "--target-processes", "all", "--profile-from-start", "off",
                   "--replay-mode", "kernel", "--clock-control", "none",
                   "--cache-control", cache, "--kernel-name-base", "function",
                   "--kernel-name", KERNEL_FILTER]
        for section in SECTIONS:
            command += ["--section", section]
        command += ["--metrics", ",".join(COUNTERS), "-o", str(prefix), sys.executable,
                    "-m", MODULE, "--output", str(root), "--mode", "probe", "--case",
                    case, "--backend", backend]
        print("NCU", prefix.name, flush=True)
        run_logged(command, prefix)
        native = list(folder.glob(prefix.name + ".ncu-rep*"))
    if len(native) != 1:
        raise ValueError(f"missing or ambiguous Nsight Compute report for {prefix}")
    csv = prefix.with_suffix(".csv")
    if not csv.exists():
        result = subprocess.run([tool, "--import", str(native[0]), "--page", "raw",
                                 "--csv"], text=True, capture_output=True, check=True)
        csv.write_text(result.stdout, encoding="utf-8")


def collect_compute(root: Path, tool: str) -> None:
    for case in CASES:
        for backend in BACKENDS:
            for cache in (("none", "all") if case == "b32_l4096" else ("none",)):
                capture_compute(root, tool, case, backend, cache)


def layout_control(root: Path, tool: str) -> None:
    ext = load_cuda_mla_decode(torch.device("cuda"), required=True)
    (root / "diagnostic").mkdir(exist_ok=True)
    case = "b32_l4096"
    rows = []
    for seed in (17, 42, 99):
        torch.manual_seed(seed)
        args = make_operator_inputs(*CASES[case])
        calls = {"cuda": partial(cuda_mla_decode_attention, *args, ext, variant="original"),
                 "cuda_padding": partial(cuda_mla_decode_attention, *args, ext, variant="padded")}
        torch.testing.assert_close(calls["cuda_padding"](), calls["cuda"](), atol=0, rtol=0)
        order = ("cuda", "cuda_padding") if seed == 42 else ("cuda_padding", "cuda")
        rows.append(dict(case=case, seed=seed, graph_us={key: graph_times(calls[key]) for key in order}))
    write_json(root / "diagnostic" / "padding_timing.json", rows)
    for backend in ("cuda", "cuda_padding"):
        capture_compute(root, tool, case, backend, "none")
        export_source(root, tool, case, backend, root / "diagnostic" / f"{backend}_source.csv")


def export_source(root: Path, tool: str, case: str, backend: str, output: Path) -> None:
    native, = (root / "ncu").glob(f"{case}_{backend}_none.ncu-rep*")
    command = [tool, "--import", str(native), "--page", "source",
               "--print-source", "cuda,sass", "--csv"]
    source = subprocess.run(command, text=True, capture_output=True, check=True)
    output.write_text(source.stdout, encoding="utf-8")


def capture_graph(call, nodes: int) -> torch.cuda.CUDAGraph:
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            call()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def pipeline_study(root: Path, tool: str) -> None:
    ext = load_cuda_mla_decode(torch.device("cuda"), required=True)
    ext.prepare_async(torch.cuda.current_device())
    rows = []
    for group, seeds in (("screen", (17, 42, 99)), ("heldout", (7, 23, 61))):
        for batch in (1, 8, 16, 32):
            for length in (128, 1024, 4096):
                for seed in seeds:
                    torch.manual_seed(seed)
                    args = make_operator_inputs(batch, length)
                    calls = {key: partial(cuda_mla_decode_attention, *args, ext, variant=value)
                             for key, value in VARIANTS.items()}
                    calls["triton"] = partial(mla_decode_attention, *args)
                    reference = calls["cuda"]()
                    for key, call in calls.items():
                        if key != "cuda":
                            tolerance = dict(atol=.002, rtol=.008) if key == "triton" else dict(atol=0, rtol=0)
                            torch.testing.assert_close(call(), reference, **tolerance)
                    nodes = 16
                    graphs = {key: capture_graph(call, nodes) for key, call in calls.items()}
                    samples = {key: [] for key in calls}
                    order = list(calls)
                    rng = random.Random(seed)
                    for _ in range(3):
                        rng.shuffle(order)
                        for key in order:
                            samples[key].append(time_graph(graphs[key], nodes))
                    rows.append(dict(group=group, batch=batch, context=length, seed=seed, graph_us=samples))
                    del graphs
                print(group, batch, length, flush=True)
        write_json(root / "pipeline_timing.json", dict(environment=environment(), rows=rows))
    for backend in (*VARIANTS, "triton"):
        case = "b32_l4096"
        capture_compute(root, tool, case, backend, "none")
        if backend != "triton":
            export_source(root, tool, case, backend, root / "ncu" / f"{backend}_source.csv")


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("all", "timing", "nsys", "ncu", "layout", "pipeline", "trace", "probe"),
                   default="all")
    p.add_argument("--case", choices=CASES)
    p.add_argument("--backend", choices=(*VARIANTS, "triton"))
    p.add_argument("--nsys", default="/usr/local/bin/nsys")
    p.add_argument("--ncu")
    args = p.parse_args()
    root = args.output.resolve()
    if args.mode == "trace":
        trace()
        return
    if args.mode == "probe":
        if not args.case or not args.backend:
            p.error("probe requires --case and --backend")
        probe(args.case, args.backend)
        return
    if args.mode in ("all", "pipeline") and root.exists():
        p.error("choose a fresh --output for a full study")
    root.mkdir(parents=True, exist_ok=True)
    if args.mode in ("all", "timing"):
        print("Independent graph timings", flush=True)
        independent_timing(root)
    if args.mode in ("all", "nsys"):
        print("Nsight Systems", flush=True)
        collect_systems(root, args.nsys)
    if args.mode in ("all", "ncu"):
        collect_compute(root, select_ncu(args.ncu))
    if args.mode == "layout":
        layout_control(root, select_ncu(args.ncu))
    if args.mode == "pipeline":
        pipeline_study(root, select_ncu(args.ncu))


if __name__ == "__main__":
    main()
