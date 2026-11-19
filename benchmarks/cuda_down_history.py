"""Replay retained FP16 M8 down kernels for a comparable Nsight learning trace."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from benchmarks.cuda_operator_profile import Workload, environment, nvtx, run_logged, write_json


def trace(output: Path) -> None:
    import torch
    rows = []
    for variant, choice in (("baseline", dict(mode="gemm", config=10)),
                            ("current", dict(mode="decode", config=7))):
        case = dict(id=f"qwen_fp16_m8_down_{variant}", model="qwen", role="down",
                    dtype="fp16", shape=[8, 1024, 3072])
        for regime in ("resident", "rotating28"):
            work = Workload(case, regime, 42, choice)
            accuracy = work.validate()
            for _ in range(3):
                work.run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                work.run()
            graph.replay()
            torch.cuda.synchronize()
            with nvtx(f"op_capture/{case['id']}/{regime}/graph"):
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
            rows.append(dict(case=case, regime=regime, choice=choice, accuracy=accuracy,
                             operations=3*work.operations))
            del graph, work
    write_json(output / "trace.meta.json", dict(environment=environment(), rows=rows,
               provenance="Fresh replay of retained old/current kernels; not a historical capture or unprofiled benchmark"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "trace"), default="all")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "trace":
        trace(args.output)
        return
    prefix = args.output / "down_fp16_m8_old_current_graph"
    if prefix.with_suffix(".nsys-rep").exists():
        raise ValueError("Preserve existing reports; use a fresh directory")
    command = ["/usr/local/bin/nsys", "profile", "--trace=cuda,nvtx,osrt",
               "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node", "-o", str(prefix),
               sys.executable, "-m", "benchmarks.cuda_down_history", "--mode", "trace", "--output", str(args.output)]
    run_logged(command, prefix)
    run_logged(["/usr/local/bin/nsys", "export", "--type=sqlite", "-o", str(prefix.with_suffix(".sqlite")),
                str(prefix.with_suffix(".nsys-rep"))], args.output / "export")
    from benchmarks.cuda_operator_profile_report import analyze_nsys
    result = analyze_nsys(prefix.with_suffix(".sqlite"))
    if len(result) != 4:
        raise ValueError("expected old/current x resident/rotating28 windows")
    for row in result:
        stages = 2 if row["case"].endswith("_current") else 1
        operations = 96 if row["regime"] == "resident" else 84
        if row["kernel_calls"] != operations*stages:
            raise ValueError((row["case"], row["regime"], row["kernel_calls"]))
    write_json(args.output / "analysis.json", dict(rows=result))
    print(prefix.with_suffix(".nsys-rep"), flush=True)


if __name__ == "__main__":
    main()
