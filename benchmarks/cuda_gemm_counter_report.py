"""Render the 2026-11-16 counter study without requiring a GPU."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def metric(row: dict, name: str) -> float:
    return float(row["metrics"][name].replace(",",""))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=Path("results/cuda_gemm_counter_20261116"))
    parser.add_argument("--output",type=Path,default=Path("benchmarks/_cuda_gemm_counter_20261116.md"))
    args=parser.parse_args()
    timings=json.loads((args.input_dir/"timings.json").read_text())
    counters=json.loads((args.input_dir/"counters.json").read_text())
    def timing(dtype: str,shape: tuple[int,int,int]) -> dict:
        return next(r for r in timings["rows"] if r["dtype"]==f"torch.{dtype}" and tuple(r["shape_mnk"])==shape)
    def efficiency(row: dict,cfg: str="cfg10") -> float:
        return statistics.median(s["us"]["cublas"]/s["us"][cfg] for s in row["seeds"])
    def samples(dtype: str,size: int,backend: str) -> list[dict]:
        rows=[r for r in counters["rows"] if r["group"]=="primary" and r["dtype"]==dtype and r["shape_mnk"]==[size]*3 and r["backend"]==backend]
        if len(rows)!=counters["repeats"]:
            raise ValueError(f"Incomplete primary captures: {dtype}/{size}/{backend}")
        return rows
    def med(rows: list[dict],key: str) -> float:
        return statistics.median(metric(r,key) for r in rows)
    tensor="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
    tensor_active="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
    occupancy="sm__warps_active.avg.pct_of_peak_sustained_active"
    lines=["# CUDA GEMM: 1024 versus 4096 counter study — 2026-11-16", "",
           "## Measurement boundaries", "",
           f"GPU: {timings['gpu']}; PyTorch {timings['torch']}; WSL2; Nsight Compute 2026.3.1. {timings['method']}. Both backends use FP32 accumulation and preallocated output. Outputs pass FP64 sampled checks and full-output cuBLAS comparison before timing and before each counter capture. No CUDA kernel implementation or production dispatch was changed.", "",
           f"Primary counters: {counters['repeats']} independent captures/backend/dtype/size, alternating backend order; each captures one eager kernel after ten warmups. Fixed-kernel counter controls have one capture/point. Eight counter sections include detailed memory tables. Profiling uses kernel replay with clocks/cache control disabled. Counter medians and graph timings are separate datasets: do not use profiler durations as benchmark times, or subtract stall ratios from wall time. Historical 2026-11-15 measurements remain unchanged.", "",
           "## 1. Unprofiled performance", "",
           "Frozen configurations: cfg10 at 1024; cfg17 at 4096, matching the earlier comparison. Efficiency = median of paired cuBLAS/custom time ratios across seeds, hence can differ from a ratio of displayed medians.", "",
           "| dtype | size | custom µs | cuBLAS µs | custom/cuBLAS throughput |", "|---|---:|---:|---:|---:|"]
    for dtype,tdtype in (("fp16","float16"),("bf16","bfloat16")):
        for size,cfg in ((1024,"cfg10"),(4096,"cfg17")):
            r=timing(tdtype,(size,)*3)
            lines.append(f"| {dtype} | {size} | {r['median_us'][cfg]:.2f} | {r['median_us']['cublas']:.2f} | {efficiency(r,cfg):.1%} |")
    lines += ["", "## 2. Tensor activity and achieved occupancy", "",
              "Tensor elapsed uses the full kernel interval; Tensor active uses SM-active cycles. Pipe-active cycles are not useful FLOPs/peak or a speedup predictor: notably BF16 4096 has similar Tensor activity on both backends yet different graph timings. Instruction scheduling, work per active cycle and clocks still matter. The occupancy column uses active SM cycles and is not a direct fraction of device peak throughput.", "",
              "| dtype | size | backend | Tensor elapsed % [min,max] | Tensor active % | achieved occupancy % | registers/thread | grid blocks | launch waves/SM |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for dtype in ("fp16","bf16"):
        for size in (1024,4096):
            for backend in ("cuda","cublas"):
                rr=samples(dtype,size,backend)
                values=[metric(r,tensor) for r in rr]
                lines.append(f"| {dtype} | {size} | {backend} | {statistics.median(values):.2f} [{min(values):.2f},{max(values):.2f}] | {med(rr,tensor_active):.2f} | {med(rr,occupancy):.2f} | {med(rr,'launch__registers_per_thread'):.0f} | {med(rr,'launch__grid_size'):.0f} | {med(rr,'launch__waves_per_multiprocessor'):.2f} |")
    lines += ["", "Kernel symbols (first capture; cuBLAS config labels above identify the paired custom config, not an algorithm forced on cuBLAS):", ""]
    for dtype in ("fp16","bf16"):
        for size in (1024,4096):
            lines.append(f"- {dtype} / {size} / cuBLAS: `{samples(dtype,size,'cublas')[0]['kernel']}`")
    lines += ["", "## 3. Warp states", "",
              "Values below are Nsight `smsp__average_warps_issue_stalled_*_per_issue_active.ratio`, labelled **cycles per instruction** by `WarpStateStatistics.section` in the installed tool. They are not percentages of kernel time. In particular, math-pipe throttle can coexist with high Tensor utilization, and occupancy/stall values alone do not establish causality.", "",
              "| dtype | size | backend | barrier | long scoreboard | short scoreboard | wait | math-pipe throttle |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for dtype in ("fp16","bf16"):
        for size in (1024,4096):
            for backend in ("cuda","cublas"):
                rr=samples(dtype,size,backend)
                vals=[med(rr,f"smsp__average_warps_issue_stalled_{name}_per_issue_active.ratio") for name in ("barrier","long_scoreboard","short_scoreboard","wait","math_pipe_throttle")]
                lines.append(f"| {dtype} | {size} | {backend} | "+" | ".join(f"{v:.3f}" for v in vals)+" |")
    lines += ["", "## 4. Fixed cfg10: reduction length versus output grid", "",
              "All four points use the exact same custom kernel and full output tiles. Counter controls have one capture per point except the repeated 1024³ primary. Timing efficiency remains paired over three seeds.", "",
              "| dtype | M=N | K | custom µs | cuBLAS µs | efficiency | Tensor elapsed % | achieved occupancy % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dtype,tdtype in (("fp16","float16"),("bf16","bfloat16")):
        for mn in (1024,4096):
            for k in (1024,4096):
                t=timing(tdtype,(mn,mn,k))
                rr=[r for r in counters["rows"] if r["dtype"]==dtype and r["shape_mnk"]==[mn,mn,k] and r["backend"]=="cuda" and r["config"]==10]
                if not rr:
                    raise ValueError("Missing cfg10 control")
                lines.append(f"| {dtype} | {mn} | {k} | {t['median_us']['cfg10']:.2f} | {t['median_us']['cublas']:.2f} | {efficiency(t):.1%} | {med(rr,tensor):.2f} | {med(rr,occupancy):.2f} |")
    lines += ["", "For fixed 1024×1024 output, descriptive fit `T = intercept + slope*K` over K=256/512/1024/2048/4096. Intercept bundles startup, teardown, epilogue and other fixed work; it is not an independently measured prologue duration.", "",
              "| dtype | backend/config | intercept µs | slope µs/K | R² |", "|---|---|---:|---:|---:|"]
    for dtype in ("float16","bfloat16"):
        x=[256,512,1024,2048,4096]
        for name in ("cfg10","cfg17","cublas"):
            y=[timing(dtype,(1024,1024,k))["median_us"][name] for k in x]
            slope=sum((k-statistics.mean(x))*(v-statistics.mean(y)) for k,v in zip(x,y))/sum((k-statistics.mean(x))**2 for k in x)
            intercept=statistics.mean(y)-slope*statistics.mean(x)
            r2=1-sum((v-intercept-slope*k)**2 for k,v in zip(x,y))/sum((v-statistics.mean(y))**2 for v in y)
            lines.append(f"| {dtype} | {name} | {intercept:.3f} | {slope:.6f} | {r2:.5f} |")
    lines += ["", "## 5. Full-tile scheduling boundary", "",
              "N=K=1024, cfg10, M a multiple of 64; no partially filled output tiles. At 36 SMs and four resident CTAs/SM, M=1152 has 144 CTAs (one full capacity wave), while M=1216 has 152. `launch waves/SM` is a derived launch metric, not measured scheduling efficiency.", "",
              "| dtype | M | grid blocks | custom µs | cuBLAS µs | Tensor elapsed % (one capture) | occupancy % (one capture) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for dtype,tdtype in (("fp16","float16"),("bf16","bfloat16")):
        for m in (1152,1216):
            t=timing(tdtype,(m,1024,1024))
            r=next(r for r in counters["rows"] if r["dtype"]==dtype and r["shape_mnk"]==[m,1024,1024])
            lines.append(f"| {dtype} | {m} | {metric(r,'launch__grid_size'):.0f} | {t['median_us']['cfg10']:.2f} | {t['median_us']['cublas']:.2f} | {metric(r,tensor):.2f} | {metric(r,occupancy):.2f} |")
    lines += ["", "## Interpretation and limits", "",
              "The small matrix has both fewer output CTAs and fewer K iterations. Relative startup/teardown costs and underfilled/short-lived CTA execution reduce Tensor utilization more than in cuBLAS. Longer K and/or a larger output grid improve utilization with the exact cfg10 kernel. The full-tile wave-boundary timing discontinuity supports scheduling sensitivity independently of edge masking.", "",
              "An occupancy-only diagnosis is contradicted by FP16 cuBLAS: it is faster at 1024³ while using lower achieved occupancy. Barrier and math-pipe stall ratios do not rise uniquely at the small shape; their larger 4096 values must not be read as lost-time percentages. The custom kernel has two CTA barriers per BK=32 iteration, cp.async staging, ldmatrix loads, and scalar narrow/store epilogue work; these are plausible optimization targets, but this study does not individually time them or prove a single instruction is responsible for the entire gap.", "",
              "L2 hit rates, memory bytes, spill metrics, and the exact kernel symbols are retained in counters.json with their original units. Warm-cache results cannot establish cold-DRAM behavior. The requested sections do not expose a direct shared-bank-conflict count in this SM120 capture; address models and scoreboard stalls cannot substitute for one. Previous stage-depth and A-swizzle controls remain separate historical evidence, not newly measured improvements. No kernel optimization is claimed here.", "",
              "## Reproduction", "", "```bash",
              "bash benchmarks/run_cuda_gemm.sh --counter-study",
              "python -m benchmarks.cuda_gemm_counter_report",
              "```", "",
              f"Local raw artifacts: `{args.input_dir.as_posix()}/timings.json`, `counters.json`, and `captures/` (versioned ncu reports, raw CSV, full commands/status JSON, logs). Each capture verifies accuracy before starting the profiled range. The operator implementation and previous dated reports are preserved.", ""]
    args.output.write_text("\n".join(lines),encoding="utf-8")
    print(args.output)


if __name__=="__main__":
    main()
