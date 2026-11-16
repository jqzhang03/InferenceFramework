"""Render controlled GEMM attribution evidence; no GPU needed."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=Path("results/cuda_gemm_profile"))
    parser.add_argument("--output",type=Path,default=Path("benchmarks/_cuda_gemm_profile_20261115.md"))
    args=parser.parse_args()
    root=args.input_dir
    data=read(root/"ablation.json")
    rows=read(root/"row_grid.json")
    stages=read(root/"stage_ablation.json")
    layout=read(root/"layout_ablation.json")
    timeline=read(root/"timeline_summary.json")
    def find(source: dict,group: str,dtype: str,shape: tuple[int,int,int]) -> dict:
        return next(r for r in source["timings"] if r["group"]==group and r["dtype"]==dtype and tuple(r["shape_mnk"])==shape)
    def ratio(row: dict,numerator: str,denominator: str) -> float:
        return statistics.median(seed["us"][numerator]/seed["us"][denominator] for seed in row["seeds"])
    lines=["# CUDA GEMM profiling and controlled attribution — 2026-11-15", "",
        "## Scope and evidence boundary", "",
        "Historical scope: the attribution dataset below predates the same-day Nsight Compute recovery. Linux ncu 2026.3.1 subsequently collected counters successfully; see the separate [recovery record](_ncu_recovery_20261115.md). Counter-unavailable statements below describe the original 2025.1.1 attempt only, and the original timing data are retained.", "",
        "Question: why does the current handwritten FP16/BF16 GEMM get closer to cuBLAS at 4096³ than at 1024³? This report preserves the original 21 production configuration IDs and default address layout. Added CUDA entry points are diagnostic controls, not new engine dispatch or automatic optimization.", "",
        f"GPU: {data['gpu']}; PyTorch {data['torch']}; WSL2. Controlled timing: {data['method']}. Each variant passes FP64 sampled/full-reference and full-output cuBLAS checks before timing. Time columns are medians of per-seed medians; ratios/changes are medians of paired-seed ratios and need not equal ratios of the displayed time medians. No performance test runs concurrently with another GPU profiler. Clocks/caches are not externally locked, so profiling times are diagnostic and not a replacement for the original frozen-configuration benchmark.", "",
        "Nsight Systems 2026.5.1 successfully collected CUDA kernel timestamps, launch resources, and graph/eager gaps. Nsight Compute 2025.1.1 failed before collecting counters: `Failed to initialize the profiler: LibraryNotLoaded` (exit 9). Therefore no measured warp-stall, Tensor Core utilization, achieved occupancy, DRAM/L2 bandwidth, or bank-conflict-rate percentages are available. CUDA occupancy API values below are static residency limits, not achieved occupancy.", "",
        "Definitions: [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html). Conclusions below come from local measurements, not from that guide.", "",
        "## 1. Kernel time versus launch/graph gaps", "",
        "Medians over three NVTX ranges, 20 calls/range, using GPU timestamps from the SQLite export. NVTX ranges include synchronization so all twenty kernels fit inside their labels. Kernel duration excludes inter-kernel gaps. Profiled timings must not be mixed arithmetically with separate unprofiled runs.", "",
        "| dtype | size | backend/config | kernel µs/call | inter-kernel gap µs | kernel+gap span µs/call |", "|---|---:|---|---:|---:|---:|"]
    for dtype in ("fp16","bf16"):
        for size,config in ((1024,10),(4096,17)):
            for backend in ("cuda","cublas"):
                r=next(r for r in timeline["summary"] if r["range"]==f"gemm/{backend}/{dtype}/{size}/cfg{config}/graph")
                lines.append(f"| {dtype} | {size} | {backend}/cfg{config} | {r['kernel_us_per_call']:.3f} | {r['gap_mean_us']:.3f} | {r['span_us_per_call']:.3f} |")
    lines += ["", "The 1024³ graph gaps are approximately equal for custom and cuBLAS, while the kernel durations themselves differ by several microseconds. Host/Python launch starvation is not the main cause of this graph-replay gap. Eager tracing records larger gaps (see raw JSON), which do not explain the original graph benchmark. This does not exclude startup/teardown work *inside* each kernel/CTA.", "",
        "## 2. Registers and residency", "",
        "| config | size | registers/thread | local bytes/thread | dynamic shared bytes | resident blocks/SM limit | theoretical occupancy | grid blocks | waves model |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cfg in (10,17):
        for size in (1024,4096):
            r=next(r for r in data["resources"] if r["dtype"]=="torch.float16" and r.get("config")==cfg and r.get("size")==size and r["padding"]==0)
            i=r["info"]
            lines.append(f"| {cfg} | {size} | {i['registers_per_thread']} | {i['local_bytes_per_thread']} | {i['dynamic_shared_bytes']} | {i['active_blocks_per_sm_limit']} | {i['theoretical_occupancy']:.1%} | {i['grid_blocks']} | {i['waves']:.3f} |")
    lines += ["", "Both shapes have the same static residency ceiling; neither reports local-memory allocation. The claim '1024 has lower theoretical occupancy or register spilling' is unsupported. The grids differ substantially: 128 blocks versus 2048, on 36 SMs with room for four blocks/SM. A simple equal-duration wave model gives 88.9% versus 94.8% fill, but this is a model, not a measured active-SM fraction.", "",
        "Unused-shared-memory control: keep the exact kernel instructions/layout, reserve extra dynamic shared memory to reduce allowable residency. FP16 cfg10:", "",
        "| size | 4 blocks/SM, 24 KiB | 3 blocks/SM, 32 KiB | 2 blocks/SM, 40 KiB | 1 block/SM, 56 KiB |", "|---:|---:|---:|---:|---:|"]
    for size in (1024,4096):
        r=find(data,"residency_cfg10","torch.float16",(size,size,size))
        lines.append(f"| {size} | " + " | ".join(f"{r['median_us'][k]:.2f} µs" for k in ("base","pad8k","pad16k","pad32k")) + " |")
    lines += ["", "Reducing residency to one block is costly for both shapes. Four-to-three blocks has a much smaller effect. This validates residency sensitivity, but does not make a different static occupancy ceiling the cause of the original 1024-versus-4096 difference.", "",
        "## 3. Independent K and output-grid controls", "",
        "Use the exact same cfg10 kernel for every cell. Change K while holding M/N and the grid fixed; then change M/N while holding K fixed. Ratios are medians of three paired-seed cuBLAS/custom time ratios.", "",
        "| dtype | M=N | K | custom µs | cuBLAS µs | cuBLAS efficiency |", "|---|---:|---:|---:|---:|---:|"]
    effects=[]
    for dtype in ("torch.float16","torch.bfloat16"):
        cells={}
        for mn in (1024,4096):
            for k in (1024,4096):
                r=find(data,"k_sweep",dtype,(mn,mn,k))
                efficiency=ratio(r,"cublas","cfg10")
                cells[mn,k]=efficiency
                lines.append(f"| {dtype.removeprefix('torch.')} | {mn} | {k} | {r['median_us']['cfg10']:.2f} | {r['median_us']['cublas']:.2f} | {efficiency:.1%} |")
        k_effect=((cells[1024,4096]-cells[1024,1024])+(cells[4096,4096]-cells[4096,1024]))/2
        grid_effect=((cells[4096,1024]-cells[1024,1024])+(cells[4096,4096]-cells[1024,4096]))/2
        effects.append(f"- {dtype.removeprefix('torch.')}: total efficiency change {(cells[4096,4096]-cells[1024,1024])*100:.2f} percentage points; symmetric two-factor decomposition: K {k_effect*100:+.2f} pp, M/N {grid_effect*100:+.2f} pp.")
    lines += ["", "Both K length and output shape matter. For transparency, an algebraic symmetric two-factor decomposition of the above four cells is:", ""] + effects + ["",
        "These percentage points describe controlled *shape effects*, not hardware stall percentages. The M/N term also changes cache behavior and possibly the cuBLAS algorithm; it must not be relabeled entirely as 'wave loss'. The K term establishes an amortization/steady-state effect, but does not by itself identify a particular barrier or instruction.", "",
        "For fixed 1024×1024 output, fit `T = intercept + slope*K` over K=256/512/1024/2048/4096 (descriptive model):", "",
        "| dtype | backend | intercept µs | slope µs/K | R² |", "|---|---|---:|---:|---:|"]
    for dtype in ("torch.float16","torch.bfloat16"):
        sweep=[find(data,"k_sweep",dtype,(1024,1024,k)) for k in (256,512,1024,2048,4096)]
        xs=[r["shape_mnk"][2] for r in sweep]
        for backend in ("cfg10","cublas"):
            ys=[r["median_us"][backend] for r in sweep]
            slope,intercept=statistics.linear_regression(xs,ys)
            residual=sum((y-(intercept+slope*x))**2 for x,y in zip(xs,ys))
            total=sum((y-statistics.mean(ys))**2 for y in ys)
            lines.append(f"| {dtype.removeprefix('torch.')} | {backend} | {intercept:.3f} | {slope:.6f} | {1-residual/total:.6f} |")
    lines += ["", "The custom fit has a nontrivial fixed component plus a different per-K slope. The intercept bundles kernel/CTA startup, prologue/epilogue, scheduling and other non-K work; it is not a direct stopwatch measurement of pipeline warmup. A sole 'launch overhead' or sole 'pipeline depth' explanation would be too strong.", "",
        "## 4. Scheduling steps with no partial tiles", "",
        "N=K=1024; vary only M in multiples of 64, so every 64×128 output tile is full. FP16 cfg10:", "",
        "| M | grid blocks | custom µs | cuBLAS µs |", "|---:|---:|---:|---:|"]
    for m in (960,1024,1088,1152,1216,1280,1536,4096):
        r=find(rows,"row_grid","torch.float16",(m,1024,1024))
        lines.append(f"| {m} | {m//64*8} | {r['median_us']['cfg10']:.2f} | {r['median_us']['cublas']:.2f} |")
    lines += ["", "Crossing from 144 blocks (M=1152) to 152 (M=1216) crosses the nominal 36×4 residency capacity. Custom time jumps much more than the 5.6% extra work. This is direct evidence of grid/scheduling granularity, without an edge-mask confound. cuBLAS has its own steps and, for BF16, changes kernel families; the experiment does not isolate every cache/dispatch mechanism.", "",
        "## 5. Pipeline and shared-layout hypotheses tested", "",
        "Nsight kernel names/resources confirm different cuBLAS choices: FP16 uses a named 64×64_32×6 CUTLASS kernel in both sizes; BF16 uses named 128×128_32×5 at 1024 and 256×64_32×4 at 4096. The static trace contains no measured memory-stall rate.", "",
        "Fixed 64×64 tile and warp layout; vary stage count. A second control pads unused shared memory so stages 2–6 all reserve 48 KiB and have the same residency limit. FP16 at 1024³:", "",
        "| stages | natural µs | equalized shared-memory µs |", "|---:|---:|---:|"]
    natural=find(stages,"stage_natural","torch.float16",(1024,1024,1024))
    equal=find(stages,"stage_equalized_48k","torch.float16",(1024,1024,1024))
    for st in (2,3,4,5,6):
        lines.append(f"| {st} | {natural['median_us'][f'stages{st}']:.2f} | {equal['median_us'][f'stages{st}']:.2f} |")
    lines += ["", "Simply increasing stage count did not close the gap, including with equalized shared-memory capacity. Therefore the cuBLAS kernel's stage-count label is not sufficient to explain its advantage.", "",
        "A-address bank-layout intervention: the original BK=32 address model maps eight 16-byte ldmatrix row segments to bank groups `[0,5,2,7,0,5,2,7]`; the alternative uses `[0,4,1,5,2,6,3,7]`. This predicts different bank behavior; no bank-conflict hardware counter was collected. Keep mathematical operations, block/warp tiles and stages fixed, and change only the A copy/read address mapping:", "",
        "| dtype | size | cfg10 original µs | cfg10 alternative µs | alternative speed change | cfg17 original µs | cfg17 alternative µs |", "|---|---:|---:|---:|---:|---:|---:|"]
    for dtype in ("torch.float16","torch.bfloat16"):
        for size in (1024,4096):
            r=find(layout,"layout",dtype,(size,size,size))
            times=r["median_us"]
            change=ratio(r,"cfg10","cfg10_altA")-1
            lines.append(f"| {dtype.removeprefix('torch.')} | {size} | {times['cfg10']:.2f} | {times['cfg10_altA']:.2f} | {change:+.1%} | {times['cfg17']:.2f} | {times['cfg17_altA']:.2f} |")
    lines += ["", "The intervention does not establish this predicted bank conflict as the dominant practical gap. The source-level mapping changes can also alter address instructions/register allocation, so an address model must not be substituted for measured bank-stall rates. The alternative layout remains diagnostic and is not enabled by default.", "",
        "## Conclusion", "",
        "The verified explanation is a combination of shorter-K amortization/steady-state costs and output-grid/scheduling/shape-selection effects. The measured relative importance differs between FP16 and BF16 because the baseline selects different kernels. Host launch gaps, a lower static occupancy ceiling, and register spilling are not supported as the principal explanations. Merely deepening the pipeline or changing this A swizzle does not reproduce cuBLAS efficiency. Attribution below this level (specific wait/barrier/ldmatrix/epilogue cycle shares) remains unmeasured because hardware counters could not be initialized; no invented stall percentages are reported.", "",
        "## Reproduce", "", "```bash",
        "bash benchmarks/run_cuda_gemm.sh --profile --mode ncu",
        "bash benchmarks/run_cuda_gemm.sh --profile --mode suite",
        "bash benchmarks/run_cuda_gemm.sh --profile --mode rows",
        "bash benchmarks/run_cuda_gemm.sh --profile --mode stages",
        "bash benchmarks/run_cuda_gemm.sh --profile --mode layout",
        "bash benchmarks/run_cuda_gemm.sh --profile --mode nsys",
        "python -m benchmarks.cuda_gemm_profile --mode analyze",
        "python -m benchmarks.cuda_gemm_profile_report",
        "```", "",
        "Raw JSON, profiler logs, `.nsys-rep` and SQLite are local ignored files under `results/cuda_gemm_profile/`. `--mode tools` locates profilers; `--mode trace` profiles one case. The WSL helper supplies this machine's toolchain/Python defaults; override CUDA_HOME/GEMM_PYTHON on other systems. Diagnostic timings are separate dated observations and do not overwrite the original performance report.", ""]
    args.output.write_text("\n".join(lines),encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
