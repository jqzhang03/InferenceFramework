"""Render the standalone FP8 GEMM benchmark; no GPU required."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("results/cuda_fp8_gemm_grouped_20261116.json"))
    parser.add_argument("--profile-dir",type=Path,default=Path("results/cuda_fp8_grouped_profile_20261116"))
    parser.add_argument("--output",type=Path,default=Path("benchmarks/_cuda_fp8_gemm_20261116.md"))
    args=parser.parse_args()
    data=json.loads(args.input.read_text())
    lines=["# Native SM120 CUDA FP8 GEMM — 2026-11-16", "",
           "## Implementation and contract", "",
           "Independent handwritten CUDA operator: `C[M,N] = alpha * A[M,K] * W[N,K]^T`. A and W contain packed FP8, output is BF16, accumulation is FP32. W is pretransposed/packed row-major [N,K]; it is not the [K,N] layout of the earlier FP16/BF16 experiment. The custom kernel calls no cuBLAS/CUTLASS/Triton kernels. It is not wired into inference-engine dispatch.", "",
           "The native path targets **SM120a** and uses `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4`, cp.async 16-byte global-to-shared copies, zero-filled K/M/N tails, ldmatrix register fragments, XOR-swizzled shared memory, multistage ring buffers, and paired BF16 stores where alignment permits. Grouped configurations visit up to eight M tiles before advancing N, improving opportunities to reuse weight tiles in L2. Configs vary block/warp/K tiles and pipeline depth; the final tuple field is M-group size (1 = original raster). This is the consumer Blackwell warp-MMA path, not a tcgen05/TMEM implementation.", "",
           "Requires SM120, CUDA Toolkit 12.8+, contiguous 16-byte-aligned A/W and K divisible by 16. Partial output tiles, K not divisible by BK, empty M/N, and K=0 are supported by the custom path. Other K alignment is rejected explicitly. Finite host alpha is applied in the epilogue and may represent the product of per-tensor dequantization scales. No block scaling, fused quantization, autograd, beta accumulation, TP, or engine integration is provided. E4M3 is benchmarked; E5M2×E5M2 is separately tested in the custom kernel only, with no cuBLASLt performance claim.", "",
           "Instruction definitions: [NVIDIA PTX ISA 8.7](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html). Baseline API: [NVIDIA cuBLASLt documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul). SM120a is explicitly selected because the `.kind` qualifier needs the architecture-specific target in CUDA 12.8.", "",
           "## Method", "",
           f"GPU: {data['gpu']}; PyTorch {data['torch']}; WSL2. {data['layout']}. {data['method']}.", "",
           "Custom and cuBLASLt receive exactly the same quantized inputs, scalar scale product, output dtype and stream. Per-tensor E4M3 quantization uses amax/448. Compilation, quantization, transpose/packing, memory allocation, descriptor/algorithm setup, graph capture, and checks are excluded from timings. These are repeatedly reused buffer/operator measurements, not serving speedups or quantization-inclusive latency. Weight matrices larger than L2 cannot be assumed cache-resident merely because addresses are reused.", "",
           "Custom candidates and up to 32 cuBLASLt heuristic candidates are timed on separate seed-20261116 buffers. Every selected candidate passes an FP64 decoded-input check. The selected custom config and cuBLASLt algorithm attributes (ID/tile/stages/split-K/reduction/swizzle/custom/workspace) are frozen and checked across validation inputs. cuBLASLt uses CUBLAS_COMPUTE_32F, FAST_ACCUM=0 and up to 32 MiB workspace. This is a tuned heuristic subset, not an exhaustive search of all cuBLASLt algorithms. Plan creation is outside the timed path; plans own their descriptors and workspace.", "",
           "Validation uses seeds 7/17/42, 100 graph nodes and seven randomized paired rounds/seed. Times/TFLOPS are medians of per-seed medians. Efficiency is median paired cuBLASLt/custom time ratio, so it need not equal the ratio of displayed medians. GPU clocks are not locked; seed ranges show run variation. Previous dated FP16/BF16 results are not a same-run comparison against FP8.", "",
           "## Performance", "",
           "| M×N×K | cfg | block / K / stages / group-M | CUDA µs | cuBLASLt µs | CUDA TFLOPS | cuBLASLt TFLOPS | relative throughput [seed range] |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in data["results"]:
        m,n,k=r["shape_mnk"];cu=r["median_us"]["cuda"];lt=r["median_us"]["cublas"]
        bm,bn,bk,wm,wn,st,gm=data["configs"][r["config"]]
        ratios=[s["us"]["cublas"]/s["us"]["cuda"] for s in r["seeds"]]
        lines.append(f"| {m}×{n}×{k} | {r['config']} | {bm}×{bn} / {bk} / {st} / {gm} | {cu:.2f} | {lt:.2f} | {2*m*n*k/cu/1e6:.2f} | {2*m*n*k/lt/1e6:.2f} | {statistics.median(ratios):.1%} [{min(ratios):.1%}, {max(ratios):.1%}] |")
    arith=[s["accuracy"]["cuda"]["arithmetic_relative_l2"] for r in data["results"] for s in r["seeds"]]
    quant=[s["accuracy"]["cuda"]["including_quantization_relative_l2"] for r in data["results"] for s in r["seeds"]]
    parity=[s["accuracy"]["cuda_vs_cublas_relative_l2"] for r in data["results"] for s in r["seeds"]]
    lines += ["", "## Accuracy", "",
              "All benchmark outputs are finite and pass the decoded-FP8 reference check (relative L2 limit 0.004) and full-output cuBLASLt comparison. Reference outputs use FP64 multiplication over every K term: full small outputs, or a deterministic sample of up to 64×64 positions for large outputs.", "",
              f"- Maximum arithmetic/output-rounding relative L2 against decoded-FP8 FP64 reference: **{max(arith):.8f}**.",
              f"- Maximum total relative L2 against the common original FP32 input reference (including E4M3 quantization): **{max(quant):.8f}**.",
              f"- Maximum full-output relative L2 versus cuBLASLt: **{max(parity):.8f}**.", "",
              "Quantization error is reported separately and is not an arithmetic-error tolerance. These random-matrix tests do not establish end-model quality. GPU tests additionally cover all configurations, odd M/N, partial K tiles, grouped-raster tails, zero/constant/identity patterns, E5M2 arithmetic, output scaling, graph capture, non-default streams, empty shapes, K=0, invalid API inputs and cuBLASLt plan mismatch.", ""]
    ablation=Path("results/cuda_fp8_group_ablation_20261116.json")
    if ablation.exists():
        controls=json.loads(ablation.read_text())
        lines += ["## Grouped-scheduling control", "",
                  "Same input buffers, tile/warp/K/stage configuration, and math; only CTA raster changes. Three seeds × seven randomized paired rounds × 100 graph nodes. All group-only pairs have bitwise-identical outputs. Values below are median paired speedup (original/grouped time); >1 is faster. This is a timing intervention, not a measured breakdown of DRAM/L2 traffic.", "",
                  "| M×N×K | cfg0→18 (64×64) | cfg3→19 (64×128) | cfg11→22 (128×128, BK128) |", "|---|---:|---:|---:|"]
        for r in controls["rows"]:
            shape="×".join(map(str,r["shape_mnk"]))
            values=[r["paired_speedup"][key] for key in ("0_to_18","3_to_19","11_to_22")]
            lines.append(f"| {shape} | "+" | ".join(f"{v:.3f}×" for v in values)+" |")
        lines += ["", "Grouping helps some memory-sensitive shapes/configurations and can be neutral or slightly slower elsewhere. The best 128×11008×4096 configuration also changes tile size (cfg25, 128×64); its gain cannot be attributed wholly to grouping. Keep the original and grouped candidates in tuning rather than enabling grouping universally.", ""]
    profile=args.profile_dir/"counters.json"
    if profile.exists():
        counters=json.loads(profile.read_text())
        lines += ["## Native instruction and counter validation", "",
                  "The final binary's SASS contains `QMMA.16832.F32.E4M3.E4M3`, `LDSM`, and `LDGSTS`; FP8 is multiplied natively rather than dequantized into a BF16 GEMM. One profiled launch/backend/size after ten warmups validates counters; these are diagnostic single samples, separate from the repeated benchmark.", "",
                  "| size | backend | Tensor pipe active / elapsed % | achieved occupancy % |", "|---:|---|---:|---:|"]
        for r in counters:
            mm=r["metrics"]
            lines.append(f"| {r['shape_mnk'][0]} | {r['backend']} | {float(mm['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed']):.2f} | {float(mm['sm__warps_active.avg.pct_of_peak_sustained_active']):.2f} |")
        lines += ["", "Pipe activity is not useful-FLOP utilization or a direct throughput ratio. Counter timings are not substituted for graph benchmark times. Full logs, raw units, native instruction counts and disassembly are retained locally.", ""]
    lines += ["## Reproduction and artifacts", "", "```bash",
              "bash benchmarks/run_cuda_gemm.sh --fp8 --check-only",
              "bash benchmarks/run_cuda_gemm.sh --fp8 --output results/cuda_fp8_gemm_grouped_20261116.json",
              "bash benchmarks/run_cuda_gemm.sh --fp8 --tuning-report results/cuda_fp8_gemm_grouped_20261116.json --output results/cuda_fp8_gemm_frozen.json",
              "bash benchmarks/run_cuda_gemm.sh --fp8-profile --input results/cuda_fp8_gemm_grouped_20261116.json --output-dir results/cuda_fp8_grouped_profile_20261116",
              "python -m pytest tests/test_cuda_fp8_gemm.py -q",
              "python -m benchmarks.cuda_fp8_gemm_report", "```", "",
              f"Formal raw results: `{args.input.as_posix()}`. Counter/disassembly evidence: `{args.profile_dir.as_posix()}/`. Earlier `cuda_fp8_gemm_final_20261116.json` / `cuda_fp8_profile_20261116/` are separate initial 18-config observations; do not combine their times arithmetically with the later grouped-config run. The helper supplies this machine's Python/toolkit defaults; native compilation uses GCC 13 and sm_120a. The FP16/BF16 implementation and its original 21 config IDs remain unchanged.", ""]
    args.output.write_text("\n".join(lines),encoding="utf-8")
    print(args.output)


if __name__=="__main__":
    main()
