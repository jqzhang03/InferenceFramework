"""Render the maintained CUDA GEMM report from frozen-config validation runs."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/_cuda_gemm_report_20261115.md"))
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    first = reports[0]
    keys = [(r["dtype"], tuple(r["shape_mnk"])) for r in first["results"]]
    for report in reports:
        if not report["checks"]["passed"]:
            raise ValueError("correctness did not pass")
        for field in ("gpu", "sm", "torch", "cuda", "configs", "iterations", "repeats", "timing", "cublas"):
            if report[field] != first[field]:
                raise ValueError(f"incompatible reports: {field}")
        if [(r["dtype"],tuple(r["shape_mnk"])) for r in report["results"]] != keys:
            raise ValueError("workload mismatch")
        if [r["config"] for r in report["results"]] != [r["config"] for r in first["results"]]:
            raise ValueError("validation configurations were not frozen")
    lines = [
        "# CUDA FP16/BF16 GEMM — 2026-11-15",
        "",
        "This is a standalone handwritten CUDA operator experiment. It is not wired into the inference engine and is not INT4 GEMM. The 2026-10-05 FP16 report remains a separate historical result.",
        "",
        "## Implementation and supported inputs",
        "",
        "`cuda_gemm.cu` computes contiguous row-major `C[M,N] = A[M,K] @ B[K,N]` with FP16 or BF16 inputs/output and FP32 accumulation. The custom path calls no cuBLAS kernels. It uses `mma.sync.m16n8k16`, `ldmatrix`, XOR-swizzled shared memory, and a two-/three-/four-stage `cp.async` ring pipeline. A block barrier protects shared-memory reuse. The 21 candidate configurations vary block/warp tiles, K tile, and pipeline stages; tuning checks every candidate for correctness before timing it.",
        "",
        "The fast path requires N and K divisible by 8, SM80+, and 16-byte-aligned inputs. M and the final block tiles may be partial: loads zero-fill and stores are masked. Other N/K sizes and K=0 use an explicit scalar CUDA fallback, whose speed is not represented by the performance table. Empty M/N are safe no-ops. The API validates shape/dtype/device/contiguity/alignment and rejects output aliasing. It uses the current device and PyTorch CUDA stream and supports CUDA Graph capture. No autograd, transpose, alpha/beta, TP, or engine integration is provided. GPU validation below is limited to the named RTX 5060 Ti.",
        "",
        "Instruction semantics: [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) (`mma`, `ldmatrix`, `cp.async`).",
        "",
        "## Measurement",
        "",
        f"- GPU: {first['gpu']}, SM {first['sm']}, {first['sm_count']} SMs; {first['platform']}.",
        f"- PyTorch {first['torch']}; PyTorch CUDA {first['cuda']}; toolchain: `{first['nvcc'].splitlines()[-1]}`.",
        f"- Source HEAD `{first['git_head']}`; dirty working tree: {first['git_worktree_dirty']}. New operator files are part of that dirty tree, not the clean HEAD.",
        f"- Validation seeds: {', '.join(str(r['seed']) for r in reports)}. Each backend: {first['iterations']} calls per CUDA Graph replay × {first['repeats']} paired rounds per seed; backend order randomized each round.",
        "- Buffers preallocated; same inputs, output dtype, FP32 accumulation, stream, and timing method. Warm-cache CUDA Events measure graph replay; compilation, allocation, host dispatch, graph capture, accuracy checks and tuning are excluded. These are isolated operator times, not application latency.",
        "- Baseline: direct `cublasGemmEx`, `CUBLAS_COMPUTE_32F`, default Tensor Core algorithm, reduced-precision reduction disabled. It is not an exhaustive cuBLASLt algorithm search. Efficiency below is cuBLAS time / custom CUDA time; >100% means faster than this particular baseline.",
        "- Configurations selected in a separate screen, then frozen for all validation seeds. Table times and TFLOPS are medians of per-seed medians; the ratio range is across seeds. A tuning winner is not a guarantee for other hardware/workloads.",
        "- Both dtypes use the same original FP32 random matrices at each seed/shape. FP64 reference uses all K terms, with full outputs for small tests and a reproducible 64×64 output sample for large benchmarks. The full custom output is also checked against cuBLAS. Rounded-input error isolates GEMM arithmetic/output rounding; original-input error also includes FP16/BF16 input quantization.",
        "",
        "## Speed",
        "",
        "| dtype | M×N×K | config | CUDA µs | cuBLAS µs | CUDA TFLOPS | cuBLAS TFLOPS | cuBLAS efficiency, median [seed range] |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i,(dtype,shape) in enumerate(keys):
        rows = [report["results"][i] for report in reports]
        median = lambda fn: statistics.median(fn(row) for row in rows)
        ratios = [row["cuda_over_cublas"] for row in rows]
        lines.append(f"| {dtype.removeprefix('torch.')} | {'×'.join(map(str,shape))} | {rows[0]['config']} | "
                     f"{median(lambda r:r['time_us']['cuda']):.2f} | {median(lambda r:r['time_us']['cublas']):.2f} | "
                     f"{median(lambda r:r['tflops']['cuda']):.2f} | {median(lambda r:r['tflops']['cublas']):.2f} | "
                     f"{statistics.median(ratios):.1%} [{min(ratios):.1%}, {max(ratios):.1%}] |")
    lines += ["", "## Accuracy", "",
              "All listed entries passed. Values below are the worst relative L2 error across all validation seeds/shapes of each dtype, not an elementwise relative error near zero.", "",
              "| dtype | CUDA vs FP64 rounded-input reference | cuBLAS vs same reference | CUDA vs FP64 original-input reference | cuBLAS vs same original-input reference |", "|---|---:|---:|---:|---:|"]
    for dtype in ("torch.float16", "torch.bfloat16"):
        rows = [r for report in reports for r in report["results"] if r["dtype"]==dtype]
        vals = [max(r["accuracy"][backend]["relative_l2"] for r in rows) for backend in ("cuda","cublas")]
        vals += [max(r["accuracy"]["including_input_rounding"][f"{backend}_relative_l2"] for r in rows) for backend in ("cuda","cublas")]
        lines.append(f"| {dtype.removeprefix('torch.')} | " + " | ".join(f"{v:.6g}" for v in vals) + " |")
    lines += ["", "Correctness checks cover both dtypes × 21 configurations × 10 shape/pattern cases (420 checks), including M=1, partial M/N/K tiles, unaligned N/K fallback, K=0, zeros, identity, cancellation and mixed magnitudes; plus non-default-stream graph replay, empty outputs and invalid-input rejection. These tests establish the tested input range, not exhaustive numerical behavior for overflow/NaN/Inf.",
              "", "Compute Sanitizer memcheck was attempted on this machine, but did not initialize: `Failed to initialize WDDM debugger interface` / `Device not supported`. The process exited 1. No memory-sanitizer pass is claimed; ordinary GPU correctness checks passed. Resolving the Windows debugger-interface prerequisite is outside this experiment.",
              "", "## Reproduce", "", "```bash",
              "# CUDA toolkit, PyTorch CUDA, Ninja and a supported C++ compiler required.",
              "# Portable entry point (set CUDA_HOME/PATH and TORCH_CUDA_ARCH_LIST as needed):",
              "python -m benchmarks.cuda_gemm_bench --check-only",
              "python -m benchmarks.cuda_gemm_bench --output results/cuda_gemm_20261115_tune.json --iterations 50 --repeats 5",
              "# This machine's WSL helper; repeat with --seed 17 and --seed 7:",
              "bash benchmarks/run_cuda_gemm.sh --validate",
              "python -m benchmarks.cuda_gemm_report results/cuda_gemm_20261115_seed42.json results/cuda_gemm_20261115_seed17.json results/cuda_gemm_20261115_seed7.json",
              "```", "",
              "The helper defaults to `/usr/local/cuda` and the local `nano-vllm` conda Python; override `CUDA_HOME` and `GEMM_PYTHON` elsewhere. A normal benchmark call tunes automatically; `--tuning-report` loads and freezes an existing report's configurations. Optional `--shapes 'M,N,K;...'`, `--configs`, `--iterations`, and `--repeats` control the sweep.",
              "", "Raw local artifacts (ignored by Git):"]
    lines += [f"- `{path.as_posix()}`" for path in args.inputs]
    lines += ["", "The remaining gap is shape dependent. Small/skinny matrices still suffer from tile utilization, occupancy, and launch/synchronization costs. Further work could test split-K and register prefetching under the same accuracy and frozen-validation protocol. No unmeasured improvement is claimed.", ""]
    args.output.write_text("\n".join(lines),encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
