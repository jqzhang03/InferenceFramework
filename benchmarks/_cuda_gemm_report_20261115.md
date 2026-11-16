# CUDA FP16/BF16 GEMM — 2026-11-15

This is a standalone handwritten CUDA operator experiment. It is not wired into the inference engine and is not INT4 GEMM. The 2026-10-05 FP16 report remains a separate historical result.

## Implementation and supported inputs

`cuda_gemm.cu` computes contiguous row-major `C[M,N] = A[M,K] @ B[K,N]` with FP16 or BF16 inputs/output and FP32 accumulation. The custom path calls no cuBLAS kernels. It uses `mma.sync.m16n8k16`, `ldmatrix`, XOR-swizzled shared memory, and a two-/three-/four-stage `cp.async` ring pipeline. A block barrier protects shared-memory reuse. The 21 candidate configurations vary block/warp tiles, K tile, and pipeline stages; tuning checks every candidate for correctness before timing it.

The fast path requires N and K divisible by 8, SM80+, and 16-byte-aligned inputs. M and the final block tiles may be partial: loads zero-fill and stores are masked. Other N/K sizes and K=0 use an explicit scalar CUDA fallback, whose speed is not represented by the performance table. Empty M/N are safe no-ops. The API validates shape/dtype/device/contiguity/alignment and rejects output aliasing. It uses the current device and PyTorch CUDA stream and supports CUDA Graph capture. No autograd, transpose, alpha/beta, TP, or engine integration is provided. GPU validation below is limited to the named RTX 5060 Ti.

Instruction semantics: [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) (`mma`, `ldmatrix`, `cp.async`).

## Measurement

- GPU: NVIDIA GeForce RTX 5060 Ti, SM 12.0, 36 SMs; Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43.
- PyTorch 2.8.0+cu128; PyTorch CUDA 12.8; toolchain: `Build cuda_12.8.r12.8/compiler.35583870_0`.
- Source HEAD `abb3ae557976d90e0279690ff1b9ee6d0ab1944b`; dirty working tree: True. New operator files are part of that dirty tree, not the clean HEAD.
- Validation seeds: 42, 17, 7. Each backend: 100 calls per CUDA Graph replay × 7 paired rounds per seed; backend order randomized each round.
- Buffers preallocated; same inputs, output dtype, FP32 accumulation, stream, and timing method. Warm-cache CUDA Events measure graph replay; compilation, allocation, host dispatch, graph capture, accuracy checks and tuning are excluded. These are isolated operator times, not application latency.
- Baseline: direct `cublasGemmEx`, `CUBLAS_COMPUTE_32F`, default Tensor Core algorithm, reduced-precision reduction disabled. It is not an exhaustive cuBLASLt algorithm search. Efficiency below is cuBLAS time / custom CUDA time; >100% means faster than this particular baseline.
- Configurations selected in a separate screen, then frozen for all validation seeds. Table times and TFLOPS are medians of per-seed medians; the ratio range is across seeds. A tuning winner is not a guarantee for other hardware/workloads.
- Both dtypes use the same original FP32 random matrices at each seed/shape. FP64 reference uses all K terms, with full outputs for small tests and a reproducible 64×64 output sample for large benchmarks. The full custom output is also checked against cuBLAS. Rounded-input error isolates GEMM arithmetic/output rounding; original-input error also includes FP16/BF16 input quantization.

## Speed

| dtype | M×N×K | config | CUDA µs | cuBLAS µs | CUDA TFLOPS | cuBLAS TFLOPS | cuBLAS efficiency, median [seed range] |
|---|---|---:|---:|---:|---:|---:|---:|
| float16 | 64×4096×4096 | 4 | 59.87 | 50.66 | 35.87 | 42.39 | 84.6% [84.6%, 84.7%] |
| float16 | 256×4096×4096 | 16 | 221.89 | 199.67 | 38.71 | 43.02 | 90.0% [90.0%, 91.4%] |
| float16 | 1024×1024×1024 | 10 | 61.96 | 51.07 | 34.66 | 42.05 | 82.4% [82.3%, 82.4%] |
| float16 | 4096×4096×4096 | 17 | 3108.40 | 2940.21 | 44.22 | 46.74 | 94.6% [94.3%, 94.7%] |
| float16 | 128×11008×4096 | 12 | 300.73 | 264.07 | 38.38 | 43.71 | 87.2% [85.8%, 90.1%] |
| float16 | 512×4096×11008 | 16 | 1147.90 | 1024.53 | 40.22 | 45.07 | 89.3% [88.9%, 89.4%] |
| float16 | 8192×4096×4096 | 10 | 6243.69 | 5887.89 | 44.02 | 46.69 | 94.3% [94.3%, 94.6%] |
| bfloat16 | 64×4096×4096 | 4 | 60.60 | 51.54 | 35.44 | 41.66 | 85.3% [84.9%, 85.3%] |
| bfloat16 | 256×4096×4096 | 16 | 216.52 | 198.67 | 39.67 | 43.24 | 91.4% [90.8%, 91.8%] |
| bfloat16 | 1024×1024×1024 | 10 | 61.84 | 52.82 | 34.73 | 40.66 | 85.4% [85.3%, 85.6%] |
| bfloat16 | 4096×4096×4096 | 17 | 3037.49 | 2882.44 | 45.25 | 47.68 | 94.7% [94.7%, 94.9%] |
| bfloat16 | 128×11008×4096 | 12 | 299.81 | 357.16 | 38.50 | 32.32 | 119.1% [119.1%, 120.4%] |
| bfloat16 | 512×4096×11008 | 5 | 1128.20 | 1052.95 | 40.92 | 43.85 | 93.2% [93.1%, 93.4%] |
| bfloat16 | 8192×4096×4096 | 10 | 6103.46 | 5671.16 | 45.04 | 48.47 | 92.9% [92.5%, 93.5%] |

## Accuracy

All listed entries passed. Values below are the worst relative L2 error across all validation seeds/shapes of each dtype, not an elementwise relative error near zero.

| dtype | CUDA vs FP64 rounded-input reference | cuBLAS vs same reference | CUDA vs FP64 original-input reference | cuBLAS vs same original-input reference |
|---|---:|---:|---:|---:|
| float16 | 0.000214069 | 0.000214069 | 0.000369296 | 0.000369296 |
| bfloat16 | 0.00173511 | 0.00173511 | 0.00295088 | 0.00295088 |

Correctness checks cover both dtypes × 21 configurations × 10 shape/pattern cases (420 checks), including M=1, partial M/N/K tiles, unaligned N/K fallback, K=0, zeros, identity, cancellation and mixed magnitudes; plus non-default-stream graph replay, empty outputs and invalid-input rejection. These tests establish the tested input range, not exhaustive numerical behavior for overflow/NaN/Inf.

Compute Sanitizer memcheck was attempted on this machine, but did not initialize: `Failed to initialize WDDM debugger interface` / `Device not supported`. The process exited 1. No memory-sanitizer pass is claimed; ordinary GPU correctness checks passed. Resolving the Windows debugger-interface prerequisite is outside this experiment.

## Reproduce

```bash
# CUDA toolkit, PyTorch CUDA, Ninja and a supported C++ compiler required.
# Portable entry point (set CUDA_HOME/PATH and TORCH_CUDA_ARCH_LIST as needed):
python -m benchmarks.cuda_gemm_bench --check-only
python -m benchmarks.cuda_gemm_bench --output results/cuda_gemm_20261115_tune.json --iterations 50 --repeats 5
# This machine's WSL helper; repeat with --seed 17 and --seed 7:
bash benchmarks/run_cuda_gemm.sh --validate
python -m benchmarks.cuda_gemm_report results/cuda_gemm_20261115_seed42.json results/cuda_gemm_20261115_seed17.json results/cuda_gemm_20261115_seed7.json
```

The helper defaults to `/usr/local/cuda` and the local `nano-vllm` conda Python; override `CUDA_HOME` and `GEMM_PYTHON` elsewhere. A normal benchmark call tunes automatically; `--tuning-report` loads and freezes an existing report's configurations. Optional `--shapes 'M,N,K;...'`, `--configs`, `--iterations`, and `--repeats` control the sweep.

Raw local artifacts (ignored by Git):
- `results/cuda_gemm_20261115_seed42.json`
- `results/cuda_gemm_20261115_seed17.json`
- `results/cuda_gemm_20261115_seed7.json`

The remaining gap is shape dependent. Small/skinny matrices still suffer from tile utilization, occupancy, and launch/synchronization costs. Further work could test split-K and register prefetching under the same accuracy and frozen-validation protocol. No unmeasured improvement is claimed.
