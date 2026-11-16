# Native SM120 CUDA FP8 GEMM — 2026-11-16

## Implementation and contract

Independent handwritten CUDA operator: `C[M,N] = alpha * A[M,K] * W[N,K]^T`. A and W contain packed FP8, output is BF16, accumulation is FP32. W is pretransposed/packed row-major [N,K]; it is not the [K,N] layout of the earlier FP16/BF16 experiment. The custom kernel calls no cuBLAS/CUTLASS/Triton kernels. It is not wired into inference-engine dispatch.

The native path targets **SM120a** and uses `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4`, cp.async 16-byte global-to-shared copies, zero-filled K/M/N tails, ldmatrix register fragments, XOR-swizzled shared memory, multistage ring buffers, and paired BF16 stores where alignment permits. Grouped configurations visit up to eight M tiles before advancing N, improving opportunities to reuse weight tiles in L2. Configs vary block/warp/K tiles and pipeline depth; the final tuple field is M-group size (1 = original raster). This is the consumer Blackwell warp-MMA path, not a tcgen05/TMEM implementation.

Requires SM120, CUDA Toolkit 12.8+, contiguous 16-byte-aligned A/W and K divisible by 16. Partial output tiles, K not divisible by BK, empty M/N, and K=0 are supported by the custom path. Other K alignment is rejected explicitly. Finite host alpha is applied in the epilogue and may represent the product of per-tensor dequantization scales. No block scaling, fused quantization, autograd, beta accumulation, TP, or engine integration is provided. E4M3 is benchmarked; E5M2×E5M2 is separately tested in the custom kernel only, with no cuBLASLt performance claim.

Instruction definitions: [NVIDIA PTX ISA 8.7](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html). Baseline API: [NVIDIA cuBLASLt documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul). SM120a is explicitly selected because the `.kind` qualifier needs the architecture-specific target in CUDA 12.8.

## Method

GPU: NVIDIA GeForce RTX 5060 Ti; PyTorch 2.8.0+cu128; WSL2. A[M,K], W[N,K], C[M,N], row major; E4M3/E4M3 -> BF16; FP32 accumulation. prepacked common inputs; no quantization/transpose in timings; 100-node graphs; paired randomized rounds; cuBLASLt heuristics tuned outside validation.

Custom and cuBLASLt receive exactly the same quantized inputs, scalar scale product, output dtype and stream. Per-tensor E4M3 quantization uses amax/448. Compilation, quantization, transpose/packing, memory allocation, descriptor/algorithm setup, graph capture, and checks are excluded from timings. These are repeatedly reused buffer/operator measurements, not serving speedups or quantization-inclusive latency. Weight matrices larger than L2 cannot be assumed cache-resident merely because addresses are reused.

Custom candidates and up to 32 cuBLASLt heuristic candidates are timed on separate seed-20261116 buffers. Every selected candidate passes an FP64 decoded-input check. The selected custom config and cuBLASLt algorithm attributes (ID/tile/stages/split-K/reduction/swizzle/custom/workspace) are frozen and checked across validation inputs. cuBLASLt uses CUBLAS_COMPUTE_32F, FAST_ACCUM=0 and up to 32 MiB workspace. This is a tuned heuristic subset, not an exhaustive search of all cuBLASLt algorithms. Plan creation is outside the timed path; plans own their descriptors and workspace.

Validation uses seeds 7/17/42, 100 graph nodes and seven randomized paired rounds/seed. Times/TFLOPS are medians of per-seed medians. Efficiency is median paired cuBLASLt/custom time ratio, so it need not equal the ratio of displayed medians. GPU clocks are not locked; seed ranges show run variation. Previous dated FP16/BF16 results are not a same-run comparison against FP8.

## Performance

| M×N×K | cfg | block / K / stages / group-M | CUDA µs | cuBLASLt µs | CUDA TFLOPS | cuBLASLt TFLOPS | relative throughput [seed range] |
|---|---:|---|---:|---:|---:|---:|---:|
| 64×4096×4096 | 18 | 64×64 / 64 / 2 / 8 | 30.62 | 26.35 | 70.14 | 81.50 | 86.1% [85.9%, 86.3%] |
| 256×4096×4096 | 0 | 64×64 / 64 / 2 / 1 | 109.12 | 102.65 | 78.72 | 83.68 | 93.8% [92.0%, 95.1%] |
| 1024×1024×1024 | 14 | 64×128 / 64 / 2 / 1 | 28.89 | 26.88 | 74.34 | 79.89 | 93.1% [93.1%, 95.1%] |
| 2048×2048×2048 | 18 | 64×64 / 64 / 2 / 8 | 199.82 | 197.68 | 85.98 | 86.91 | 99.0% [98.4%, 101.1%] |
| 4096×4096×4096 | 14 | 64×128 / 64 / 2 / 1 | 1483.12 | 1457.04 | 92.67 | 94.33 | 97.0% [95.5%, 98.2%] |
| 128×11008×4096 | 25 | 128×64 / 128 / 2 / 8 | 142.61 | 128.86 | 80.94 | 89.58 | 89.1% [88.6%, 90.4%] |
| 512×4096×11008 | 18 | 64×64 / 64 / 2 / 8 | 549.82 | 504.20 | 83.97 | 91.57 | 91.8% [91.2%, 92.5%] |
| 8192×4096×4096 | 14 | 64×128 / 64 / 2 / 1 | 2983.56 | 2921.25 | 92.13 | 94.10 | 97.7% [97.1%, 97.9%] |

## Accuracy

All benchmark outputs are finite and pass the decoded-FP8 reference check (relative L2 limit 0.004) and full-output cuBLASLt comparison. Reference outputs use FP64 multiplication over every K term: full small outputs, or a deterministic sample of up to 64×64 positions for large outputs.

- Maximum arithmetic/output-rounding relative L2 against decoded-FP8 FP64 reference: **0.00169845**.
- Maximum total relative L2 against the common original FP32 input reference (including E4M3 quantization): **0.03939476**.
- Maximum full-output relative L2 versus cuBLASLt: **0.00005921**.

Quantization error is reported separately and is not an arithmetic-error tolerance. These random-matrix tests do not establish end-model quality. GPU tests additionally cover all configurations, odd M/N, partial K tiles, grouped-raster tails, zero/constant/identity patterns, E5M2 arithmetic, output scaling, graph capture, non-default streams, empty shapes, K=0, invalid API inputs and cuBLASLt plan mismatch.

## Grouped-scheduling control

Same input buffers, tile/warp/K/stage configuration, and math; only CTA raster changes. Three seeds × seven randomized paired rounds × 100 graph nodes. All group-only pairs have bitwise-identical outputs. Values below are median paired speedup (original/grouped time); >1 is faster. This is a timing intervention, not a measured breakdown of DRAM/L2 traffic.

| M×N×K | cfg0→18 (64×64) | cfg3→19 (64×128) | cfg11→22 (128×128, BK128) |
|---|---:|---:|---:|
| 128×11008×4096 | 1.919× | 1.103× | 0.994× |
| 512×4096×11008 | 1.123× | 1.019× | 0.991× |
| 4096×4096×4096 | 1.012× | 0.995× | 0.994× |

Grouping helps some memory-sensitive shapes/configurations and can be neutral or slightly slower elsewhere. The best 128×11008×4096 configuration also changes tile size (cfg25, 128×64); its gain cannot be attributed wholly to grouping. Keep the original and grouped candidates in tuning rather than enabling grouping universally.

## Native instruction and counter validation

The final binary's SASS contains `QMMA.16832.F32.E4M3.E4M3`, `LDSM`, and `LDGSTS`; FP8 is multiplied natively rather than dequantized into a BF16 GEMM. One profiled launch/backend/size after ten warmups validates counters; these are diagnostic single samples, separate from the repeated benchmark.

| size | backend | Tensor pipe active / elapsed % | achieved occupancy % |
|---:|---|---:|---:|
| 1024 | cuda | 72.06 | 23.59 |
| 1024 | cublas | 76.39 | 12.38 |
| 4096 | cuda | 98.18 | 32.65 |
| 4096 | cublas | 98.47 | 16.48 |

Pipe activity is not useful-FLOP utilization or a direct throughput ratio. Counter timings are not substituted for graph benchmark times. Full logs, raw units, native instruction counts and disassembly are retained locally.

## Reproduction and artifacts

```bash
bash benchmarks/run_cuda_gemm.sh --fp8 --check-only
bash benchmarks/run_cuda_gemm.sh --fp8 --output results/cuda_fp8_gemm_grouped_20261116.json
bash benchmarks/run_cuda_gemm.sh --fp8 --tuning-report results/cuda_fp8_gemm_grouped_20261116.json --output results/cuda_fp8_gemm_frozen.json
bash benchmarks/run_cuda_gemm.sh --fp8-profile --input results/cuda_fp8_gemm_grouped_20261116.json --output-dir results/cuda_fp8_grouped_profile_20261116
python -m pytest tests/test_cuda_fp8_gemm.py -q
python -m benchmarks.cuda_fp8_gemm_report
```

Formal raw results: `results/cuda_fp8_gemm_grouped_20261116.json`. Counter/disassembly evidence: `results/cuda_fp8_grouped_profile_20261116/`. Earlier `cuda_fp8_gemm_final_20261116.json` / `cuda_fp8_profile_20261116/` are separate initial 18-config observations; do not combine their times arithmetically with the later grouped-config run. The helper supplies this machine's Python/toolkit defaults; native compilation uses GCC 13 and sm_120a. The FP16/BF16 implementation and its original 21 config IDs remain unchanged.
