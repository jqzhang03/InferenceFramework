# CUDA GEMM: 1024 versus 4096 counter study — 2026-11-16

## Measurement boundaries

GPU: NVIDIA GeForce RTX 5060 Ti; PyTorch 2.8.0+cu128; WSL2; Nsight Compute 2026.3.1. 100 graph nodes, 7 randomized paired rounds, seeds 7/17/42; fixed configs, warm cache. Both backends use FP32 accumulation and preallocated output. Outputs pass FP64 sampled checks and full-output cuBLAS comparison before timing and before each counter capture. No CUDA kernel implementation or production dispatch was changed.

Primary counters: 3 independent captures/backend/dtype/size, alternating backend order; each captures one eager kernel after ten warmups. Fixed-kernel counter controls have one capture/point. Eight counter sections include detailed memory tables. Profiling uses kernel replay with clocks/cache control disabled. Counter medians and graph timings are separate datasets: do not use profiler durations as benchmark times, or subtract stall ratios from wall time. Historical 2026-11-15 measurements remain unchanged.

## 1. Unprofiled performance

Frozen configurations: cfg10 at 1024; cfg17 at 4096, matching the earlier comparison. Efficiency = median of paired cuBLAS/custom time ratios across seeds, hence can differ from a ratio of displayed medians.

| dtype | size | custom µs | cuBLAS µs | custom/cuBLAS throughput |
|---|---:|---:|---:|---:|
| fp16 | 1024 | 65.88 | 51.31 | 81.4% |
| fp16 | 4096 | 3202.71 | 3003.41 | 93.8% |
| bf16 | 1024 | 61.93 | 53.04 | 85.5% |
| bf16 | 4096 | 3108.11 | 2873.21 | 93.6% |

## 2. Tensor activity and achieved occupancy

Tensor elapsed uses the full kernel interval; Tensor active uses SM-active cycles. Pipe-active cycles are not useful FLOPs/peak or a speedup predictor: notably BF16 4096 has similar Tensor activity on both backends yet different graph timings. Instruction scheduling, work per active cycle and clocks still matter. The occupancy column uses active SM cycles and is not a direct fraction of device peak throughput.

| dtype | size | backend | Tensor elapsed % [min,max] | Tensor active % | achieved occupancy % | registers/thread | grid blocks | launch waves/SM |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| fp16 | 1024 | cuda | 69.12 [67.51,70.10] | 80.16 | 22.81 | 128 | 128 | 0.89 |
| fp16 | 1024 | cublas | 82.95 [78.25,83.00] | 95.71 | 15.76 | 96 | 256 | 3.56 |
| fp16 | 4096 | cuda | 95.65 [95.62,95.72] | 96.05 | 32.59 | 125 | 2048 | 14.22 |
| fp16 | 4096 | cublas | 99.05 [99.04,99.07] | 99.40 | 16.58 | 96 | 4096 | 56.89 |
| bf16 | 1024 | cuda | 69.46 [66.56,70.35] | 80.09 | 22.87 | 128 | 128 | 0.89 |
| bf16 | 1024 | cublas | 78.52 [75.06,79.20] | 92.26 | 8.33 | 234 | 64 | 1.78 |
| bf16 | 4096 | cuda | 95.64 [95.63,95.66] | 96.06 | 32.59 | 125 | 2048 | 14.22 |
| bf16 | 4096 | cublas | 95.55 [93.04,95.56] | 97.54 | 8.34 | 228 | 1024 | 28.44 |

Kernel symbols (first capture; cuBLAS config labels above identify the paired custom config, not an algorithm forced on cuBLAS):

- fp16 / 1024 / cuBLAS: `void Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_nn_align8>(Params)`
- fp16 / 4096 / cuBLAS: `void Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_nn_align8>(Params)`
- bf16 / 1024 / cuBLAS: `void Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nn_align8>(Params)`
- bf16 / 4096 / cuBLAS: `void Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn_align8>(Params)`

## 3. Warp states

Values below are Nsight `smsp__average_warps_issue_stalled_*_per_issue_active.ratio`, labelled **cycles per instruction** by `WarpStateStatistics.section` in the installed tool. They are not percentages of kernel time. In particular, math-pipe throttle can coexist with high Tensor utilization, and occupancy/stall values alone do not establish causality.

| dtype | size | backend | barrier | long scoreboard | short scoreboard | wait | math-pipe throttle |
|---|---:|---|---:|---:|---:|---:|---:|
| fp16 | 1024 | cuda | 1.353 | 0.512 | 0.510 | 2.390 | 5.888 |
| fp16 | 1024 | cublas | 0.902 | 0.384 | 0.116 | 1.488 | 6.233 |
| fp16 | 4096 | cuda | 2.380 | 0.337 | 0.656 | 2.546 | 9.461 |
| fp16 | 4096 | cublas | 1.142 | 0.086 | 0.030 | 1.617 | 8.371 |
| bf16 | 1024 | cuda | 1.318 | 0.527 | 0.510 | 2.390 | 5.886 |
| bf16 | 1024 | cublas | 0.095 | 0.161 | 0.264 | 3.473 | 7.358 |
| bf16 | 4096 | cuda | 2.368 | 0.338 | 0.656 | 2.546 | 9.449 |
| bf16 | 4096 | cublas | 0.043 | 0.090 | 0.089 | 4.145 | 8.892 |

## 4. Fixed cfg10: reduction length versus output grid

All four points use the exact same custom kernel and full output tiles. Counter controls have one capture per point except the repeated 1024³ primary. Timing efficiency remains paired over three seeds.

| dtype | M=N | K | custom µs | cuBLAS µs | efficiency | Tensor elapsed % | achieved occupancy % |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 1024 | 1024 | 65.88 | 51.31 | 81.4% | 69.12 | 22.81 |
| fp16 | 1024 | 4096 | 229.03 | 201.29 | 88.4% | 79.39 | 23.35 |
| fp16 | 4096 | 1024 | 820.14 | 746.98 | 91.6% | 88.57 | 32.48 |
| fp16 | 4096 | 4096 | 3221.78 | 3003.41 | 93.2% | 95.67 | 32.62 |
| bf16 | 1024 | 1024 | 61.93 | 53.04 | 85.5% | 69.46 | 22.87 |
| bf16 | 1024 | 4096 | 220.02 | 198.25 | 89.7% | 78.03 | 23.31 |
| bf16 | 4096 | 1024 | 813.21 | 768.90 | 94.6% | 88.20 | 32.49 |
| bf16 | 4096 | 4096 | 3096.70 | 2873.21 | 92.8% | 95.70 | 32.63 |

For fixed 1024×1024 output, descriptive fit `T = intercept + slope*K` over K=256/512/1024/2048/4096. Intercept bundles startup, teardown, epilogue and other fixed work; it is not an independently measured prologue duration.

| dtype | backend/config | intercept µs | slope µs/K | R² |
|---|---|---:|---:|---:|
| float16 | cfg10 | 9.972 | 0.053107 | 0.99922 |
| float16 | cfg17 | 8.846 | 0.052759 | 0.99946 |
| float16 | cublas | 1.722 | 0.048578 | 0.99990 |
| bfloat16 | cfg10 | 10.484 | 0.051039 | 0.99990 |
| bfloat16 | cfg17 | 9.921 | 0.051261 | 0.99980 |
| bfloat16 | cublas | 6.163 | 0.046760 | 0.99986 |

## 5. Full-tile scheduling boundary

N=K=1024, cfg10, M a multiple of 64; no partially filled output tiles. At 36 SMs and four resident CTAs/SM, M=1152 has 144 CTAs (one full capacity wave), while M=1216 has 152. `launch waves/SM` is a derived launch metric, not measured scheduling efficiency.

| dtype | M | grid blocks | custom µs | cuBLAS µs | Tensor elapsed % (one capture) | occupancy % (one capture) |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 1152 | 144 | 62.53 | 52.14 | 78.93 | 25.51 |
| fp16 | 1216 | 152 | 82.33 | 60.41 | 66.77 | 25.39 |
| bf16 | 1152 | 144 | 62.24 | 52.99 | 78.31 | 25.22 |
| bf16 | 1216 | 152 | 77.76 | 78.23 | 64.87 | 25.59 |

## Interpretation and limits

The small matrix has both fewer output CTAs and fewer K iterations. Relative startup/teardown costs and underfilled/short-lived CTA execution reduce Tensor utilization more than in cuBLAS. Longer K and/or a larger output grid improve utilization with the exact cfg10 kernel. The full-tile wave-boundary timing discontinuity supports scheduling sensitivity independently of edge masking.

An occupancy-only diagnosis is contradicted by FP16 cuBLAS: it is faster at 1024³ while using lower achieved occupancy. Barrier and math-pipe stall ratios do not rise uniquely at the small shape; their larger 4096 values must not be read as lost-time percentages. The custom kernel has two CTA barriers per BK=32 iteration, cp.async staging, ldmatrix loads, and scalar narrow/store epilogue work; these are plausible optimization targets, but this study does not individually time them or prove a single instruction is responsible for the entire gap.

L2 hit rates, memory bytes, spill metrics, and the exact kernel symbols are retained in counters.json with their original units. Warm-cache results cannot establish cold-DRAM behavior. The requested sections do not expose a direct shared-bank-conflict count in this SM120 capture; address models and scoreboard stalls cannot substitute for one. Previous stage-depth and A-swizzle controls remain separate historical evidence, not newly measured improvements. No kernel optimization is claimed here.

## Reproduction

```bash
bash benchmarks/run_cuda_gemm.sh --counter-study
python -m benchmarks.cuda_gemm_counter_report
```

Local raw artifacts: `results/cuda_gemm_counter_20261116/timings.json`, `counters.json`, and `captures/` (versioned ncu reports, raw CSV, full commands/status JSON, logs). Each capture verifies accuracy before starting the profiled range. The operator implementation and previous dated reports are preserved.
