# CUDA GEMM profiling and controlled attribution — 2026-11-15

## Scope and evidence boundary

Historical scope: the attribution dataset below predates the same-day Nsight Compute recovery. Linux ncu 2026.3.1 subsequently collected counters successfully; see the separate [recovery record](_ncu_recovery_20261115.md). Counter-unavailable statements below describe the original 2025.1.1 attempt only, and the original timing data are retained.

Question: why does the current handwritten FP16/BF16 GEMM get closer to cuBLAS at 4096³ than at 1024³? This report preserves the original 21 production configuration IDs and default address layout. Added CUDA entry points are diagnostic controls, not new engine dispatch or automatic optimization.

GPU: NVIDIA GeForce RTX 5060 Ti; PyTorch 2.8.0+cu128; WSL2. Controlled timing: 30 nodes per graph; 5 paired randomized rounds; 3 seeds; preallocated buffers; warm cache. Each variant passes FP64 sampled/full-reference and full-output cuBLAS checks before timing. Time columns are medians of per-seed medians; ratios/changes are medians of paired-seed ratios and need not equal ratios of the displayed time medians. No performance test runs concurrently with another GPU profiler. Clocks/caches are not externally locked, so profiling times are diagnostic and not a replacement for the original frozen-configuration benchmark.

Nsight Systems 2026.5.1 successfully collected CUDA kernel timestamps, launch resources, and graph/eager gaps. Nsight Compute 2025.1.1 failed before collecting counters: `Failed to initialize the profiler: LibraryNotLoaded` (exit 9). Therefore no measured warp-stall, Tensor Core utilization, achieved occupancy, DRAM/L2 bandwidth, or bank-conflict-rate percentages are available. CUDA occupancy API values below are static residency limits, not achieved occupancy.

Definitions: [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html). Conclusions below come from local measurements, not from that guide.

## 1. Kernel time versus launch/graph gaps

Medians over three NVTX ranges, 20 calls/range, using GPU timestamps from the SQLite export. NVTX ranges include synchronization so all twenty kernels fit inside their labels. Kernel duration excludes inter-kernel gaps. Profiled timings must not be mixed arithmetically with separate unprofiled runs.

| dtype | size | backend/config | kernel µs/call | inter-kernel gap µs | kernel+gap span µs/call |
|---|---:|---|---:|---:|---:|
| fp16 | 1024 | cuda/cfg10 | 58.192 | 0.174 | 58.352 |
| fp16 | 1024 | cublas/cfg10 | 47.995 | 0.172 | 48.151 |
| fp16 | 4096 | cuda/cfg17 | 3058.654 | 0.202 | 3058.851 |
| fp16 | 4096 | cublas/cfg17 | 2920.873 | 0.214 | 2921.075 |
| bf16 | 1024 | cuda/cfg10 | 61.288 | 0.173 | 61.453 |
| bf16 | 1024 | cublas/cfg10 | 49.451 | 0.171 | 49.602 |
| bf16 | 4096 | cuda/cfg17 | 3050.267 | 0.212 | 3050.469 |
| bf16 | 4096 | cublas/cfg17 | 2870.233 | 0.260 | 2870.435 |

The 1024³ graph gaps are approximately equal for custom and cuBLAS, while the kernel durations themselves differ by several microseconds. Host/Python launch starvation is not the main cause of this graph-replay gap. Eager tracing records larger gaps (see raw JSON), which do not explain the original graph benchmark. This does not exclude startup/teardown work *inside* each kernel/CTA.

## 2. Registers and residency

| config | size | registers/thread | local bytes/thread | dynamic shared bytes | resident blocks/SM limit | theoretical occupancy | grid blocks | waves model |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1024 | 128 | 0 | 24576 | 4 | 33.3% | 128 | 0.889 |
| 10 | 4096 | 128 | 0 | 24576 | 4 | 33.3% | 2048 | 14.222 |
| 17 | 1024 | 125 | 0 | 24576 | 4 | 33.3% | 128 | 0.889 |
| 17 | 4096 | 125 | 0 | 24576 | 4 | 33.3% | 2048 | 14.222 |

Both shapes have the same static residency ceiling; neither reports local-memory allocation. The claim '1024 has lower theoretical occupancy or register spilling' is unsupported. The grids differ substantially: 128 blocks versus 2048, on 36 SMs with room for four blocks/SM. A simple equal-duration wave model gives 88.9% versus 94.8% fill, but this is a model, not a measured active-SM fraction.

Unused-shared-memory control: keep the exact kernel instructions/layout, reserve extra dynamic shared memory to reduce allowable residency. FP16 cfg10:

| size | 4 blocks/SM, 24 KiB | 3 blocks/SM, 32 KiB | 2 blocks/SM, 40 KiB | 1 block/SM, 56 KiB |
|---:|---:|---:|---:|---:|
| 1024 | 62.77 µs | 63.20 µs | 65.80 µs | 97.55 µs |
| 4096 | 3053.00 µs | 3092.32 µs | 3125.48 µs | 4957.72 µs |

Reducing residency to one block is costly for both shapes. Four-to-three blocks has a much smaller effect. This validates residency sensitivity, but does not make a different static occupancy ceiling the cause of the original 1024-versus-4096 difference.

## 3. Independent K and output-grid controls

Use the exact same cfg10 kernel for every cell. Change K while holding M/N and the grid fixed; then change M/N while holding K fixed. Ratios are medians of three paired-seed cuBLAS/custom time ratios.

| dtype | M=N | K | custom µs | cuBLAS µs | cuBLAS efficiency |
|---|---:|---:|---:|---:|---:|
| float16 | 1024 | 1024 | 62.22 | 51.42 | 82.8% |
| float16 | 1024 | 4096 | 222.70 | 196.43 | 88.2% |
| float16 | 4096 | 1024 | 805.61 | 712.53 | 88.4% |
| float16 | 4096 | 4096 | 3102.46 | 2953.83 | 94.7% |
| bfloat16 | 1024 | 1024 | 62.49 | 53.50 | 85.5% |
| bfloat16 | 1024 | 4096 | 219.91 | 197.51 | 89.9% |
| bfloat16 | 4096 | 1024 | 805.93 | 760.75 | 95.1% |
| bfloat16 | 4096 | 4096 | 3015.44 | 2891.48 | 95.9% |

Both K length and output shape matter. For transparency, an algebraic symmetric two-factor decomposition of the above four cells is:

- float16: total efficiency change 11.82 percentage points; symmetric two-factor decomposition: K +5.76 pp, M/N +6.05 pp.
- bfloat16: total efficiency change 10.38 percentage points; symmetric two-factor decomposition: K +2.56 pp, M/N +7.82 pp.

These percentage points describe controlled *shape effects*, not hardware stall percentages. The M/N term also changes cache behavior and possibly the cuBLAS algorithm; it must not be relabeled entirely as 'wave loss'. The K term establishes an amortization/steady-state effect, but does not by itself identify a particular barrier or instruction.

For fixed 1024×1024 output, fit `T = intercept + slope*K` over K=256/512/1024/2048/4096 (descriptive model):

| dtype | backend | intercept µs | slope µs/K | R² |
|---|---|---:|---:|---:|
| float16 | cfg10 | 10.444 | 0.051613 | 0.999740 |
| float16 | cublas | 3.423 | 0.047067 | 0.999984 |
| bfloat16 | cfg10 | 11.265 | 0.050827 | 0.999895 |
| bfloat16 | cublas | 7.150 | 0.046376 | 0.999882 |

The custom fit has a nontrivial fixed component plus a different per-K slope. The intercept bundles kernel/CTA startup, prologue/epilogue, scheduling and other non-K work; it is not a direct stopwatch measurement of pipeline warmup. A sole 'launch overhead' or sole 'pipeline depth' explanation would be too strong.

## 4. Scheduling steps with no partial tiles

N=K=1024; vary only M in multiples of 64, so every 64×128 output tile is full. FP16 cfg10:

| M | grid blocks | custom µs | cuBLAS µs |
|---:|---:|---:|---:|
| 960 | 120 | 62.29 | 46.33 |
| 1024 | 128 | 61.58 | 51.06 |
| 1088 | 136 | 61.88 | 51.42 |
| 1152 | 144 | 62.58 | 51.63 |
| 1216 | 152 | 78.21 | 58.31 |
| 1280 | 160 | 79.38 | 58.64 |
| 1536 | 192 | 88.98 | 70.61 |
| 4096 | 512 | 213.42 | 180.99 |

Crossing from 144 blocks (M=1152) to 152 (M=1216) crosses the nominal 36×4 residency capacity. Custom time jumps much more than the 5.6% extra work. This is direct evidence of grid/scheduling granularity, without an edge-mask confound. cuBLAS has its own steps and, for BF16, changes kernel families; the experiment does not isolate every cache/dispatch mechanism.

## 5. Pipeline and shared-layout hypotheses tested

Nsight kernel names/resources confirm different cuBLAS choices: FP16 uses a named 64×64_32×6 CUTLASS kernel in both sizes; BF16 uses named 128×128_32×5 at 1024 and 256×64_32×4 at 4096. The static trace contains no measured memory-stall rate.

Fixed 64×64 tile and warp layout; vary stage count. A second control pads unused shared memory so stages 2–6 all reserve 48 KiB and have the same residency limit. FP16 at 1024³:

| stages | natural µs | equalized shared-memory µs |
|---:|---:|---:|
| 2 | 62.33 | 68.81 |
| 3 | 62.94 | 65.69 |
| 4 | 67.66 | 73.51 |
| 5 | 67.18 | 67.24 |
| 6 | 67.87 | 67.95 |

Simply increasing stage count did not close the gap, including with equalized shared-memory capacity. Therefore the cuBLAS kernel's stage-count label is not sufficient to explain its advantage.

A-address bank-layout intervention: the original BK=32 address model maps eight 16-byte ldmatrix row segments to bank groups `[0,5,2,7,0,5,2,7]`; the alternative uses `[0,4,1,5,2,6,3,7]`. This predicts different bank behavior; no bank-conflict hardware counter was collected. Keep mathematical operations, block/warp tiles and stages fixed, and change only the A copy/read address mapping:

| dtype | size | cfg10 original µs | cfg10 alternative µs | alternative speed change | cfg17 original µs | cfg17 alternative µs |
|---|---:|---:|---:|---:|---:|---:|
| float16 | 1024 | 61.83 | 64.54 | -4.4% | 61.35 | 63.42 |
| float16 | 4096 | 3089.91 | 3128.59 | -2.1% | 3070.41 | 3081.32 |
| bfloat16 | 1024 | 62.05 | 64.42 | -3.7% | 61.31 | 63.50 |
| bfloat16 | 4096 | 3072.78 | 3132.83 | -1.9% | 3048.03 | 3068.81 |

The intervention does not establish this predicted bank conflict as the dominant practical gap. The source-level mapping changes can also alter address instructions/register allocation, so an address model must not be substituted for measured bank-stall rates. The alternative layout remains diagnostic and is not enabled by default.

## Conclusion

The verified explanation is a combination of shorter-K amortization/steady-state costs and output-grid/scheduling/shape-selection effects. The measured relative importance differs between FP16 and BF16 because the baseline selects different kernels. Host launch gaps, a lower static occupancy ceiling, and register spilling are not supported as the principal explanations. Merely deepening the pipeline or changing this A swizzle does not reproduce cuBLAS efficiency. Attribution below this level (specific wait/barrier/ldmatrix/epilogue cycle shares) remains unmeasured because hardware counters could not be initialized; no invented stall percentages are reported.

## Reproduce

```bash
bash benchmarks/run_cuda_gemm.sh --profile --mode ncu
bash benchmarks/run_cuda_gemm.sh --profile --mode suite
bash benchmarks/run_cuda_gemm.sh --profile --mode rows
bash benchmarks/run_cuda_gemm.sh --profile --mode stages
bash benchmarks/run_cuda_gemm.sh --profile --mode layout
bash benchmarks/run_cuda_gemm.sh --profile --mode nsys
python -m benchmarks.cuda_gemm_profile --mode analyze
python -m benchmarks.cuda_gemm_profile_report
```

Raw JSON, profiler logs, `.nsys-rep` and SQLite are local ignored files under `results/cuda_gemm_profile/`. `--mode tools` locates profilers; `--mode trace` profiles one case. The WSL helper supplies this machine's toolchain/Python defaults; override CUDA_HOME/GEMM_PYTHON on other systems. Diagnostic timings are separate dated observations and do not overwrite the original performance report.
