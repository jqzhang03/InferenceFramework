# Qwen3-0.6B CUDA projection integration — 2026-11-16

RTX 5060 Ti / SM120, BF16 model, TP=1, eager prefill and CUDA Graph decode. 112 decoder projections are installed (QKV/O and MLP); vocabulary head and FlashAttention are unchanged.

Latency/SLO-aware prefill quotas and TPOT adaptation are disabled; fixed 2048-token budget fits each prefill in one forward. Earlier adaptive-scheduler attempts remain under results/qwen_cuda_inference_20261116 and are not pooled here. Their first-chunk logit diagnostic compared different positions and must not be used as precision evidence.

Three synthetic token workloads, fixed 32-token outputs, five distinct prompts per condition, two fresh-process rounds in opposite backend orders. One warmup per condition. Initialization/compilation/packing are excluded; dynamic FP8 activation quantization and allocation are included. This is offline generation, not HTTP/SLO goodput or real-text quality evaluation.

Reported time is the mean of the two per-round medians (five complete generate() wall times per round). Speedup is library time / backend time. Round-to-round variation is retained below; there is no claim of a universal speedup.

| Batch / input tokens | PyTorch ms | CUDA initial ms | CUDA tuned ms | BF16 hybrid ms | FP8 hybrid ms |
|---|---:|---:|---:|---:|---:|
| 1 / 128 | 195.39 | 143.59 | 143.88 | 145.35 | 131.58 |
| 8 / 128 | 170.36 | 237.59 | 186.14 | 170.97 | 166.69 |
| 1 / 1024 | 168.45 | 159.71 | 158.76 | 161.69 | 168.31 |

## Repeat variation and token parity

| Backend | Batch / input | Round medians (ms) | Whole request token parity against paired PyTorch |
|---|---|---|---|
| CUDA initial | 1 / 128 | 150.73, 136.45 | 6/10 |
| CUDA initial | 8 / 128 | 241.64, 233.54 | 80/80 |
| CUDA initial | 1 / 1024 | 159.73, 159.68 | 10/10 |
| CUDA tuned | 1 / 128 | 134.26, 153.50 | 6/10 |
| CUDA tuned | 8 / 128 | 181.55, 190.73 | 62/80 |
| CUDA tuned | 1 / 1024 | 159.46, 158.06 | 10/10 |
| BF16 hybrid | 1 / 128 | 154.31, 136.39 | 6/10 |
| BF16 hybrid | 8 / 128 | 175.48, 166.47 | 80/80 |
| BF16 hybrid | 1 / 1024 | 165.98, 157.39 | 10/10 |
| FP8 hybrid | 1 / 128 | 131.08, 132.08 | 4/10 |
| FP8 hybrid | 8 / 128 | 193.58, 139.81 | 32/80 |
| FP8 hybrid | 1 / 1024 | 143.80, 192.81 | 8/10 |

## Fixed-input first-position logit diagnostic

Two synthetic prompts (128/1024 tokens), one output position each; collected after timing. These are numerical checks, not perplexity/task-quality evidence.

| Backend | Input length | Relative L2 vs BF16 library | Top-1 equal |
|---|---:|---:|---|
| CUDA initial | 128 | 0.0114365 | True |
| CUDA initial | 1024 | 0 | True |
| CUDA tuned | 128 | 0.0114365 | True |
| CUDA tuned | 1024 | 0 | True |
| BF16 hybrid | 128 | 0 | True |
| BF16 hybrid | 1024 | 0 | True |
| FP8 hybrid | 128 | 0.0986956 | True |
| FP8 hybrid | 1024 | 0.0708904 | True |

## Follow-up: FP8 prefill only

A separate two-process screen replaces FP8 selections below 32 rows with the BF16 hybrid selections. It was run after the primary comparisons, not in the reverse-order pairing; it does not establish a stable improvement.

| Batch / input | Mean of round medians (ms) | Round medians (ms) |
|---|---:|---|
| 1 / 128 | 156.36 | 136.47, 176.24 |
| 8 / 128 | 168.51 | 178.01, 159.01 |
| 1 / 1024 | 173.45 | 197.44, 149.45 |

## Optimizations and limits

- Vectorized row-major GEMV avoids the large MMA tile for small batches. Runtime GEMM adds 16/32-row tiles (configs 21–23); original standalone IDs stay unchanged.
- Offline shape tuning freezes tile/warp/stage choices. BF16 hybrid keeps the library on slower or unmeasured shapes; FP8 hybrid additionally considers native FP8 MMA.
- FP8 packs weights once, quantizes BF16 activations per row in a CUDA kernel, and applies row scales in the native FP8 MMA epilogue. This replaces multi-operation runtime quantization.
- The initial CUDA variant already includes GEMV, but uses cfg10 for GEMM. The tuned variant includes the new small tiles. Thus the initial/tuned comparison isolates configuration selection, not the entire GEMV contribution.
- Raw tuning uses repeated warm-cache matrices and is distinct from model inference. FP8/BF16 compare different arithmetic precisions. Some BF16 generation token sequences also diverge; do not describe these paths as bitwise equivalent.
- Original BF16 weights remain resident; transposed copies and optional FP8 copies increase memory consumption. These are not weight-memory compression measurements.
- Defaults remain PyTorch. Unsupported layouts fall back; FP8 requires BF16 Qwen3 on SM120; TP>1, PD and combination with existing quantization are rejected. Mixed-batch CUDA Graph capture is unverified; decode replay is tested.
- GPU tests: 3 passed (FP16/BF16 accuracy + changing-input graph replay; FP8 zero row, per-row scale epilogue and graph replay). Tuning additionally checks every candidate and freezes selections before held-out BF16 activation checks.

## Reproduce

```bash
bash benchmarks/run_qwen_cuda.sh --backend cuda_fp8_auto --output results/qwen_cuda_tune.json
bash benchmarks/run_qwen_cuda.sh --backend cuda --output results/qwen_cuda_bf16_tune.json
bash benchmarks/run_qwen_cuda.sh --pytest
bash benchmarks/run_qwen_cuda.sh --mode suite --output results/qwen_cuda_controlled_20261116
bash benchmarks/run_qwen_cuda.sh --mode prefill-suite --output results/qwen_cuda_controlled_20261116
python -m benchmarks.qwen_cuda_linear_report
```

Raw reports and frozen selectors are local ignored artifacts under results/. Historical standalone results remain separate.
