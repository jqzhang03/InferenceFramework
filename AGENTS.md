# Repository Guidelines

### Warp-owned M1-4 GEMV (2026-11-17 follow-up)

`run_qwen_cuda.sh --warp-gemv-study --output <fresh-dir>` compares Decode38–49
against frozen runtime, ordinary GEMV and shared SIMT. Each 128-thread CTA uses
8/16/32-lane groups, 1/2 outputs and 1/2 accumulation chains; M1-4 are specialized.
The kernel uses register reuse and shuffle, no shared allocation, CTA barrier or
Split-K buffer. Keep all lanes participating in shuffles on boundary outputs.
Held-out42/99/123 accepted 6 BF16 shapes (M2 QKV/gate-up, all M3 projections) and
12 FP16 shapes (all M2-4 projections) on RTX5060Ti, now in bundled cuda_decode
selectors. M1, BF16 M4, M8+ and unmatched shapes retain prior choices. Explicit
tuning still overrides. Preserve profiles/cuda_warp_gemv_20261117/; use independent
graph timings, not NCU replay durations. --warp-gemv-pytest passed 29 tests (24 GPU,
5 profiling parser), including GPU boundary,
stream, graph, selector and existing model checks plus profiling attribution.
See benchmarks/_cuda_warp_gemv_20261117.md. Earlier BF16-unchanged notes describe
the prior tile/pipeline study, before this follow-up.

### CUDA pipeline tuning follow-up (2026-11-17)

`run_qwen_cuda.sh --pipeline-study --output <fresh-dir>` screens native CUDA
tile/warp/stage configurations, validates on held-out seeds, and collects NCU
controls. `--pipeline-pytest` passed 15 GPU tests, including actual selector
installation, changed-input graph replay and default BF16 model generation.
New candidate IDs: Decode 24–37, GEMM 24–35, FP8 28–33; historical IDs remain.
RTX 5060 Ti frozen selectors now cover FP16 cuda_decode B8 projections and
BF16-activation cuda_fp8 M128 QKV; explicit tuning wins, unmatched shapes retain
legacy dispatch. The default BF16 cuda_decode selector is unchanged. Preserve
Split-K finish/FP8 pack and all 87 native NCU reports under
profiles/cuda_pipeline_20261117/. More work per thread, deeper stages and fewer
shared conflicts did not consistently improve latency; use independent paired
graph timings, not replay durations. DeepSeek float probes are not deployed INT4.
See benchmarks/_cuda_pipeline_20261117.md for gates, counters and limitations.

### Custom-only CUDA profiling (2026-11-17)

`run_qwen_cuda.sh --operator-profile --output <fresh-dir>` runs independent
timings, eager/graph Nsight Systems capture and warm/flushed-cache Nsight Compute
capture for runtime CUDA projection kernels. `--operator-pytest` checks attribution
and multi-kernel CSV units. Formal ranges are `op_capture/`; initialization and
FP64 references are outside them. Keep Split-K finish and FP8 pack kernels.
DeepSeek floating-point shapes are standalone probes, not deployed INT4 MoE.
Use three-seed graph timings for performance, not NCU replay durations; active
GPU span is not occupancy. Preserve native .nsys-rep/.ncu-repz, raw units and logs.
See benchmarks/_cuda_operator_profile_20261117.md. Runtime kernels/selectors unchanged.

### Qwen3-0.6B default linear backend (2026-11-17)

Config.linear_backend=None resolves after loading HF metadata: FP16/BF16 Qwen3
with (hidden/intermediate/layers/Q heads/KV heads/head_dim) =
(1024,3072,28,16,8,128), TP=1, no weight quantization and no explicit PD uses
cuda_decode. Other combinations resolve to torch; explicit backends always win.
This applies to LLM, create_app and the server CLI; --linear-backend torch restores
the library baseline, and --cuda-linear-tuning supplies frozen selectors.
Default CUDA prevents execution_mode=auto from choosing PD; explicit PD retains
torch by default. Config tests: run_qwen_cuda.sh --config-pytest (17 passed).
--decode-pytest passed seven GPU tests including real default-backend Qwen3-0.6B
generation (112 adapters, eight requests x four tokens, decode graph replay).
Historical benchmark reports retain their original explicit backends and results.

### cfg22 Nsight Compute diagnosis (2026-11-17)

`run_qwen_cuda.sh --cfg22-ncu` runs the isolated timing/counter harness;
`--cfg22-pytest` validates diagnostic A swizzle in FP16/BF16 with graph replay,
and `--cfg22-report` renders benchmarks/_cfg22_ncu_20261117.md. Raw results remain
under results/cfg22_ncu_20261116/ (38 isolated + two in-model application replays).
Runtime cuda_gemm diagnostic variant14 retains cfg22 geometry while changing
only A swizzle; it is not normal cfg14 and is not deployed. Normal cfg22 and
selectors are unchanged. BF16 B8 model QKV/gate-up L2 hits are 8.35%/8.70% and
DRAM reaches 81.54%/85.09% of NCU sustained peak. The measured streaming path is
primarily weight-bandwidth limited; warm-cache grid/latency/shared conflicts
must be evaluated separately. Do not use NCU replay timings as benchmark wins.
Use timing_amortized.json (three seeds), not the early single-node timing.json
pilot. NCU native output is compressed .ncu-repz; preserve reports and raw units.

### Current Decode full inference profiling (2026-11-16)

`run_qwen_cuda.sh --nsys --suite decode-current --output <fresh-dir>` profiles
cuda_decode using bundled selectors with default scheduling/multi-step/mixed
Graph flags, plus controlled and eager diagnostics. Resource caps remain explicit
(16 sequences, 2048 batched/context tokens, 0.55 GPU memory utilization).
`--full-nsys-report --root <dir>` runs qwen_decode_full_report.py, producing
phase/component CSV/JSON and the dated full report. Startup/warmup NVTX scopes
are separate from nsys_capture. Count Mixed forwards independently; the default
B=8 run recorded 3 prefill, 3 mixed, 93 decode forwards across three batches.
Eager leaf-module launch correlation provides non-overlapping GPU attribution;
only unique RoPE/Embedding/Norm/SiLU kernel identities transfer to graph labels.
Do not sum inclusive CPU NVTX with GPU kernel time or equate GPU activity union
with occupancy. See benchmarks/_qwen_decode_full_nsys_20261116.md.

### CUDA Decode weight reuse (2026-11-16)

`layers/csrc/cuda_decode.cu` adds NT-oriented m16n8 MMA, shared XOR swizzle,
cp.async staging, FP32 Split-K/finish, and batched SIMT candidates with padded
uint4 shared vectors and 8/16/32-lane shuffle reduction. Configs 0–22 are normal;
23 is no-swizzle diagnostic only. `linear_backend="cuda_decode"` automatically
loads cuda_decode_sm120.json on RTX 5060 Ti BF16; otherwise it retains legacy
dispatch unless supplied matching selectors. Only measured winning O/down shapes
enable new kernels; M=1 and QKV/gate-up keep previous kernels. The initial rollout
was opt-in; the 2026-11-17 Qwen3-0.6B default is described above.
FP16/BF16 correctness and graph replay are GPU tested; TP=1 mixed only.
`cuda_decode_study.py` tunes, validates rotating 28 weights, freezes deployment,
replays model pairs and collects ncu/nsys; helper flags are --decode-study,
--decode-pytest, --decode-report. Keep dated raw results and old kernels intact.
Representative down bank-conflict counters fell 516330 -> 245 with swizzle;
not all residual conflicts are zero. Compute Sanitizer could not initialize the
WDDM debugger interface; do not claim a passed memcheck. See
benchmarks/_cuda_decode_reuse_20261116.md and paired Windows traces under
profiles/cuda_decode_reuse_20261116/.

### Qwen CUDA projection integration (2026-11-16)

Nsight trace collection: `bash benchmarks/run_qwen_cuda.sh --nsys --output <fresh-dir>`
and `--nsys-report --root <dir>` use qwen_cuda_nsys.py / qwen_cuda_nsys_report.py.
Six graph/eager/baseline/FP8 cases capture full-process CUDA/NVTX/OSRT/cuBLAS with
CUDA Graph node tracing, analyze only nsys_capture, and correlate LM-head/sampling
kernels with CUDA API launch IDs. Eager has CPU module NVTX hooks, no CUDA Event
hooks; do not use its wall time as an uninstrumented performance result. Outputs
are Windows-readable .nsys-rep plus SQLite/JSON. Preserve existing trace folders.
See benchmarks/_qwen_cuda_nsys_20261116.md; GPU active union is not occupancy,
and long cudaMemcpyAsync CPU duration may be GPU wait rather than transfer time.

`nanovllm/layers/cuda_linear.py` installs per-layer adapters in ModelRunner before
warmup/capture. Config.linear_backend resolves the Qwen3-0.6B default described
above; cuda/cuda_auto and cuda_fp8/cuda_fp8_auto remain opt-in, Qwen3 TP=1 mixed execution only, without other
weight quantization. The decoder's QKV/O and MLP projections are covered; the
tied LM head and FlashAttention are unchanged. Runtime sources live in
layers/csrc and are packaged; standalone benchmark sources retain their historical
config IDs. Runtime BF16/FP16 adds vectorized GEMV and small-M configs 21–23.
FP8 uses fused per-row dynamic BF16 activation quantization and a scaled native
MMA epilogue. Weight copies increase memory consumption. Auto dispatch uses a
frozen device/backend-matched report and falls back on unseen shapes. Graph
and precision tests: tests/test_cuda_linear.py. Benchmark helper:
benchmarks/run_qwen_cuda.sh; report: benchmarks/_qwen_cuda_linear_20261116.md.
Do not infer model quality from random-input GEMM accuracy or token parity.

## Project Structure & Module Organization

All source code lives under `nanovllm/`:

- `engine/` – orchestration and storage: `LLMEngine`, `Scheduler`, `BlockManager`, `Sequence`, `ModelRunner`, host-side KV transfer
- `server.py` – OpenAI-style HTTP/SSE service and in-memory conversation handling
- `layers/` – model operations: attention, linear, layernorm, rotary embedding, sampler
- `models/` – model definitions and `model_type` registry
- `utils/` – weight loading and the global inference context

Root-level `example.py` and `bench.py` are runnable demos; `assets/` holds images; `pyproject.toml` defines packaging and dependencies. `benchmarks/` holds the performance tooling (see `INTERVIEW.md` §10 基准档案): `bench.py` (throughput/latency/SLO/vLLM comparison), `context_concurrency.py` (concurrency × context-length sweep and live KV capacity budget), `kv_fp8_calibrate.py` (real-text FP8 KV calibration and held-out precision comparison), `scheduling_ablation.py` (continuous arrivals, scheduler-policy ablations, and prefix/deferred-free lifecycle counters), `admission_ablation.py` (online accept/defer/reject comparison across admission-off, cache-blind, and cache-aware variants), `compare_quantization.py` (paired INT4/NVFP4 arrival-trace comparison), `prefix_cache_probe.py` / `prefix_cache_verify.py` / `prefix_estimate_check.py` (prefix-hit estimate vs committed reuse, in-process and over HTTP), `profiler.py` (torch.profiler prefill/decode breakdown), plus dev scripts to drive runs from Windows into WSL. Tests live under `tests/`; the pure-Python portions do not need a GPU. `test_nvfp4.py` covers E2M1 packing, both scale layouts, software/native CUDA GEMMs, and empty MoE expert batches; `test_quantization_comparison.py` checks paired workload validation; `test_prefix_feature_lifecycle.py` covers generation invalidation, cache allocation/eviction, swap release, deferred-free commit, and metric reset boundaries; `test_prefix_estimate.py` covers the cross-thread snapshot estimate, version-based staleness, and `pending_free` exclusion; `test_decode_burst_yield.py` covers the burst return contract and the arrival-pressure predicate; `test_tpot_throttle.py` covers per-request metric idempotency and the TPOT throttle deadband/queue gate.

`benchmarks/nsys_profile.py` emits an NVTX-annotated eager inference trace and a per-module
current-stream CUDA Event elapsed-time summary for Nsight Systems. Use TP=1 and focus the UI on
`nsys_capture`. Detailed layer ranges cover Qwen3 and DeepSeek-V2 (MLA, dense/MoE MLP, and
shared experts). For DeepSeek-V2-Lite on a 16 GB GPU, use INT4 group size 64 with streaming
load. The WSL 2024.6.2 CLI produced an NVTX-only report; the upgraded 2026.5.1 CLI
captured CUDA API, kernel, and memory events with ordinary full-process capture on this machine.
An NVTX-triggered capture attempt produced no report, so omit `--capture-range`. Event times can
include stream idle gaps and nested module ranges overlap; use the Nsight kernel timeline for
kernel-active durations.

`benchmarks/compare_ablation.py` compares scheduling/admission reports only when their exact
prompt, sampling, arrival-time, seed, and SLO workload fingerprints match. Admission reports
include joint TTFT/TPOT SLO request and output-token goodput.

`benchmarks/admission_ablation.py --variants` can select a subset of admission variants for
parameter screens that reuse a separately measured control; the default still runs all three.
`benchmarks/supplemental_report.py` aggregates the local supplemental JSON/CSV outputs and
generates the maintained report and figures.

`benchmarks/sharegpt_trace.py` builds de-identified ShareGPT request traces; it uses
reference assistant reply length only to derive a capped `max_tokens` budget and can
generate synthetic arrival timestamps. `benchmarks/derive_trace_variants.py` creates
smoke, calibration, held-out, and prefix variants. The benchmark loaders retain
prompt-only JSONL support and can replay recorded `arrival_time_s` with
`--arrival-mode trace`; synthetic arrivals must not be described as production traffic.

benchmarks/spec_bench.py pairs none/ngram/medusa/eagle decoding on identical tokenized inputs,
checkpoints completed pairs for --resume, supports Qwen chat-template rendering, and records
EOS and output fingerprints. derive_spec_arrival_trace.py and derive_chat_template_trace.py
prepare speculative arrival and prompt traces; spec_weight_smoke.py checks draft checkpoint
loading. admission_ablation.py --speculative none|ngram adds the same decoding toggle to
continuous-arrival admission comparisons. kv_fp8_calibrate.py supports teacher-forced
multi-position precision checks; context_concurrency.py records per-point capacity and
continues after captured CUDA OOMs.
derive_matched_sharegpt_split.py creates disjoint calibration/held-out prompt splits, and
multimodel_smoke.py checks short generation on a selected set of model checkpoints.

`benchmarks/moe_int4_ablation.py` profiles real packed-INT4 MoE layers by input-token count
and reports independent routing, dispatch, gate/up, activation, down, and scatter timings.
`benchmarks/moe_int4_fuse_sweep.py` compares split/fused serving on paired seeded arrival
traces across request rates and output lengths, including token fingerprints and SLO goodput.
`benchmarks/moe_int4_fixed_shape_diag.py` isolates split/fused correctness and performance with
fixed prompts and output traces, and verifies model-input fingerprints and model-call shapes.
`benchmarks/moe_int4_arrival_diag.py` records online request-to-row traces and captures target
logits, decoder/MoE activations, logical MLA KV rows, and frozen-KV row-order replays.
`tests/test_moe_int4_segment.py` checks loop parity for fused/split gate/up and empty expert
segments. Pure INT4 MoE segmentation is opt-in and currently supports TP=1 with BF16/FP16
activations; AWQ, `w_deq`, and other quantization modes use the loop path. CUDA Graph capture
is unverified.

## Build, Test, and Development Commands

```bash
pip install -e .        # install the package in editable mode
python example.py       # end-to-end inference; expects a local Qwen3-0.6B checkpoint
python bench.py         # throughput benchmark (256 sequences)
python benchmarks/bench.py --num-seqs 256            # full metrics: TTFT/TPOT/E2E/p50/p99/SLO
python benchmarks/bench.py --num-seqs 256 --shared-prefix-len 512   # prefix-cache workload
python benchmarks/context_concurrency.py --context-lengths 512,1024,2048,4096 --concurrency 1,2,4,8,16 --repeats 3
python benchmarks/kv_fp8_calibrate.py --calibration-file calibration.jsonl --eval-file heldout.jsonl --output results/kv_eval.json
python benchmarks/profiler.py --num-seqs 64 --max-input-len 512 --max-output-len 64  # prefill/decode breakdown
mkdir -p profiles  # ignored report directory required before nsys starts
PYTHONPATH=. /usr/local/bin/nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,cublas --stats=true --sample=none --cpuctxsw=none -o profiles/qwen3_0.6b_nsys python benchmarks/nsys_profile.py --num-seqs 8 --min-input-len 512 --max-input-len 512 --min-output-len 32 --max-output-len 32 --gpu-event-json profiles/qwen3_0.6b_nsys_gpu_events.json  # WSL2 NVTX + CUDA Event summary
python benchmarks/scheduling_ablation.py --num-seqs 128 --arrival-rate 16 --arrival-mode poisson
python benchmarks/moe_int4_ablation.py ~/huggingface/DeepSeek-V2-Lite --int4-group-size 64 --token-batches 1,8,32
python benchmarks/moe_int4_fixed_shape_diag.py ~/huggingface/DeepSeek-V2-Lite --num-seqs 24 --input-length 64 --seeds 7,17,42 --temperatures 0.6,0.8,1.0 --output-lengths 64,128 --repeats 3 --warmup-repeats 2 --output results/moe_int4_fixed_shape_diag_20261012.json
python benchmarks/compare_quantization.py results/int4.json results/nvfp4.json --variant all_on
python benchmarks/admission_ablation.py --num-seqs 128 --arrival-rate 16 --arrival-mode poisson --shared-prefix-len 512
nanovllm-serve /path/to/model --host 127.0.0.1 --port 8000         # OpenAI-style HTTP/SSE service
```

`benchmarks/bench.py` consumes the timing instrumentation on `Sequence` (`t_submitted`/`t_first_token`/`t_completed`, driver-side only) exported through `LLMEngine.collect_metrics()`; keep those fields when touching the engine. Never name a script `profile.py` under `benchmarks/` — it shadows the stdlib `profile` module that torch's `cProfile` import chain needs.

Requires Python 3.10–3.12 and an NVIDIA GPU; `flash-attn`, `triton`, and NCCL are hard dependencies. There is no build step beyond `pip install`.

## Coding Style & Naming Conventions

- Python, 4-space indentation, type hints on all public signatures.
- Prefer dataclasses with `slots=True` for config-like objects (`Config`, `SamplingParams`, `Context`).
- Name modules after their layer (`engine/`, `layers/`) and classes after the concept (`Scheduler`, `BlockManager`, `Sequence`).
- No linter or formatter is configured; keep new code consistent with surrounding style.

## Testing Guidelines

A pure-Python pytest suite exists under `tests/` (scheduler/block-manager/registration, model metadata, and host KV-transfer logic; no GPU required for those cases). When adding tests:

- Place them under `tests/` with `test_*.py` naming.
- Run with `pytest`.
- Scheduler and block-manager logic is pure Python and testable without a GPU; GPU paths (`ModelRunner`, attention kernels) require CUDA hardware.

## Commit & Pull Request Guidelines

Git history uses short imperative summaries, often `fix(scope): message` (e.g., `fix(model_runner): correct seqlen_k to chunk boundary`). Changes land via pull requests on branches named after the change (e.g., `fix/decoding-positions`). PR descriptions should state the problem, the change, and any benchmark impact; link related issues when present.

## Architecture Notes for Contributors

Offline inference runs as: `LLM.generate` → `LLMEngine` submits sequences → scheduler picks prefill/decode/mixed/spec batches → `ModelRunner` packs tensors → registry-selected model forward → sampler returns tokens. The `Context` singleton in `nanovllm/utils/context.py` passes per-step tensors (slot mappings, block tables) from the runner into attention kernels; keep this interface stable when modifying layers.

The serving path accepts requests continuously through `nanovllm/server.py` and feeds them to one engine worker. Predictive admission estimates prompt work, output length from request caps and a global completion-length EWMA, online Prefill/Decode rates, queue pressure, and projected KV use; it accepts, FIFO-defers, or rejects requests with a bounded wait and a hard queue cap. Prompt work is charged net of the block-aligned prefix the cache can serve: while the worker is idle the manager rebuilds one snapshot per prefix-cache pool (`{hash: (token_ids, block_id, in_free)}`, keyed by `kv_generation`) and walks the candidate's chained block hashes through `_estimate_prefix_cache_match`, comparing token ids as well as hashes; the match names the physical blocks so `_projected_kv_blocks` can promise each block to a single request instead of over-counting a shared prefix. This runs on the event loop and must not read the live tables the engine thread is mutating. Responses report `admission.estimated_cache_hit_tokens` next to `context.prefix_cache_hit_tokens`, and `/health` adds the engine-side cumulative reuse counters; `--no-cache-affinity-admission` disables both scheduler cache affinity and the cache-aware estimate. Managed conversations have a prompt-token budget and cumulative prefix-hit statistics; their text history stays in host memory while KV reuse is supplied by the shared prefix cache. Scheduler policies include mixed/chunked prefill, generation-invalidated lazy Top-W cache-affinity features (`prefix_feature_cache=False` reparses each query for ablation), aging, recompute-aware preemption, and TTFT/TPOT-target-aware quotas. Request-level `tpot_slo_ms` and optional `default_tpot_slo_ms` targets prioritize active decode work using a request-local inter-token EWMA; the policy adapts prefill budgets and reports target attainment, but does not provide a hard real-time guarantee. Bounded multi-step decode (default maximum 4 forwards) runs only on pure non-speculative decode windows and stops when local prefill work is waiting; a burst also ends when the arrival signal is set (`decode_burst_yield_on_arrival`: `LLMEngine.request_decode_burst_yield()` driven by server-side `_sync_decode_burst_pressure()`, or the `set_decode_burst_yield_callback()` probe used by trace replay). The queued-prefill check alone cannot fire in mixed mode, because requests are only submitted between engine steps and a burst starts only with an empty waiting queue, so the arrival signal is the path that matters there (measured 53.6 ms vs 68.3 ms p50 for a request arriving mid-burst, signal on/off, `max_decode_steps=8`). TPOT-aware quotas shrink the per-step prefill budget toward `prefill_reserve_tokens` when an active TPOT target is missed; two guards keep that from starving prefill — `tpot_prefill_throttle_margin` (deadband + proportional ramp instead of a cliff that saturates on any overshoot) and `tpot_throttle_max_waiting` (disable the throttle once the waiting queue is deep, where the bottleneck is prefill throughput, not decode). Without them the throttle self-locked at the quota floor for every target from 10 to 160 ms; with them the same 128-request / 16 req/s / 20 ms-target run went 350 → 1140 tok/s, TTFT p50 44.5 s → 1.6 s, and a target the hardware cannot deliver is still missed (`tpot_starvation` in the ablation JSON reports budget-scale avg/min and starved-step counts). Per-request metrics are recorded through `LLMEngine._record_once()`, which is idempotent per `seq_id`: a sequence that finishes inside a burst is reported by both the burst and the outer batch, and the duplicate records made benchmarks stop early (128 requests → 95 distinct records). SSE delivers tokens accumulated within a burst together. Use `multi_step_decode=False` / `--no-multi-step-decode` and `tpot_aware_scheduling=False` / `--no-tpot-aware-scheduling` for ablations. Idle reusable prefix blocks are reclaimed LRU under allocation pressure or by an optional configured free-block cap; sequence completion defers physical release until the whole postprocess batch is committed. MHA mixed batches, including Mistral rolling and Gemma-2 split, can use bounded lazy CUDA Graph capture up to a configurable token cap. MLA mixed, spec-mixed, oversized, and sparse24 graph forwards fall back to eager; decode graphs cover supported rolling/split inputs, while rolling verification remains eager. `execution_mode="auto"` selects mixed scheduling on one visible GPU and experimental local PD separation on a compatible multi-GPU setup. CPU KV swap supports TP=1 MHA/MLA caches in `auto` and FP8 formats, including Mistral ring and Gemma-2 split pools; TP>1 falls back to recompute, transfers are synchronous, and PD mode retains its separate KV-format/rolling constraints. New combinations need GPU validation.

NVFP4 weights use E2M1 nibbles, E4M3 scales per 16 K values, and one FP32 tensor scale. `nvfp4_backend="software"` is the default Triton BF16-dequantization path; `nvfp4_backend="cublaslt"` dynamically quantizes BF16 activations and calls cuBLASLt block-scaled FP4 MMA on Blackwell (SM100+) with cuBLASLt 12.8+. Native weights store one swizzled scale buffer. MoE routers stay unquantized, routed experts still use the per-expert path, empty expert batches skip the GEMM, and MLA `kv_b` retains a small `w_deq` view for absorbed decode. Compare both NVFP4 backends and INT4 with the same `scheduling_ablation.py` trace and `compare_quantization.py`; DeepSeek-V2-Lite INT4 requires group size 64. The tested native combination is TP=1 with eager execution on RTX 5060 Ti / SM120; CUDA Graph capture and TP>1 are unverified. On one 8-request trace, heuristic-ranked cuBLASLt reached the same joint-SLO goodput as INT4 and higher goodput than software NVFP4, while TTFT remained slower than INT4; see `INTERVIEW.md` §10.3.15.

Pure packed-INT4 MoE experts can opt into `moe_int4_backend="segment"`: routed rows are sorted into expert offset/count segments and Triton runs GEMMs without padded batches, using pointer tables to retain the per-expert weight buffers. `moe_int4_fuse_gate_up=True` combines gate/up work in one launch; false enables the split ablation. The backend currently supports TP=1 BF16/FP16 activations, common supported INT4 group size, and no AWQ or `w_deq`; unsupported combinations retain the existing automatic grouped or per-expert path. CUDA Graph capture is unverified. On DeepSeek-V2-Lite, split and fused outputs matched the loop bitwise across all 26 MoE layers, three seeds, and batch sizes 1/8/32 (234 layer cases). Across 24 paired 8-request traces at 4/8 req/s and 16/64 output tokens, fused throughput had a -2.4% median delta (7 wins, 17 losses); all runs met 10 s TTFT / 600 ms TPOT. A separate 16 req/s, 64-output-token stress point measured +5.4% median but ranged from -20.3% to +31.9% across six pairs; all requests met its 150 ms TPOT target. The SLO targets were not tight enough to expose a goodput knee, and these measurements do not establish a stable fusion gain. See `INTERVIEW.md` §10.3.16.

The 2026-10-11 follow-up used three seeds × three repeats × 24 requests per condition at
16/24/32 req/s. With a 150 ms TPOT target, 64- and 128-token outputs stayed at 100% joint-SLO
attainment through 32 req/s. At 125 ms, 64-token split fell to 192/216 requests at 32 req/s
while 128-token outputs stayed at 216/216. At 100 ms, attainment fell for both lengths;
128-token fused runs reached 143/216 vs split 63/216 at 24 req/s and 141/216 vs 72/216 at
32 req/s. This is workload-specific: 100 ms fused throughput was +3.53% median over 54 pairs
(range -20.31% to +28.38%), with token parity in 42/54 pairs. See `INTERVIEW.md` §10.3.16,
`results/moe_int4_fuse_load150_20261011.json`, `results/moe_int4_fuse_load125_20261011.json`,
and `results/moe_int4_fuse_load100_long_20261011.json`.

The 2026-10-11 100–125 ms refinement used 24 requests per condition, seeds 7/17/42, five paired repeats, 24/32 req/s, and 64/128 output tokens (240 pairs). Fused throughput median delta was +2.24% for 128 tokens (70 wins/50 losses; -17.93% to +18.16%) and +0.81% for 64 tokens (64/56; -28.65% to +29.87%); joint-SLO goodput was not repeat-stable. GPU SM-clock medians were 2775 MHz and decode batch p50/max was 24 rows, but exact model-call shape histograms matched in only 53/240 pairs. Token parity was 212/240; every mismatch was seed 42, request 23, token index 48. Subsequent fixed-shape and online row-order diagnostics are recorded in §§10.3.17–10.3.18; fusion remains experimental. See `benchmarks/_moe_report.md`, `INTERVIEW.md` §10.3.16, and `results/moe_int4_fuse_load105_120_rep5_20261011.json`.

The 2026-10-12 fixed-shape diagnostic used DeepSeek-V2-Lite, TP=1 INT4 group64, eager execution, 24 all-at-once requests with 64-token synthetic prompts, seeds 7/17/42, temperatures 0.6/0.8/1.0, and fixed 64/128-token outputs. Each variant received two full-batch 128-token warmups. Random sampling matched for 432/432 request outputs and greedy matched for 144/144. Performance replayed the split token trace on both backends: all 54 pairs had matching model-input fingerprints and shape histograms. Fused throughput was +3.52% median overall (38/54 wins; -10.05% to +21.60%), with +3.49% at 64 tokens and +3.56% at 128 tokens; per-length ranges crossed zero. The seed-42/request-23/token-48 greedy forced-prefix sentinel had identical full logits and bitwise-identical inputs, routes, gate/up/down projections, and outputs across 26 MoE layers. SLO targets were report-only (60 s TTFT / 10 s TPOT), so these data do not establish serving goodput. See `benchmarks/moe_int4_fixed_shape_diag.py`, `results/moe_int4_fixed_shape_diag_20261012.json`, and `INTERVIEW.md` §10.3.17.

The 2026-10-12 online MLA follow-up replayed the seed-42, 32 req/s, 24-request, 64-in/64-out, 115 ms TPOT trace for five paired repetitions. All five split/fused target calls had different 24-row request orders. With each side's layer-1 hidden states and logical KV fixed, rehydrating the same cache and replaying under the other side's order reproduced all 24 attention outputs bitwise in both directions (10/10 state/order checks); the actual-order replays also matched live outputs. One of five pairs again diverged at request 23 / output token 48. Its current layer-1 query hidden was bitwise equal, but 19,575 of 64,512 logical KV elements differed (max abs 0.125); the attention output differed in 958/2,048 elements (max abs 9.77e-4). The evidence rules out the current decode-row permutation for this captured state and points to an earlier difference in KV formation or cache state; it does not yet locate the first differing cached position or identify a block-mapping bug. Keep fusion experimental and do not use diagnostic wall times as performance results. See `benchmarks/moe_int4_arrival_diag.py`, `benchmarks/_moe_report.md`, `INTERVIEW.md` §10.3.18, and `results/moe_int4_mla_row_permutation_rep5_20261012.json`.

For the HTTP path, `create_app()` and `nanovllm-serve` default to eager mixed prefill/decode
batches; `--mixed-cudagraph` opts into lazy capture. The direct `LLM`/`Config` default remains
unchanged. This serving default follows the 2026-10-09 Qwen3-0.6B/RTX 5060 Ti same-trace result;
other models and GPUs need their own GPU validation.

### 2026-10-15 supplemental screening

The Qwen3-0.6B ShareGPT scheduling screen used five paired 500-request traces at 8 req/s. The all-on bundle averaged -6.50% output throughput (paired-bootstrap 95% CI -7.99% to -4.52%) and -59.09% joint-SLO output goodput (-68.30% to -49.89%); both metrics fell in all five seeds. A seed-0 one-factor screen improved when TPOT-aware scheduling was disabled, but it was a single screening run and remained below the separate baseline run. Do not treat these measurements as support for enabling the bundle by default.

Admission-off had higher joint-SLO output goodput at the tested full-trace operating points (24 req/s with 0/20/50% shared-prefix requests, and 32/48/64/96 req/s without shared prefixes). The supplement added 12 target/work points at TPOT≤20 ms, 24 soft-pressure/wait points at TPOT≤80 ms, three calibration seeds for a selected candidate across TPOT 20/40/80/120 ms, and two held-out seeds on 100 prompts at 384 req/s. At TPOT≤80 ms the selected cache-aware candidate had lower calibration goodput and a higher reject rate than admission-off; at TPOT≤20 ms it produced goodput where admission-off produced zero. The held-out trace used a different size and offered rate. There is no product rejection budget, so no default threshold is recommended. The synthetic Qwen KV matrix measured 425 BF16 blocks / 108,800 token slots and 835 FP8 E4M3 blocks / 213,760 slots. The default cost policy recomputed its two observed preemptions; a forced swap-path comparison recorded 470 vs 511 output token/s across three repeats (-8.1%), which is a path diagnostic and not a serving gain. FP8 held-out logit checks cover 16 prompts at the first decode position only.
The 20%/50% shared-prefix variants used a synthetic 256-token common template, not a real system prompt.

INT4 MoE segment split/fused matched the loop bitwise for 26 layers × 2 seeds × 5 row counts (260 cases per variant). The fused microbenchmark was 11.90×–42.95× faster across 1/4/16/64/128 rows, but this is kernel-local. In 36 online ShareGPT pairs, fused throughput had +0.91% median delta (-14.94% to +23.07%; 20 wins/16 losses), and every-request output parity held in 19/36 pairs. A repeated 8 req/s, 128-output, 100 ms TPOT condition (5 seeds × 5 repeats) measured +0.51% median throughput delta (-12.87% to +16.46%; 14 wins/11 losses) and full output parity in 14/25 pairs; keep fusion experimental. `MLAAttention._assemble` now derives each query end from its per-row start plus query length. The post-change seed-42 replay still had one mismatching request out of 16, so this does not close the online divergence. Speculative screening covered n-gram only; trained EAGLE/Medusa weights were unavailable.

See [`benchmarks/_supplemental_report.md`](benchmarks/_supplemental_report.md), [`benchmarks/_supplemental_figures/`](benchmarks/_supplemental_figures/), and `python -m benchmarks.supplemental_report` for generated tables and charts. The raw results are ignored under `results/supplemental/` and are local experiment artifacts.

### 2026-10-16 supplemental follow-up

The current-tree scheduling rerun used 500 Qwen3-0.6B requests at synthetic Poisson 8 req/s,
five paired seeds, TTFT 2 s / TPOT 80 ms. without_tpot_aware versus all_on improved output
throughput by +113.5 token/s (95% CI +61.8 to +165.2) and joint-SLO output goodput by
+213.9 token/s (+47.8 to +405.7); against baseline, the goodput CI crossed zero. Keep the
2026-10-15 negative result as a separate historical run. Matched admission used 250-request
calibration and held-out traces at 96 req/s, with 3/2 seeds; admission-off rejected none and
had higher goodput in every pair. No product rejection budget is defined.

The 2026-10-16 FP8 teacher-forced evaluation covers 250 calibration and 250 held-out prompts at
decode positions 1/8/32/64/128. The BF16/FP8 capacity matrix has 24 points per dtype and no
failures; pools hold 108,800 and 211,712 token slots, but the larger FP8 pool did not improve
joint-SLO attainment at the tested long-context points. N-gram online A/B gained 2.5–3.1%
median throughput on repeated synthetic prompts but showed no stable gain on chat-template
ShareGPT. Gamma2 Medusa smoke failed on a checkpoint/head-count mismatch while EAGLE loaded.
The gamma4 Medusa/EAGLE screen had output parity in 0/3 seeds for both modes; EAGLE was slower
in all three. Do not treat the Medusa positive median as quality-preserving speedup.

INT4 multi-model smoke completed 16/16 requests on Qwen2.5-7B, Mistral-7B-v0.1, and
Llama-3.1-8B; concurrency/context sweeps remain outstanding. After the MLA query-range change,
five paired MoE diagnostics at seed42 / 8 req/s / 16 requests showed 5/5 token parity and 10/10
row-permutation replay checks. The historical seed42 / 32 req/s / 24-request divergence condition
was not rerun; its first differing KV position remains unknown. All 2026-10-16 results are from HEAD a5a2e0e71d91753e2384285b079aaaf0535b84c6 with
git_worktree_dirty=true; raw JSON/CSV are local ignored artifacts under results/supplemental/.
See INTERVIEW.md §10.3.20 and benchmarks/_supplemental_report_20261016.md.
supplemental_report.py appends tracked 2026 dated follow-up Markdown files after the generated
2026-10-15 body, so regeneration retains this historical update.

### Standalone CUDA GEMM operator

`benchmarks/cuda_gemm.cu` and `cuda_gemm_bench.py` implement and validate independent
FP16/BF16 row-major GEMM (FP32 accumulation); they do not change engine dispatch.
The SM80+ fast path uses MMA, ldmatrix, XOR-swizzled shared memory, and cp.async.
N/K must be divisible by 8 for the fast path; other sizes use a scalar fallback.
Inputs must be contiguous and 16-byte aligned. Kernels use the current CUDA stream;
CUDA Graph capture and boundary cases are checked by `--check-only`.
The baseline calls cuBLAS GemmEx directly, with reduced-precision reduction disabled.
Use `python -m benchmarks.cuda_gemm_bench` to tune; freeze configs with
`--tuning-report` for independent validation seeds. `cuda_gemm_report.py` checks
report compatibility and generates `benchmarks/_cuda_gemm_report_20261115.md`.
Keep its warm-cache operator results separate from serving throughput and the
historical 2026-10-05 FP16 experiment. Tested GPU: RTX 5060 Ti / SM120 only.

`cuda_gemm_profile.py` adds controlled K/grid/residency/stage/layout experiments
and Nsight Systems SQLite analysis; `cuda_gemm_profile_report.py` renders the
dated attribution report. CUDA `gemm_out_smem`, `kernel_info`,
`gemm_diagnostic_out`, `diagnostic_info`, and `launch_empty` are diagnostic APIs.
They preserve the 21 normal config IDs/default address layout. Reserved unused
shared memory limits residency while retaining the same kernel instructions;
the alternative A swizzle remains diagnostic. Occupancy API values are static
limits, not measured achieved occupancy. Nsight Compute 2025.1.1 failed with
`LibraryNotLoaded`; the same-day recovery selected the existing Linux CLI
2026.3.1 bundled under `/mnt/d/NsightCompute/target/linux-desktop-glibc_2_11_3-x64/`.
`cuda_gemm_profile.py` now selects the newest discovered Linux ncu, with
`--ncu` / `NCU_BIN` overrides. `ncu_diagnose.py` isolates failures with a standalone
CUDA probe (`NCU_PROBE_CXX` overrides this machine's GCC 13 default). Versioned
report directories retain failed historical attempts; successful runs export raw
CSV and command/status JSON. See `benchmarks/_ncu_recovery_20261115.md`.
Nsight Systems 2026.5.1 kernel tracing succeeded. Never label
the algebraic K/MN shape-effect decomposition as hardware stall percentages.

`cuda_gemm_counter_study.py` reruns randomized paired graph timings and serial
Nsight Compute captures for both backends, dtypes and sizes; run it through
`bash benchmarks/run_cuda_gemm.sh --counter-study`. Primary counters have three
independent captures; fixed-cfg10 rectangular K/grid/wave controls have one per
point and separately repeated unprofiled timings. `cuda_gemm_counter_report.py`
renders `benchmarks/_cuda_gemm_counter_20261116.md`. Trace/counter `--shape M N K`
overrides `--size` and validates outputs before profiling. Detailed memory tables
are included. Preserve raw CSV units: duration units can change with shape, and
warp-state ratios are cycles/instruction, not percentages of runtime. Lower
occupancy alone is not a diagnosis: FP16 cuBLAS has lower achieved occupancy but
higher Tensor activity at 1024. Keep the new counter evidence separate from the
historical 2026-11-15 attribution and recovery datasets.

### Standalone native FP8 GEMM

`benchmarks/cuda_fp8_gemm.cu` / `cuda_fp8_gemm_bench.py` implement native SM120a
warp FP8 MMA with FP32 accumulation and BF16 output. The contract is contiguous
`A[M,K]`, prepacked `W[N,K]`, `C[M,N]`, with finite scalar alpha, 16-byte-aligned
inputs and K divisible by 16. Custom kernels handle partial M/N/K tiles, empty
outputs and K=0. Other K alignment is rejected. E4M3 is benchmarked; native E5M2
correctness is tested separately without a cuBLASLt E5M2 performance claim.
The 28 configs cover tile/warp/stage variants and optional M-group-8 CTA raster
for L2 weight reuse. SASS must contain FP8 `QMMA`, `LDSM` and `LDGSTS`; no BF16
dequantization GEMM or engine integration is involved.

The cuBLASLt plan owns layouts/algorithms/workspace (up to 32 MiB), uses
COMPUTE_32F / FAST_ACCUM=0, and tunes a heuristic subset outside validation.
Algorithm attributes are frozen and checked across seeds. Both paths receive
the same packed inputs and output type. Quantization/transpose/allocation/setup
are excluded; report decoded-input arithmetic error separately from FP8
quantization error. Do not claim end-model quality or serving speedup.
Use the WSL helper `--fp8`, `--fp8-pytest`, `--fp8-ablation`, `--fp8-profile`;
`cuda_fp8_gemm_report.py` renders the dated report. `--tuning-report` freezes
both selectors; `--shapes` accepts semicolon or plus-separated M,N,K triples.
Compile explicitly for sm_120a with CUDA 12.8+ and the helper's GCC 13 default.
See `benchmarks/_cuda_fp8_gemm_20261116.md`; preserve initial and grouped runs
as separate observations. Original FP16/BF16 files and 21 config IDs are unchanged.

## Documentation maintenance

When changing a feature, update the relevant Markdown in the same change: `README.md` for user-facing behavior and limits, `AGENTS.md` / `CLAUDE.md` for development and architecture guidance, `INTERVIEW.md` for the maintained implementation/roadmap/benchmark record, and experiment reports when their evidence or status changes. Keep dated benchmark measurements as historical observations; do not rewrite them to imply the new code was measured. State supported combinations, fallbacks, and unverified paths next to feature claims.
