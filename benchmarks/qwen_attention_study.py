"""Compare Qwen3 KV-head layouts with matched inference and Nsight traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing, nullcontext
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm import LLM


WORKLOADS = ((1, 128), (8, 128), (1, 4096), (8, 4096))
LAYOUTS = {16: "mha16", 8: "gqa8", 4: "gqa4", 1: "mqa1"}
OUTPUT_TOKENS = 32


def run_case(engine: LLM, batch: int, length: int, seed: int) -> dict:
    import torch
    from nanovllm import SamplingParams

    rng = random.Random(f"{seed}/{batch}/{length}")
    prompts = [[rng.randrange(100, 10000) for _ in range(length)] for _ in range(batch)]
    params = SamplingParams(temperature=0.8, max_tokens=OUTPUT_TOKENS, ignore_eos=True)
    torch.manual_seed(seed)
    start = perf_counter()
    results = engine.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    wall_s = perf_counter() - start
    metrics = engine.collect_metrics()
    outputs = [result["token_ids"] for result in results]
    if (len(outputs) != batch or any(len(tokens) != OUTPUT_TOKENS for tokens in outputs)
            or metrics["num_preemptions"]):
        raise RuntimeError("Workload did not complete at the requested shape without preemption")
    requests = metrics["per_request"]
    stats = metrics["step_stats"]
    return dict(wall_s=wall_s, tokens_per_s=batch * OUTPUT_TOKENS / wall_s,
                ttft_ms=median((r["t_first_token"] - r["t_submitted"]) * 1000 for r in requests),
                tpot_ms=median((r["t_completed"] - r["t_first_token"]) * 1000 / (OUTPUT_TOKENS - 1)
                               for r in requests),
                prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                output_sha256=hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),
                prefill_tokens=stats["prefill_tokens"], decode_tokens=stats["decode_tokens"],
                decode_replays=metrics["decode_cudagraph"]["replays"],
                mixed_replays=metrics["mixed_cudagraph"]["replays"])


def score_text(engine: LLM, corpus: Path) -> dict:
    import torch
    import torch.nn.functional as F

    from benchmarks.kv_fp8_calibrate import load_texts, tokenize_texts
    from nanovllm.engine.sequence import Sequence
    from nanovllm.utils.context import get_context, reset_context

    texts, _ = load_texts(str(corpus), [], 64)
    rows = [row for row in tokenize_texts(engine.tokenizer, texts, 128) if len(row) >= 16][:32]
    if len(rows) != 32:
        raise ValueError("Need 32 texts of at least 16 tokens")
    runner = engine.model_runner
    seqs = [Sequence(row) for row in rows]
    for seq in seqs:
        seq.num_scheduled_tokens = len(seq)
    indices, targets, offset = [], [], 0
    for row in rows:
        indices.extend(range(offset, offset + len(row) - 1))
        targets.extend(row[1:])
        offset += len(row)
    with torch.inference_mode():
        try:
            inputs, positions = runner.prepare_prefill(seqs)
            # Use the normal prefill attention and suppress writes to the live KV pool.
            get_context().slot_mapping = torch.full_like(inputs, -1, dtype=torch.int32)
            hidden = runner.model(inputs, positions)
        finally:
            reset_context()
        hidden = hidden[torch.tensor(indices, device=hidden.device)]
        target = torch.tensor(targets, device=hidden.device)
        losses, top1 = [], []
        for h, y in zip(hidden.split(64), target.split(64)):
            logits = F.linear(h, runner.model.lm_head.weight).float()
            losses.extend(F.cross_entropy(logits, y, reduction="none").tolist())
            top1.extend(logits.argmax(-1).tolist())
    if not all(map(math.isfinite, losses)):
        raise RuntimeError("Non-finite teacher-forced loss")
    mean_nll = sum(losses) / len(losses)
    return dict(prompts=len(rows), tokens=len(losses), mean_nll=mean_nll, perplexity=math.exp(mean_nll),
                text_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(), nll=losses, top1=top1)


def analyze_trace(path: Path) -> dict:
    from benchmarks.decode_overlap_study import _owning_range
    from benchmarks.qwen_cuda_nsys import category

    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        capture = db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='nsys_capture'").fetchone()
        cases = db.execute("""SELECT text,start,end,globalTid FROM NVTX_EVENTS
            WHERE start>=? AND end<=? AND text LIKE 'attention_case/%'""", capture).fetchall()
        names = dict(db.execute("SELECT id,value FROM StringIds"))
        results = {}
        for label, start, stop, thread in cases:
            ranges = db.execute("""SELECT start,end,text FROM NVTX_EVENTS
                WHERE text LIKE 'ModelRunner.run_model [%]' AND start>=? AND end<=? AND globalTid=?
                ORDER BY start""", (start, stop, thread)).fetchall()
            windows = [(a, b) for a, b, _ in ranges]
            phases = [text.split("[", 1)[1].removesuffix("]") for _, _, text in ranges]
            launches = {}
            for a, b, correlation in db.execute("""SELECT start,end,correlationId
                FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=? AND globalTid=?""",
                                                (start, stop, thread)):
                owner = _owning_range(windows, a, b)
                if owner is not None:
                    launches[correlation] = phases[owner]
            by_phase = defaultdict(Counter)
            flash_names = defaultdict(Counter)
            for a, b, correlation, name_id in db.execute("""SELECT start,end,correlationId,demangledName
                FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?""", (start, stop)):
                phase = launches.get(correlation)
                if phase is None:
                    continue
                name = names[name_id]
                bucket = category(name)
                by_phase[phase][bucket] += b - a
                if bucket == "FlashAttention":
                    flash_names[phase][name] += 1
            counts = Counter(phases)
            if counts["decode"] == 0 or not flash_names["decode"]:
                raise ValueError(f"Missing pure decode FlashAttention kernels in {label}")
            results[label.split("/")[1]] = {
                phase: dict(forwards=count,
                            gpu_kernel_us_per_forward={bucket: value / (count * 1000)
                                                       for bucket, value in by_phase[phase].items()},
                            flash_kernels=dict(flash_names[phase]))
                for phase, count in counts.items()
            }
        return results


def summarize(directory: Path) -> None:
    variants = {}
    for heads, layout in LAYOUTS.items():
        first = json.loads((directory / f"{layout}.first.timing.json").read_text(encoding="utf-8"))
        reverse = json.loads((directory / f"{layout}.reverse.timing.json").read_text(encoding="utf-8"))
        capture = json.loads((directory / f"{layout}.capture.json").read_text(encoding="utf-8"))
        quality = first.pop("quality")
        timing = {(row["case"], row["seed"]): row
                  for row in first.pop("workloads") + reverse["workloads"]}
        if first["kv_heads"] != heads:
            raise ValueError(f"Incorrect KV count for {layout}")
        variants[layout] = (first, timing, quality, capture)
    _, control, reference, control_capture = variants["gqa8"]
    results = {}
    for layout, (metadata, timing, quality, capture) in variants.items():
        if timing.keys() != control.keys() or quality["text_sha256"] != reference["text_sha256"]:
            raise ValueError(f"Workloads do not match for {layout}")
        for key, row in timing.items():
            for field in ("prompt_sha256", "decode_replays", "mixed_replays", "prefill_tokens", "decode_tokens"):
                if row[field] != control[key][field]:
                    raise ValueError(f"Shape or workload changed for {layout}/{key}/{field}")
        if len(quality["top1"]) != len(reference["top1"]):
            raise ValueError("Mismatched teacher-forced token counts")
        trace = analyze_trace(directory / f"{layout}.sqlite")
        for candidate, baseline in zip(capture["workloads"], control_capture["workloads"], strict=True):
            if (candidate["prompt_sha256"] != baseline["prompt_sha256"]
                    or candidate["decode_replays"] != trace[candidate["case"]]["decode"]["forwards"]):
                raise ValueError(f"Mismatched trace workload for {layout}/{candidate['case']}")
        accuracy = {key: value for key, value in quality.items() if key not in ("nll", "top1")}
        accuracy["top1_agreement_pct_vs_gqa8"] = (
            100 * sum(a == b for a, b in zip(quality["top1"], reference["top1"])) / quality["tokens"])
        accuracy["mean_abs_nll_delta_vs_gqa8"] = (
            sum(abs(a - b) for a, b in zip(quality["nll"], reference["nll"])) / quality["tokens"])
        cases = {}
        for batch, length in WORKLOADS:
            case = f"B{batch}_L{length}"
            metrics = {}
            keys = [key for key in control if key[0] == case]
            for metric in ("tokens_per_s", "ttft_ms", "tpot_ms", "wall_s"):
                changes = [100 * (timing[key][metric] / control[key][metric] - 1) for key in keys]
                metrics[metric] = dict(median=median(timing[key][metric] for key in keys),
                                       paired_delta_pct=median(changes), paired_min_pct=min(changes),
                                       paired_max_pct=max(changes))
            cases[case] = dict(pairs=len(keys), **metrics)
        results[layout] = dict(**metadata, bytes_per_token=metadata["cache_bytes"] // metadata["cache_token_slots"],
                               quality=accuracy, timing=cases, trace=trace)
    (directory / "comparison.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 99, 123])
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output)
        return
    if args.model is None:
        parser.error("--model is required for inference")

    import torch
    from benchmarks.nsys_profile import _wrap_method, nvtx_range
    from nanovllm import LLM

    engine = LLM(args.model, linear_backend="torch", execution_mode="mixed", max_num_seqs=8,
                 max_model_len=4352, max_num_batched_tokens=8192, gpu_memory_utilization=0.72,
                 slo_aware_scheduling=False, latency_aware_scheduling=False, tpot_aware_scheduling=False)
    runner, config = engine.model_runner, engine.config
    scope = nvtx_range if args.annotate else lambda _: nullcontext()
    try:
        hf, cache = config.hf_config, runner.kv_cache
        if (hf.model_type, hf.num_attention_heads, hf.head_dim, hf.num_hidden_layers) != ("qwen3", 16, 128, 28):
            raise ValueError("Expected the Qwen3-0.6B query-head geometry")
        layout = LAYOUTS[hf.num_key_value_heads]
        qkv_width = runner.model.model.layers[0].self_attn.qkv_proj.weight.size(0)
        if (cache.shape[-2:] != (hf.num_key_value_heads, hf.head_dim)
                or qkv_width != (hf.num_attention_heads + 2 * hf.num_key_value_heads) * hf.head_dim):
            raise RuntimeError("The projection or KV cache does not have the requested head layout")
        metadata = dict(layout=layout, q_heads=hf.num_attention_heads, kv_heads=hf.num_key_value_heads,
                        head_dim=hf.head_dim, layers=hf.num_hidden_layers, cache_dtype=str(cache.dtype),
                        cache_blocks=config.num_kvcache_blocks, cache_bytes=cache.nbytes,
                        cache_token_slots=config.num_kvcache_blocks * config.kvcache_block_size,
                        qkv_width=qkv_width,
                        device=torch.cuda.get_device_name(), torch=torch.__version__,
                        cuda=torch.version.cuda, linear_backend=config.linear_backend)
        for batch, length in WORKLOADS:
            run_case(engine, batch, length, 11)
        if args.annotate:
            _wrap_method(runner, "run_model", "ModelRunner.run_model", 2)
        results = []
        with scope("nsys_capture"):
            for seed in args.seeds:
                for batch, length in WORKLOADS:
                    case = f"B{batch}_L{length}"
                    with scope(f"attention_case/{case}/{seed}"):
                        row = dict(case=case, seed=seed, **run_case(engine, batch, length, seed))
                    results.append(row)
                    print(layout, json.dumps(row), flush=True)
        metadata["workloads"] = results
        if args.corpus:
            metadata["quality"] = score_text(engine, args.corpus)
        args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
