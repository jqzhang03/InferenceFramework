"""Compare DeepSeek-V2-Lite INT4, NVFP4 and MXFP4 on a single GPU."""
from __future__ import annotations

import argparse
import gc
from itertools import accumulate
import json
import math
from pathlib import Path
import random
from statistics import median
from time import perf_counter

import torch
import torch.nn.functional as F

from benchmarks.deepseek_w4a16 import environment, fingerprint, write_json


def select_texts(path: Path, tokenizer, count: int, length: int, seed: int) -> list:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    random.Random(seed).shuffle(records)
    selected = []
    seen = set()
    for row in records:
        tokens = tuple(tokenizer.encode(row["prompt"])[:length])
        if len(tokens) != length or tokens in seen:
            continue
        seen.add(tokens)
        selected.append((row["request_id"], list(tokens)))
        if len(selected) == count:
            return selected
    raise ValueError(f"need {count} distinct texts of at least {length} tokens in {path}")


def prepare_accuracy(prompts: list[list[int]]) -> tuple:
    ids = torch.tensor([token for prompt in prompts for token in prompt], device="cuda")
    positions = torch.tensor(
        [pos for prompt in prompts for pos in range(len(prompt))], device="cuda")
    lengths = [len(prompt) for prompt in prompts]
    cu = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device="cuda")
    return ids, positions, cu, max(lengths)


def eval_logits(model, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    # Exclude each text's final token and bound the logits workspace.
    hidden = hidden[positions.roll(-1) != 0]
    return torch.cat([model.compute_logits(chunk).cpu() for chunk in hidden.split(128)])


def reference_logits(path: Path, hf, inputs: tuple) -> torch.Tensor:
    from nanovllm.layers.moe import MoE
    from nanovllm.layers.rotary_embedding import get_rope
    from nanovllm.models.registry import get_model_class
    from nanovllm.utils.context import reset_context, set_context
    from nanovllm.utils.loader import load_model

    ids, positions, cu, longest = inputs
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf.dtype)
    get_rope.cache_clear()
    try:
        with torch.device("meta"):
            model = get_model_class(hf.model_type)(hf)
        hidden, residual = None, None

        def evaluate_chunk(module, name: str) -> None:
            nonlocal hidden, residual
            if name == "model.embed_tokens":
                hidden = module(ids)
            elif name.startswith("model.layers."):
                with torch.device("cuda"):
                    module.self_attn.rotary_emb.build_cache()
                hidden, residual = module(positions, hidden, residual)
                if isinstance(module.mlp, MoE):
                    module.mlp._gup_t = module.mlp._dn_t = None
            else:
                return
            module.to_empty(device="meta")

        set_context(True, cu, cu, longest, longest)
        try:
            load_model(model, str(path), streaming=True, chunk_hook=evaluate_chunk)
            hidden, _ = model.model.norm(hidden, residual)
        finally:
            reset_context()
        return eval_logits(model, hidden, positions)
    finally:
        get_rope.cache_clear()
        torch.set_default_dtype(original_dtype)


def quantized_logits(model, inputs: tuple) -> torch.Tensor:
    from nanovllm.utils.context import reset_context, set_context

    ids, positions, cu, longest = inputs
    set_context(True, cu, cu, longest, longest)
    try:
        hidden = model(ids, positions)
    finally:
        reset_context()
    return eval_logits(model, hidden, positions)


def accuracy(logits: torch.Tensor, baseline: torch.Tensor, labels: torch.Tensor,
             lengths: list[int]) -> dict:
    nll, kl, top1 = [], [], []
    for actual, expected, target in zip(logits.split(128), baseline.split(128), labels.split(128)):
        a = actual.cuda().float().log_softmax(dim=-1)
        b = expected.cuda().float().log_softmax(dim=-1)
        nll.extend((-a.gather(1, target[:, None]).squeeze(1)).tolist())
        kl.extend((b.exp() * (b - a)).sum(dim=-1).tolist())
        top1.extend((a.argmax(-1) == b.argmax(-1)).tolist())
    by_text = [sum(nll[end - length:end]) / length
               for end, length in zip(accumulate(lengths), lengths)]
    return dict(tokens=len(nll), mean_nll=sum(nll) / len(nll),
                perplexity=math.exp(sum(nll) / len(nll)), kl_bf16=sum(kl) / len(kl),
                top1_bf16=sum(top1) / len(top1), nll_by_text=by_text)


def weight_bytes(model) -> int:
    tensors = [*model.parameters(), *(tensor for name, tensor in model.named_buffers()
                                     if name.rsplit(".", 1)[-1].startswith("w_"))]
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    return sum(storages.values())


class Greedy(torch.nn.Module):
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)


def timed_batch(engine, prompts: list[list[int]], output_tokens: int) -> dict:
    from nanovllm import SamplingParams

    engine.reset_benchmark_metrics()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    params = SamplingParams(temperature=1.0, max_tokens=output_tokens, ignore_eos=True)
    for prompt in prompts:
        engine.add_request(prompt, params)
    outputs = {}
    steps = []
    while not engine.is_finished():
        before = perf_counter()
        completed, kind, prefill, decode = engine.step()
        steps.append(dict(kind=kind, prefill=prefill, decode=decode,
                          ms=(perf_counter() - before) * 1000))
        outputs.update(completed)
    torch.cuda.synchronize()
    seconds = perf_counter() - started
    tokens = [outputs[key] for key in sorted(outputs)]
    metrics = engine.collect_metrics()
    if (len(tokens) != len(prompts) or any(len(row) != output_tokens for row in tokens)
            or metrics["prefix_cache_hit_tokens"] or metrics["num_preemptions"]):
        raise RuntimeError("unexpected request counts, prefix hits or preemptions in timed batch")
    requests = metrics["per_request"]
    return dict(seconds=seconds, tok_per_s=len(tokens) * output_tokens / seconds,
                ttft_ms=median((r["t_first_token"] - r["t_submitted"]) * 1000 for r in requests),
                tpot_ms=median((r["t_completed"] - r["t_first_token"]) * 1000 / (output_tokens - 1)
                               for r in requests),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                decode_graph_replays=engine.model_runner.decode_cudagraph_replay_count,
                steps=steps, output_token_ids=tokens)


def run_quantized(args: argparse.Namespace, variant: str, inputs: tuple, reference: torch.Tensor,
                  accuracy_lengths: list[int], perf_prompts: list[list[int]]) -> dict:
    from nanovllm import LLM

    options = {"int4": dict(int4_group_size=64, int4_dense_path=False,
                            int4_kernel_backend="cuda", moe_int4_backend="compact"),
               "nvfp4": dict(nvfp4_backend="cuda"), "mxfp4": {}}[variant]
    before = perf_counter()
    engine = LLM(str(args.model), quantization=variant, streaming_load=True,
                 enforce_eager=not args.graph, max_num_seqs=max(args.batches),
                 max_model_len=args.length + args.output_tokens,
                 max_num_batched_tokens=max(max(args.batches) * args.length,
                                            args.length + args.output_tokens),
                 execution_mode="mixed", multi_step_decode=False, kv_swap=False,
                 mixed_cudagraph=False, mla_decode_backend="triton",
                 gpu_memory_utilization=args.gpu_memory_utilization, **options)
    model = engine.model_runner.model
    try:
        torch.cuda.synchronize()
        seconds = perf_counter() - before
        if any(not hasattr(layer.mlp.gate, "weight") for layer in model.model.layers[1:]):
            raise RuntimeError("the comparison requires unquantized routers")
        logits = quantized_logits(model, inputs)
        ids, positions, _, _ = inputs
        score = accuracy(logits, reference, ids.roll(-1)[positions.roll(-1) != 0], accuracy_lengths)
        del logits
        engine.model_runner.sampler = Greedy()
        timings = {}
        for batch in args.batches:
            timed_batch(engine, perf_prompts[:batch], args.output_tokens)
            timings[str(batch)] = [
                timed_batch(engine, perf_prompts[:batch], args.output_tokens)
                for _ in range(args.repeats)
            ]
        return dict(settings=options, load_warmup_capture_s=seconds,
                    model_weight_bytes=weight_bytes(model), kv_blocks=engine.config.num_kvcache_blocks,
                    accuracy=score, timings=timings)
    finally:
        engine.exit()


@torch.inference_mode()
def main() -> None:
    from transformers import AutoConfig, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("~/huggingface/DeepSeek-V2-Lite"))
    p.add_argument("--data", type=Path,
                   default=Path("results/supplemental/sharegpt_500_heldout250_20261016.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variants", nargs="+", choices=("int4", "nvfp4", "mxfp4"),
                   default=["int4", "nvfp4", "mxfp4"])
    p.add_argument("--texts", type=int, default=16)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    p.add_argument("--output-tokens", type=int, default=16)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--gpu-memory-utilization", type=float, default=.85)
    p.add_argument("--reference-cache", type=Path, help="Temporary BF16 logits for resuming")
    p.add_argument("--resume", action="store_true",
                   help="Complete missing variants in an existing matched run")
    args = p.parse_args()
    if min(args.batches) < 1 or args.texts < 1 or args.repeats < 1 or args.output_tokens < 2:
        p.error("texts, batches and repeats must be positive, with at least two output tokens")
    if args.length < 2 or args.length + args.output_tokens > 256:
        p.error("require at least two input tokens and at most 256 total tokens for cold prefixes")
    if not 0 < args.gpu_memory_utilization < 1:
        p.error("--gpu-memory-utilization must be between zero and one")
    if args.reference_cache is not None and args.reference_cache.resolve() == args.output.resolve():
        p.error("the reference cache must use a different path from the JSON result")
    if args.output.exists() and not args.resume:
        p.error(f"choose a new output path; {args.output} exists")
    if args.resume and (not args.output.is_file() or args.reference_cache is None
                        or not args.reference_cache.is_file()):
        p.error("--resume requires the existing result and its --reference-cache")
    if not args.resume and args.reference_cache is not None and args.reference_cache.exists():
        p.error(f"choose a new reference cache path; {args.reference_cache} exists")
    previous = json.loads(args.output.read_text(encoding="utf-8")) if args.resume else None
    args.model = args.model.expanduser()
    hf = AutoConfig.from_pretrained(args.model)
    shape = (hf.model_type, hf.hidden_size, hf.num_hidden_layers, getattr(hf, "n_routed_experts", None))
    if shape != ("deepseek_v2", 2048, 27, 64) or hf.dtype != torch.bfloat16 or hf.tie_word_embeddings:
        raise ValueError("this benchmark expects the untied BF16 DeepSeek-V2-Lite checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = select_texts(args.data, tokenizer, args.texts + max(args.batches), args.length, args.seed)
    prompts = [prompt for _, prompt in texts[:args.texts]]
    perf_prompts = [prompt for _, prompt in texts[args.texts:]]
    inputs = prepare_accuracy(prompts)
    lengths = [len(prompt) - 1 for prompt in prompts]
    ids, positions, _, _ = inputs
    labels = ids.roll(-1)[positions.roll(-1) != 0]
    if previous is None:
        print("Running layer-streamed BF16 teacher-forced reference", flush=True)
        baseline = reference_logits(args.model, hf, inputs)
        if args.reference_cache is not None:
            args.reference_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(baseline, args.reference_cache)
    else:
        baseline = torch.load(args.reference_cache, map_location="cpu", weights_only=True)
    gc.collect()
    torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = dict(environment=environment(), model=args.model.name, text_source=str(args.data),
                seed=args.seed, input_length=args.length, output_length=args.output_tokens,
                batches=args.batches, repeats=args.repeats, graph=args.graph,
                gpu_memory_utilization=args.gpu_memory_utilization,
                reference_sha256=fingerprint(baseline),
                texts=[dict(request_id=key, token_sha256=fingerprint(torch.tensor(prompt)))
                       for key, prompt in texts],
                accuracy_text_count=args.texts, bf16=accuracy(baseline, baseline, labels, lengths),
                variants={})
    if previous is not None:
        for key, value in data.items():
            if key == "variants":
                continue
            if json.loads(json.dumps(value)) != previous[key]:
                raise ValueError(f"cannot resume with different {key}")
        data["variants"] = previous["variants"]
    else:
        write_json(args.output, data)
    for variant in args.variants:
        if variant in data["variants"]:
            continue
        print(f"Running {variant}: teacher-forced accuracy and warm end-to-end batches", flush=True)
        data["variants"][variant] = run_quantized(args, variant, inputs, baseline, lengths, perf_prompts)
        gc.collect()
        torch.cuda.empty_cache()
        write_json(args.output, data)
        print(f"Saved {variant} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
