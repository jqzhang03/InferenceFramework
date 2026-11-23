"""Measure DeepSeek-V2-Lite serving steps and isolate expert dispatch costs."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import wraps
from itertools import permutations
from pathlib import Path
import random
from statistics import median
from time import perf_counter
from types import MethodType

import torch

from benchmarks.deepseek_w4a16 import ModelTrace, environment, error_metrics, memory, write_json
from benchmarks.nsys_profile import nvtx_range
from nanovllm import LLM, SamplingParams
from nanovllm.layers.cuda_int4 import set_cuda_int4_expert_fusion


WORKLOADS = {"b1_ctx64": (1, 64), "b4_ctx64": (4, 64),
             "b8_ctx64": (8, 64), "b4_ctx512": (4, 512),
             "b16_ctx64": (16, 64), "b32_ctx64": (32, 64)}
DEFAULT_WORKLOADS = ("b1_ctx64", "b4_ctx64", "b8_ctx64", "b4_ctx512")
MODES = ("triton_loop", "cuda_loop", "cuda_compact", "cuda_gpu_compact")


def reuse_index(moe, x: torch.Tensor) -> torch.Tensor:
    """Keep all 64 iterations while materializing each mask's indices once."""
    probs, indices = moe._route(x)
    slots, probs = indices.flatten(), probs.flatten()
    target = torch.arange(x.size(0), device=x.device).repeat_interleave(moe.top_k)
    out = torch.zeros_like(x)
    for expert_id, expert in enumerate(moe.experts):
        picks = (slots == expert_id).nonzero().flatten()
        rows = target.index_select(0, picks)
        scales = probs.index_select(0, picks)
        out.index_add_(0, rows, expert(x.index_select(0, rows)) * scales[:, None])
    return out


def compact_dispatch(moe, x: torch.Tensor) -> torch.Tensor:
    """Read routes once, pack valid rows, and retain the original expert accumulation order."""
    probs, indices = moe._route(x)
    expert_slots = [[] for _ in moe.experts]
    for slot, expert_id in enumerate(indices.flatten().tolist()):
        expert_slots[expert_id].append(slot)
    picks = torch.tensor([slot for slots in expert_slots for slot in slots],
                         dtype=torch.long, device=x.device)
    target = torch.div(picks, moe.top_k, rounding_mode="floor")
    packed = x.index_select(0, target)
    scales = probs.flatten().index_select(0, picks)
    out = torch.zeros_like(x)
    start = 0
    for slots, expert in zip(expert_slots, moe.experts):
        if not slots:
            continue
        stop = start + len(slots)
        ys = expert(packed[start:stop])
        out.index_add_(0, target[start:stop], ys * scales[start:stop, None])
        start = stop
    return out


class Variants:
    def __init__(self, engine: LLM, trace: ModelTrace, annotate: bool):
        self.model = engine.model_runner.model
        self.moes = [module for _, module in trace.moes]
        self.loops = [module._forward_loop for module in self.moes]
        self.compacts = [module._forward_compact_int4 for module in self.moes]
        self.cuda_linears = [(module, module._cuda_int4_ext) for module in self.model.modules()
                             if hasattr(module, "_cuda_int4_ext")]
        self.annotate = annotate

    def select(self, mode: str) -> None:
        use_cuda = mode != "triton_loop"
        for module, ext in self.cuda_linears:
            module._cuda_int4_ext = ext if use_cuda else None
        set_cuda_int4_expert_fusion(self.model, "full" if use_cuda else "none")
        for module, original, compact in zip(self.moes, self.loops, self.compacts):
            if mode == "cuda_compact":
                call = MethodType(compact_dispatch, module)
            elif mode == "cuda_index_once":
                call = MethodType(reuse_index, module)
            else:
                call = original
            module._forward_loop = scoped(call, "routed_experts") if self.annotate else call
            module._forward_compact_int4 = (scoped(compact, "routed_experts")
                                            if self.annotate else compact)
            module.int4_backend = "compact" if mode == "cuda_gpu_compact" else "loop"


def scoped(fn, phase: str):
    @wraps(fn)
    def call(*args, **kwargs):
        with nvtx_range(f"component/{phase}"):
            return fn(*args, **kwargs)
    return call


def annotate_model(model) -> None:
    from nanovllm.layers.attention_mla import MLAAttention
    from nanovllm.models.deepseek_v2 import DeepseekV2MLP

    for name, module in model.named_modules():
        if isinstance(module, MLAAttention):
            module.forward = scoped(module.forward, "attention")
        elif isinstance(module, DeepseekV2MLP):
            module.forward = scoped(module.forward, "dense_or_shared_ffn")
        elif name == "lm_head":
            module.forward = scoped(module.forward, "lm_head")


def run_batch(engine: LLM, prompts: list[list[int]], output_len: int,
              label: str | None = None) -> tuple[dict, list]:
    torch.cuda.synchronize()
    engine.reset_benchmark_metrics()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    sampling = SamplingParams(max_tokens=output_len, temperature=.8, ignore_eos=True)
    for prompt in prompts:
        engine.add_request(prompt, sampling)
    steps, outputs = [], {}
    with nvtx_range(f"sample/{label}") if label else nullcontext():
        while not engine.is_finished():
            with nvtx_range(f"step/{len(steps)}") if label else nullcontext():
                step_start = perf_counter()
                result, phase, prefill, decode = engine.step()
                steps.append(dict(phase=phase, prefill=prefill, decode=decode,
                                  ms=(perf_counter() - step_start) * 1000))
            outputs.update(result)
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    metrics = engine.collect_metrics()
    shape = [(row["phase"], row["prefill"], row["decode"]) for row in steps]
    expected = [("prefill", sum(map(len, prompts)), 0)] + [("decode", 0, len(prompts))] * (output_len - 1)
    if shape != expected or metrics["num_preemptions"] or engine.scheduler.prefix_cache_hit_tokens:
        raise RuntimeError(f"unexpected model shape or KV reuse: {shape}")
    requests = metrics["per_request"]
    return dict(wall_ms=elapsed * 1000, output_tokens_per_s=len(prompts) * output_len / elapsed,
                ttft_ms=median((r["t_first_token"] - r["t_submitted"]) * 1000 for r in requests),
                tpot_ms=median((r["t_completed"] - r["t_first_token"]) * 1000 / (output_len - 1)
                               for r in requests),
                steps=steps, memory=memory()), [outputs[key] for key in sorted(outputs)]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("~/huggingface/DeepSeek-V2-Lite"))
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS),
                        help="comma-separated subset of the timing workloads")
    parser.add_argument("--cta-fanout", type=int, default=0,
                        help="compact decode CTA fan-out per expert; 0=auto")
    parser.add_argument("--max-num-seqs", type=int, default=8,
                        help="maximum scheduled sequences (raise for b16/b32)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    llm = LLM(str(args.model.expanduser()), quantization="int4", int4_group_size=64,
              streaming_load=True, int4_dense_path=False, int4_kernel_backend="cuda",
              moe_int4_backend="compact", moe_int4_cta_fanout=args.cta_fanout,
              enforce_eager=True, execution_mode="mixed", max_model_len=520,
              max_num_batched_tokens=2048, max_num_seqs=args.max_num_seqs, gpu_memory_utilization=.85,
              multi_step_decode=False, kv_swap=False, latency_aware_scheduling=False,
              slo_aware_scheduling=False, tpot_aware_scheduling=False, cache_affinity_admission=False)
    llm.scheduler.block_manager.no_share = True
    trace = ModelTrace(llm)
    variants = Variants(llm, trace, args.trace)
    if args.trace:
        annotate_model(variants.model)
    if args.trace:
        workloads = ("b4_ctx64", "b4_ctx512")
    else:
        workloads = tuple(args.workloads.split(","))
        if not workloads or any(name not in WORKLOADS for name in workloads):
            parser.error(f"--workloads must be a subset of {sorted(WORKLOADS)}")
    vocab = llm.config.hf_config.vocab_size
    filename = "trace_workloads.json" if args.trace else "timing.json"
    output = dict(environment=environment(), kv_blocks=llm.config.num_kvcache_blocks, workloads={})
    for workload in workloads:
        batch, context = WORKLOADS[workload]
        output_len = 4
        rng = random.Random(42 + batch + context)
        prompts = [[rng.randrange(1, vocab - 10) for _ in range(context)] for _ in range(batch)]
        variants.select("cuda_loop")
        trace.begin(capture=True)
        torch.manual_seed(42)
        natural, tokens = run_batch(llm, prompts, output_len)
        routes = dict(trace.observed_routes)
        forced = torch.tensor(tokens, device="cuda").T.contiguous()

        if args.trace:
            modes = (("cuda_loop", "cuda_compact", "cuda_gpu_compact")
                     if workload == "b4_ctx64"
                     else ("cuda_loop", "cuda_gpu_compact"))
        else:
            modes = MODES
            if workload == "b4_ctx64":
                modes += ("cuda_index_once",)
        timed, parity = [], {}
        if not args.trace:
            reference = None
            for mode in ("cuda_loop", *(mode for mode in modes if mode != "cuda_loop")):
                variants.select(mode)
                trace.begin(tokens=forced, routes=routes, capture=True)
                run_batch(llm, prompts, output_len)
                logits = torch.stack(trace.logits).cpu()
                if reference is None:
                    reference = logits
                parity[mode] = error_metrics(logits, reference)
                if mode != "triton_loop" and not parity[mode]["bitwise_equal"]:
                    raise RuntimeError(f"dispatch changed model logits for {workload}: {mode}, {parity[mode]}")
        orders = list(permutations(modes))
        if len(modes) == 4:
            orders = [tuple(modes[i] for i in order) for order in
                      ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))]
        for repeat in range(1 if args.trace else args.repeats):
            for mode in orders[repeat % len(orders)]:
                variants.select(mode)
                trace.begin(tokens=forced, routes=routes)
                torch.manual_seed(42)
                label = f"{workload}/{mode}" if args.trace else None
                row, actual = run_batch(llm, prompts, output_len, label)
                if actual != tokens:
                    raise RuntimeError("forced model tokens changed")
                timed.append(dict(mode=mode, repeat=repeat, **row))
                print(workload, repeat, mode, round(row["wall_ms"], 2), flush=True)
        output["workloads"][workload] = dict(batch=batch, context=context, output_len=output_len,
                                             natural_cuda_loop=natural, logits_parity=parity, timings=timed)
        write_json(args.root / filename, output)
        del routes, forced


if __name__ == "__main__":
    main()
