"""Kernel-level A/B for the fused SiLU gate/up epilogue (compact INT4 MoE).

Uses the DeepSeek-V2-Lite routed-expert geometry (hidden 2048, intermediate
1408, 64 experts, top-k 6) and drives the gate/up kernels with synthetic expert
segments, so each MMA template (8/16/32 rows) is exercised on its own instead of
depending on how the router happens to spread a token batch. Three timings are
reported per segment size:

1. the split gate/up kernel (two outputs);
2. the split kernel plus its ``F.silu(gate)*up`` elementwise, i.e. what the
   pre-fusion Python path issued;
3. the fused kernel that writes ``silu(gate)*up`` directly.

All three are checked bitwise against each other before timing. ``--forward``
additionally times the whole MoE forward on router-produced routes, which is the
unit a decode step or a prefill chunk actually calls.

Note: torch.profiler CUDA activities fail in this WSL environment
(CUPTI_ERROR_INVALID_DEVICE), so kernel attribution comes from nsys instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F

from benchmarks.deepseek_w4a16 import environment, memory, write_json
from nanovllm.layers.cuda_int4 import install_cuda_int4
from nanovllm.layers.moe import MoE

HIDDEN = 2048
INTERMEDIATE = 1408
EXPERTS = 64
TOP_K = 6
PAD_ROWS = 32


def build_moe(seed: int = 23) -> MoE:
    torch.manual_seed(seed)
    moe = (MoE(HIDDEN, INTERMEDIATE, num_experts=EXPERTS, top_k=TOP_K,
               int4_backend="compact").cuda().to(torch.bfloat16))
    with torch.no_grad():
        for param in moe.parameters():
            param.uniform_(-0.05, 0.05)
    for expert in moe.experts:
        for linear in (expert.gate_proj, expert.up_proj, expert.down_proj):
            linear.quantize_int4(dense_path=False, group_size=64)
    install_cuda_int4(moe)
    return moe


def synthetic_route(moe: MoE, per_expert: int, seed: int) -> dict:
    """Every expert gets ``per_expert`` rows, so one MMA template is isolated."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows = EXPERTS * per_expert
    counts = torch.full((EXPERTS,), per_expert, dtype=torch.int32, device="cuda")
    offsets = torch.arange(EXPERTS, dtype=torch.int32, device="cuda") * per_expert
    padded = torch.randn(rows + PAD_ROWS, HIDDEN, generator=generator,
                         device="cuda", dtype=torch.bfloat16)
    gate = moe.experts[0].gate_proj
    group = gate.w_int4.size(1) * 2 // gate.w_int4_scale.size(1)
    block = 8 if per_expert <= 8 else 16 if per_expert <= 16 else 32
    return dict(padded=padded, offsets=offsets, counts=counts, tokens=rows,
                rows=rows, rows_cap=PAD_ROWS, group=group,
                tables=moe._ensure_int4_segment_tables(padded),
                ranges=[(per_expert, per_expert, block)],
                fanout=moe.cta_fanout or 64)


def project_gate_up(moe: MoE, route: dict, fused: bool, hidden: torch.Tensor,
                    gate_out: torch.Tensor | None = None,
                    up_out: torch.Tensor | None = None) -> None:
    extension = moe.experts[0].gate_proj._cuda_int4_ext
    tables = route["tables"]
    for lo, hi, block_m in route["ranges"]:
        if fused:
            extension.gemm_gate_up_silu_compact(
                route["padded"], tables["gate_w"], tables["gate_s"],
                tables["up_w"], tables["up_s"], route["offsets"],
                route["counts"], hidden, route["tokens"], route["rows"],
                route["group"], route["fanout"], block_m, lo, hi)
        else:
            extension.gemm_gate_up_out_compact(
                route["padded"], tables["gate_w"], tables["gate_s"],
                tables["up_w"], tables["up_s"], route["offsets"],
                route["counts"], gate_out, up_out, route["tokens"],
                route["rows"], route["group"], route["fanout"], block_m, lo, hi)


def time_calls(call, iterations: int) -> float:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def measure(moe: MoE, route: dict, repeats: int, iterations: int) -> dict:
    rows = route["rows"]
    hidden = route["padded"].new_zeros((rows + route["rows_cap"], INTERMEDIATE))
    gate_out = route["padded"].new_empty((rows + route["rows_cap"], INTERMEDIATE))
    up_out = route["padded"].new_empty((rows + route["rows_cap"], INTERMEDIATE))

    def split_kernel():
        project_gate_up(moe, route, False, None, gate_out, up_out)

    def split_full():
        project_gate_up(moe, route, False, None, gate_out, up_out)
        return F.silu(gate_out) * up_out

    def fused_kernel():
        project_gate_up(moe, route, True, hidden)

    split_kernel()
    reference = F.silu(gate_out) * up_out
    fused_kernel()
    bitwise = torch.equal(hidden[:rows], reference[:rows])
    if not bitwise:
        raise RuntimeError(f"fused gate/up is not bitwise for {route['ranges']}")
    samples = {"split_kernel": [], "split_full": [], "fused": []}
    for _ in range(repeats):
        samples["split_kernel"].append(time_calls(split_kernel, iterations))
        samples["split_full"].append(time_calls(split_full, iterations))
        samples["fused"].append(time_calls(fused_kernel, iterations))
    row = dict(tokens=route["tokens"], edges=[list(item) for item in route["ranges"]],
               bitwise=bitwise,
               split_kernel_ms=median(samples["split_kernel"]),
               split_full_ms=median(samples["split_full"]),
               fused_ms=median(samples["fused"]),
               samples=samples)
    row["kernel_delta_pct"] = (row["fused_ms"] / row["split_kernel_ms"] - 1) * 100
    row["full_delta_pct"] = (row["fused_ms"] / row["split_full_ms"] - 1) * 100
    print(f"rows/expert={row['edges'][0][0]:3d} block={row['edges'][0][2]:2d} "
          f"| split {row['split_kernel_ms']:7.3f}  +elementwise {row['split_full_ms']:7.3f}  "
          f"fused {row['fused_ms']:7.3f} ms | kernel {row['kernel_delta_pct']:+6.2f}% "
          f"full {row['full_delta_pct']:+6.2f}%", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows-per-expert", default="5,6,8,10,12,16,20,24,32,40,48,64,96")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--forward-tokens", default="",
                        help="comma-separated token counts for the whole MoE forward")
    parser.add_argument("--forward-repeats", type=int, default=5)
    parser.add_argument("--engine", action="store_true",
                        help="also run the paired prefill A/B through the real engine")
    parser.add_argument("--model", type=Path,
                        default=Path("~/huggingface/DeepSeek-V2-Lite"))
    parser.add_argument("--prompts-file",
                        default="results/sharegpt_profile_20261123.jsonl")
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--engine-pairs", type=int, default=6)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.root.mkdir(parents=True, exist_ok=True)

    moe = build_moe() if (args.rows_per_expert.strip() or args.forward_tokens) else None
    output = {"environment": environment(), "geometry": dict(
        hidden=HIDDEN, intermediate=INTERMEDIATE, experts=EXPERTS, top_k=TOP_K,
        int4_group=64), "templates": [], "memory": memory()}
    for per_expert in [int(value) for value in args.rows_per_expert.split(",")
                       if value.strip()]:
        iterations = 30 if per_expert <= 16 else 10
        with torch.inference_mode():
            route = synthetic_route(moe, per_expert, seed=per_expert)
            output["templates"].append(measure(moe, route, args.repeats, iterations))
        write_json(args.root / "kernel_ab.json", output)

    if args.forward_tokens:
        output["forward"] = []
        for count in [int(value) for value in args.forward_tokens.split(",")]:
            x = torch.randn(count, HIDDEN, device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode():
                moe.fuse_int4_gate_up_silu = False
                split = time_calls(lambda: moe(x), 5 if count > 512 else 20)
                moe.fuse_int4_gate_up_silu = True
                fused = time_calls(lambda: moe(x), 5 if count > 512 else 20)
            output["forward"].append(dict(tokens=count, split_ms=split, fused_ms=fused,
                                          delta_pct=(fused / split - 1) * 100))
            print(f"forward T={count:5d} split {split:7.3f} fused {fused:7.3f} ms "
                  f"({(fused / split - 1) * 100:+6.2f}%)", flush=True)
        write_json(args.root / "kernel_ab.json", output)
    if moe is not None:
        moe.cpu()
    if args.engine:
        output["engine_prefill"] = engine_prefill(args)
        write_json(args.root / "kernel_ab.json", output)


def engine_prefill(args: argparse.Namespace) -> dict:
    """Paired prefill A/B inside one process (no cross-process clock drift)."""
    import json

    from transformers import AutoTokenizer

    from nanovllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(args.model.expanduser()), use_fast=True)
    rows = [json.loads(line) for line in Path(args.prompts_file).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    prompts = [tokenizer.encode(row["prompt"]) for row in rows[:args.num_seqs]]
    llm = LLM(str(args.model.expanduser()), quantization="int4", int4_group_size=64,
              streaming_load=True, int4_dense_path=False,
              int4_kernel_backend="cuda", moe_int4_backend="compact",
              execution_mode="mixed", max_model_len=1024,
              gpu_memory_utilization=.85, max_num_seqs=args.num_seqs,
              multi_step_decode=False, kv_swap=False,
              latency_aware_scheduling=False, slo_aware_scheduling=False,
              tpot_aware_scheduling=False, cache_affinity_admission=False)
    moes = [module for module in llm.model_runner.model.modules()
            if isinstance(module, MoE)]
    sampling = SamplingParams(max_tokens=2, temperature=.6, ignore_eos=True)

    def run(fused: bool) -> dict:
        for module in moes:
            module.fuse_int4_gate_up_silu = fused
        llm.generate(prompts, sampling, use_tqdm=False)
        metrics = llm.collect_metrics()
        stats = metrics["step_stats"]
        ttft = [(row["t_first_token"] - row["t_submitted"]) * 1000
                for row in metrics["per_request"] if row["t_first_token"]]
        return dict(prefill_ms=stats["prefill_time"] * 1000,
                    prefill_tokens=stats["prefill_tokens"],
                    ttft_ms=median(ttft) if ttft else None)

    for fused in (True, False):  # warm both paths and the prefix cache
        run(fused)
    pairs = []
    for index in range(args.engine_pairs):
        order = (True, False) if index % 2 == 0 else (False, True)
        row = {}
        for fused in order:
            row["fused" if fused else "split"] = run(fused)
        pairs.append(row)
    fused_ms = [row["fused"]["prefill_ms"] for row in pairs]
    split_ms = [row["split"]["prefill_ms"] for row in pairs]
    result = dict(prefill_tokens=pairs[0]["fused"]["prefill_tokens"],
                  pairs=pairs, fused_median_ms=median(fused_ms),
                  split_median_ms=median(split_ms),
                  fused_samples=fused_ms, split_samples=split_ms,
                  delta_pct=(median(fused_ms) / median(split_ms) - 1) * 100)
    print(f"engine prefill ({result['prefill_tokens']} tok): "
          f"split {result['split_median_ms']:.1f} ms "
          f"fused {result['fused_median_ms']:.1f} ms "
          f"({result['delta_pct']:+.2f}%)", flush=True)
    print("  fused samples", [round(v, 1) for v in fused_ms], flush=True)
    print("  split samples", [round(v, 1) for v in split_ms], flush=True)
    llm.exit()
    return result


if __name__ == "__main__":
    main()
