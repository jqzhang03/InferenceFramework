"""Measure compact decode with and without the decode CUDA graph."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
from statistics import median

import torch

from benchmarks.deepseek_bottleneck import WORKLOADS, run_batch
from benchmarks.deepseek_w4a16 import write_json
from nanovllm import LLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("~/huggingface/DeepSeek-V2-Lite"))
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--workloads", default="b1_ctx64,b4_ctx64,b8_ctx64")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--no-silu-fusion", action="store_true",
                        help="ablate the fused SiLU gate/up MMA epilogue")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    workloads = args.workloads.split(",")
    if any(name not in WORKLOADS for name in workloads):
        parser.error(f"--workloads must be a subset of {sorted(WORKLOADS)}")
    args.root.mkdir(parents=True, exist_ok=True)

    llm = LLM(str(args.model.expanduser()), quantization="int4", int4_group_size=64,
              streaming_load=True, int4_dense_path=False, int4_kernel_backend="cuda",
              moe_int4_backend="compact", enforce_eager=args.mode == "eager",
              moe_int4_fuse_gate_up_silu=not args.no_silu_fusion,
              execution_mode="mixed", max_model_len=520,
              max_num_batched_tokens=2048, max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=.85, multi_step_decode=False, kv_swap=False,
              latency_aware_scheduling=False, slo_aware_scheduling=False,
              tpot_aware_scheduling=False, cache_affinity_admission=False)
    llm.scheduler.block_manager.no_share = True
    runner = llm.model_runner
    print("mode", args.mode, "enforce_eager", runner.enforce_eager,
          "graph_bs", getattr(runner, "graph_bs", None), flush=True)
    vocab = llm.config.hf_config.vocab_size
    output = {"mode": args.mode, "workloads": {}}

    for workload in workloads:
        batch, context = WORKLOADS[workload]
        rng = random.Random(42 + batch + context)
        prompts = [[rng.randrange(1, vocab - 10) for _ in range(context)]
                   for _ in range(batch)]
        timings, tokens = [], None
        for _ in range(args.repeats):
            torch.manual_seed(42)
            row, actual = run_batch(llm, prompts, 4)
            # run_batch resets the engine/runner counters at its start.
            row["graph_replays"] = runner.decode_cudagraph_replay_count
            row["graph_fallbacks"] = runner.decode_cudagraph_eager_fallbacks
            timings.append(row)
            tokens = actual if tokens is None else tokens
            if actual != tokens:
                raise RuntimeError(f"tokens changed for {workload}")
        output["workloads"][workload] = {
            "ttft_ms": median(row["ttft_ms"] for row in timings),
            "tpot_ms": median(row["tpot_ms"] for row in timings),
            "wall_ms": median(row["wall_ms"] for row in timings),
            "graph_replays": timings[0]["graph_replays"],
            "graph_fallbacks": timings[0]["graph_fallbacks"],
            "tokens": tokens,
            "samples": timings,
        }
        print(workload, round(output["workloads"][workload]["tpot_ms"], 2),
              output["workloads"][workload]["graph_replays"],
              output["workloads"][workload]["graph_fallbacks"], flush=True)
        write_json(args.root / f"{args.mode}.json", output)
    llm.exit()


if __name__ == "__main__":
    main()
