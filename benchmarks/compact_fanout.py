"""Sweep compact decode CTA fan-out on the four DeepSeek-V2-Lite workloads."""
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
    parser.add_argument("--fanouts", default="4,8,16,32,64,128")
    parser.add_argument("--workloads", default="b1_ctx64,b4_ctx64,b8_ctx64")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--output", default="fanout.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    fanouts = [int(value) for value in args.fanouts.split(",")]
    workloads = args.workloads.split(",")
    if any(name not in WORKLOADS for name in workloads):
        parser.error(f"--workloads must be a subset of {sorted(WORKLOADS)}")
    args.root.mkdir(parents=True, exist_ok=True)

    llm = LLM(str(args.model.expanduser()), quantization="int4", int4_group_size=64,
              streaming_load=True, int4_dense_path=False, int4_kernel_backend="cuda",
              moe_int4_backend="compact", enforce_eager=True, execution_mode="mixed",
              max_model_len=520, max_num_batched_tokens=2048,
              max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=.85, multi_step_decode=False, kv_swap=False,
              latency_aware_scheduling=False, slo_aware_scheduling=False,
              tpot_aware_scheduling=False, cache_affinity_admission=False)
    llm.scheduler.block_manager.no_share = True
    moes = [module for module in llm.model_runner.model.modules()
            if hasattr(module, "cta_fanout")]
    vocab = llm.config.hf_config.vocab_size
    output = {"fanouts": fanouts, "workloads": {}}

    for workload in workloads:
        batch, context = WORKLOADS[workload]
        rng = random.Random(42 + batch + context)
        prompts = [[rng.randrange(1, vocab - 10) for _ in range(context)]
                   for _ in range(batch)]
        rows = []
        for fanout in fanouts:
            for module in moes:
                module.cta_fanout = fanout
            timings = []
            for _ in range(args.repeats):
                row, _ = run_batch(llm, prompts, 4)
                timings.append(row)
            rows.append({
                "fanout": fanout,
                "ttft_ms": median(row["ttft_ms"] for row in timings),
                "tpot_ms": median(row["tpot_ms"] for row in timings),
                "wall_ms": median(row["wall_ms"] for row in timings),
                "samples": timings,
            })
            print(workload, fanout, round(rows[-1]["tpot_ms"], 2), flush=True)
        output["workloads"][workload] = rows
        write_json(args.root / args.output, output)
    llm.exit()


if __name__ == "__main__":
    main()
