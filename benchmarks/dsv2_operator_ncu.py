"""Drive one DeepSeek-V2-Lite prefill and decode step under NVTX for NCU.

The production ShareGPT path runs decode inside a CUDA graph, which hides the
individual kernels from Nsight Compute launcher filtering. This harness keeps
the same engine configuration as ``benchmarks/bench.py`` (INT4 group 64,
streaming load, compact MoE) but forces eager execution and wraps exactly one
prefill step and one full-batch decode step in the NVTX ranges
``ncu_prefill`` / ``ncu_decode``. Run it under ``ncu`` with
``--nvtx --nvtx-include ncu_prefill`` (or ``ncu_decode``) and a kernel-name
filter; Nsight Systems' production trace remains the throughput reference.

Example (WSL, Nsight Compute 2026.3):

    ncu --nvtx --nvtx-include ncu_prefill \
      --kernel-name-base function \
      --kernel-name regex:.*w4a16_compact_mma_gate_up_silu_kernel.* \
      --launch-count 1 --set detailed --export profile \
      python -m benchmarks.dsv2_operator_ncu --phase prefill
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

import torch

from benchmarks.bench import add_workload_args, build_workload
from nanovllm import LLM, SamplingParams


@contextmanager
def nvtx_range(name: str):
    # Start/end ranges are what NCU's --nvtx-include matches by default;
    # push/pop ranges would need the stack-suffix syntax.
    range_id = torch.cuda.nvtx.range_start(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_end(range_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/DeepSeek-V2-Lite"))
    add_workload_args(parser)
    parser.set_defaults(num_seqs=32, min_input_len=128, max_input_len=1024,
                        min_output_len=32, max_output_len=32)
    parser.add_argument("--quantization", default="int4")
    parser.add_argument("--int4-group-size", type=int, default=64)
    parser.add_argument("--streaming-load", action="store_true", default=True)
    parser.add_argument("--moe-int4-backend", default="compact")
    parser.add_argument("--moe-int4-cta-fanout", type=int, default=0,
                        help="compact MMA CTA cap; 0=auto (decode 64, prefill 64)")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--warmup-seqs", type=int, default=32)
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase", choices=("prefill", "decode", "both"), default="both")
    parser.add_argument("--output", type=Path,
                        default=Path("profiles/deepseek_rerun_20261123/ncu/harness.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model = os.path.expanduser(args.model)
    engine = LLM(
        args.model,
        enforce_eager=True,
        execution_mode="mixed",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        int4_group_size=args.int4_group_size,
        int4_dense_path=not args.streaming_load,
        streaming_load=args.streaming_load,
        moe_int4_backend=args.moe_int4_backend,
        moe_int4_cta_fanout=args.moe_int4_cta_fanout,
        multi_step_decode=False,
        kv_swap=False,
    )
    tokenizer = engine.tokenizer if args.prompts_file else None
    prompts, sampling_params = build_workload(args, tokenizer)

    warmup_prompts = prompts[: min(args.warmup_seqs, len(prompts))]
    warmup_params = SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True)
    for _ in range(args.warmup_passes):
        engine.generate(warmup_prompts, warmup_params, use_tqdm=False)
    torch.cuda.synchronize()
    print(f"[ncu-harness] warmup passes={args.warmup_passes} "
          f"requests={len(warmup_prompts)}", flush=True)

    for prompt, params in zip(prompts, sampling_params):
        engine.add_request(prompt, params)

    steps = []
    capture_prefill = args.phase in ("prefill", "both")
    capture_decode = args.phase in ("decode", "both")
    prefill_done = not capture_prefill
    decode_ready = False
    decode_done = not capture_decode
    total_steps = 0
    while (not engine.is_finished() and not (prefill_done and decode_done)
           and total_steps < 40):
        if not prefill_done:
            with nvtx_range("ncu_prefill"):
                _, kind, n_prefill, n_decode = engine.step()
            prefill_done = True
            steps.append({"range": "ncu_prefill", "kind": kind,
                          "n_prefill": n_prefill, "n_decode": n_decode})
            total_steps += 1
            continue
        if decode_ready:
            with nvtx_range("ncu_decode"):
                _, kind, n_prefill, n_decode = engine.step()
            steps.append({"range": "ncu_decode", "kind": kind,
                          "n_prefill": n_prefill, "n_decode": n_decode})
            decode_done = True
            total_steps += 1
            break
        _, kind, n_prefill, n_decode = engine.step()
        steps.append({"range": None, "kind": kind,
                      "n_prefill": n_prefill, "n_decode": n_decode})
        total_steps += 1
        if kind in ("decode", "mixed") and n_decode == len(prompts):
            decode_ready = True
        if prefill_done and not capture_decode:
            break

    torch.cuda.synchronize()
    payload = {
        "model": args.model,
        "phase": args.phase,
        "device": torch.cuda.get_device_name(0),
        "requests": len(prompts),
        "prompt_tokens": sum(map(len, prompts)),
        "quantization": args.quantization,
        "int4_group_size": args.int4_group_size,
        "streaming_load": args.streaming_load,
        "moe_int4_backend": args.moe_int4_backend,
        "execution": "eager (decode graph disabled for kernel visibility)",
        "steps": steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[ncu-harness] steps={len(steps)} captured="
          f"{[s['range'] for s in steps if s['range']]} output={args.output}",
          flush=True)
    for step in steps:
        print(f"[ncu-harness] {step['range'] or '-':13s} kind={step['kind']:7s} "
              f"prefill={step['n_prefill']:4d} decode={step['n_decode']:3d}",
              flush=True)
    engine.exit()


if __name__ == "__main__":
    main()
