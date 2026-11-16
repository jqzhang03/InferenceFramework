"""Capture an NVTX-annotated nano-vLLM inference trace for Nsight Systems.

Run this script under ``nsys profile``. The ``nsys_capture`` NVTX range marks
the inference workload after model loading and warmup. The WSL Nsight Systems
2024.6.2 CLI recorded NVTX ranges but no CUDA activity. The upgraded 2026.5.1
CLI captures CUDA activity under ordinary full-process capture; an NVTX-triggered
attempt produced no report, so select the inference range in the UI. The default
eager mode keeps individual CUDA kernels visible.

Example (WSL2):

    PYTHONPATH=. /usr/local/bin/nsys profile --trace=cuda,nvtx,osrt,cublas --stats=true \
      --sample=none --cpuctxsw=none \
      -o profiles/qwen3_0.6b_nsys \
      python benchmarks/nsys_profile.py --num-seqs 8 \
      --min-input-len 512 --max-input-len 512 \
      --min-output-len 32 --max-output-len 32 \
      --gpu-event-json profiles/qwen3_0.6b_nsys_gpu_events.json

The inference range includes engine steps, scheduler and runner stages, model
decoder layers, attention and MLP blocks, the LM head, and sampling. Detailed
module labels cover Qwen3 and DeepSeek-V2. For DeepSeek-V2-Lite on a 16 GB GPU,
use INT4 group size 64 with streaming load. NVTX/Event hooks add profiling
overhead; use the report and event summary to locate work, not to compare
throughput.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
import os
import re
import statistics

import torch

from nanovllm import LLM, SamplingParams
from bench import add_workload_args, build_workload


_MODEL_LABELS = {
    "qwen3": "Qwen3",
    "deepseek_v2": "DeepSeekV2",
}


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _wrap_method(instance, method_name: str, label: str,
                 kind_argument: int | None = None,
                 phase_state: dict[str, str] | None = None) -> bool:
    """Wrap one bound method in a named NVTX range for this profiling run."""
    original = getattr(instance, method_name, None)
    if original is None:
        return False

    @wraps(original)
    def wrapped(*args, **kwargs):
        suffix = ""
        if kind_argument is not None:
            kind = (args[kind_argument] if len(args) > kind_argument
                    else kwargs.get("kind"))
            if kind is not None:
                suffix = f" [{kind}]"
        with nvtx_range(f"{label}{suffix}"):
            previous_phase = None
            if phase_state is not None and kind is not None:
                previous_phase = phase_state["kind"]
                phase_state["kind"] = str(kind)
            try:
                return original(*args, **kwargs)
            finally:
                if previous_phase is not None:
                    phase_state["kind"] = previous_phase

    setattr(instance, method_name, wrapped)
    return True


def _module_range_label(model_type: str, module_name: str,
                        module: torch.nn.Module) -> str | None:
    """Return a stable NVTX label for supported model modules."""
    model_label = _MODEL_LABELS.get(model_type)
    if model_label is None:
        return None

    parts = module_name.split(".")
    layer_index = None
    if "layers" in parts and parts.index("layers") + 1 < len(parts):
        layer_index = parts[parts.index("layers") + 1]
    class_name = type(module).__name__

    if model_type == "qwen3":
        if class_name == "Qwen3DecoderLayer" and layer_index is not None:
            return f"{model_label}.Layer[{layer_index}]"
        if class_name == "Qwen3Attention" and layer_index is not None:
            return f"{model_label}.Layer[{layer_index}].Attention"
        if class_name == "Qwen3MLP" and layer_index is not None:
            return f"{model_label}.Layer[{layer_index}].MLP"
    elif model_type == "deepseek_v2":
        if class_name == "DeepseekV2DecoderLayer" and layer_index is not None:
            return f"{model_label}.Layer[{layer_index}]"
        if class_name == "MLAAttention" and layer_index is not None:
            return f"{model_label}.Layer[{layer_index}].MLA"
        if class_name == "DeepseekV2Moe" and layer_index is not None:
            return f"{model_label}.Layer[{layer_index}].MoE"
        if class_name == "DeepseekV2MLP" and layer_index is not None:
            suffix = "SharedMLP" if "shared_experts" in parts else "MLP"
            return f"{model_label}.Layer[{layer_index}].{suffix}"

    if module_name == "model.embed_tokens":
        return f"{model_label}.Embedding"
    if module_name == "model.norm":
        return f"{model_label}.FinalNorm"
    if module_name == "lm_head":
        return f"{model_label}.LMHead"
    return None


def _install_nvtx_ranges(engine: LLM) -> tuple[list, int, dict[str, list[tuple]]]:
    """Add profiling-only ranges without changing the engine implementation."""
    scheduler = engine.scheduler
    runner = engine.model_runner
    model_type = runner.config.hf_config.model_type
    model_label = _MODEL_LABELS.get(model_type, "Model")
    phase_state = {"kind": "unknown"}
    for instance, method_name, label, kind_argument in (
        (scheduler, "schedule", "Scheduler.schedule", None),
        (scheduler, "postprocess", "Scheduler.postprocess", None),
        (runner, "run", "ModelRunner.run", 1),
        (runner, "prepare_prefill", "ModelRunner.prepare_prefill", None),
        (runner, "prepare_decode", "ModelRunner.prepare_decode", None),
        (runner, "prepare_mixed", "ModelRunner.prepare_mixed", None),
        (runner, "prepare_spec", "ModelRunner.prepare_spec", None),
        (runner, "prepare_sample", "ModelRunner.prepare_sample", None),
        (runner, "run_model", "ModelRunner.run_model", 2),
        (runner.model, "forward", f"{model_label}.forward", None),
        (runner.model, "compute_logits", f"{model_label}.compute_logits", None),
        (runner.sampler, "forward", "Sampler.forward", None),
    ):
        _wrap_method(instance, method_name, label, kind_argument,
                     phase_state if method_name == "run" else None)

    hooks = []
    ranges = 0
    gpu_events: dict[str, list[tuple]] = {}
    for module_name, module in runner.model.named_modules():
        label = _module_range_label(model_type, module_name, module)
        if label is None:
            continue

        active_events = []

        def enter(_module, _inputs, range_name=label, active=active_events,
                  active_phase=phase_state):
            phase = active_phase["kind"]
            annotated_label = f"{range_name} [{phase}]"
            torch.cuda.nvtx.range_push(annotated_label)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            active.append((start, end, annotated_label))

        def exit_range(_module, _inputs, _output, range_name=label,
                       active=active_events):
            start, end, annotated_label = active.pop()
            end.record()
            gpu_events.setdefault(annotated_label, []).append((start, end))
            torch.cuda.nvtx.range_pop()

        hooks.append(module.register_forward_pre_hook(enter))
        hooks.append(module.register_forward_hook(exit_range))
        ranges += 1
    return hooks, ranges, gpu_events


def _summarize_gpu_events(
    gpu_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
) -> dict[str, dict[str, float | int]]:
    summary = {}
    for label, event_pairs in gpu_events.items():
        durations_ms = [start.elapsed_time(end) for start, end in event_pairs]
        summary[label] = {
            "calls": len(durations_ms),
            "total_ms": sum(durations_ms),
            "mean_ms": statistics.mean(durations_ms),
            "median_ms": statistics.median(durations_ms),
            "min_ms": min(durations_ms),
            "max_ms": max(durations_ms),
        }
    return summary


def _summarize_gpu_events_by_phase(
    module_times: dict[str, dict[str, float | int]],
    model_label: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    phase_summaries = {}
    for phase in ("prefill", "mixed", "decode"):
        suffix = f" [{phase}]"
        phase_times = {
            label[:-len(suffix)]: timing
            for label, timing in module_times.items()
            if label.endswith(suffix)
        }
        if not phase_times:
            continue

        def aggregate(predicate) -> dict[str, float | int]:
            matches = [timing for label, timing in phase_times.items()
                       if predicate(label)]
            total_ms = sum(item["total_ms"] for item in matches)
            calls = matches[0]["calls"] if matches else 0
            return {
                "modules": len(matches),
                "calls_per_module": calls,
                "total_ms": total_ms,
                "ms_per_model_forward": total_ms / calls if calls else 0.0,
            }

        layer_pattern = re.compile(
            rf"{re.escape(model_label)}\.Layer\[\d+\]")
        decoder_layers = aggregate(lambda label: layer_pattern.fullmatch(label) is not None)
        attention = aggregate(
            lambda label: label.endswith(".Attention") or label.endswith(".MLA"))
        mlp = aggregate(
            lambda label: label.endswith(".MLP") or label.endswith(".MoE"))
        shared_mlp = aggregate(lambda label: label.endswith(".SharedMLP"))
        embedding = aggregate(lambda label: label == f"{model_label}.Embedding")
        final_norm = aggregate(lambda label: label == f"{model_label}.FinalNorm")
        lm_head = aggregate(lambda label: label == f"{model_label}.LMHead")
        top_level_total_ms = sum(item["total_ms"] for item in (
            decoder_layers, embedding, final_norm, lm_head))
        forwards = decoder_layers["calls_per_module"]
        phase_summaries[phase] = {
            "model_forwards": forwards,
            "decoder_layers": decoder_layers,
            "attention_submodules": attention,
            "mlp_submodules": mlp,
            "shared_mlp_submodules": shared_mlp,
            "embedding": embedding,
            "final_norm": final_norm,
            "lm_head": lm_head,
            "top_level_model_total_ms": top_level_total_ms,
            "top_level_model_ms_per_forward": (
                top_level_total_ms / forwards if forwards else 0.0),
            "note": ("attention and MLP/MoE ranges are nested inside "
                     "decoder_layers; SharedMLP is nested inside the MoE range. "
                     "Do not add nested ranges to their parent"),
        }
    return phase_summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    add_workload_args(parser)
    parser.set_defaults(
        num_seqs=8,
        min_input_len=512,
        max_input_len=512,
        min_output_len=32,
        max_output_len=32,
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--quantization", choices=("none", "int4"), default="none",
                        help="weight quantization; DeepSeek-V2-Lite uses int4")
    parser.add_argument("--int4-group-size", type=int, default=128,
                        help="INT4 K-group size; DeepSeek-V2-Lite requires 64")
    parser.add_argument("--streaming-load", action="store_true",
                        help="load and quantize one layer at a time for large checkpoints")
    parser.add_argument("--moe-int4-backend", choices=("loop", "segment"),
                        default="loop", help="DeepSeek INT4 MoE implementation")
    parser.add_argument("--no-moe-int4-fuse-gate-up", action="store_true",
                        help="split segmented INT4 MoE gate/up kernels")
    parser.add_argument("--warmup-seqs", type=int, default=8,
                        help="number of shape-matched warmup requests; 0 uses all")
    parser.add_argument("--gpu-event-json", default="profiles/nsys_gpu_event_summary.json",
                        help="write per-module CUDA Event timings to this JSON file")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _warmup(engine: LLM, prompts: list[list[int]], sampling_params: list[SamplingParams],
            count: int) -> None:
    if count == 0:
        return
    count = min(count, len(prompts))
    vocab_size = engine.config.hf_config.vocab_size
    warmup_prompts = [[(token_id + 1) % vocab_size for token_id in prompt]
                      for prompt in prompts[:count]]
    warmup_params = [SamplingParams(
        temperature=params.temperature,
        ignore_eos=True,
        max_tokens=min(2, params.max_tokens),
    ) for params in sampling_params[:count]]
    print(f"[nsys] warmup requests={count}; inference capture starts afterward",
          flush=True)
    engine.generate(warmup_prompts, warmup_params, use_tqdm=False)
    torch.cuda.synchronize()


def main() -> None:
    args = _parse_args()
    if args.warmup_seqs < 0:
        raise ValueError("--warmup-seqs must be non-negative")
    if args.int4_group_size < 1:
        raise ValueError("--int4-group-size must be positive")
    if args.streaming_load and args.quantization == "none":
        raise ValueError("--streaming-load requires --quantization int4")
    if args.moe_int4_backend == "segment" and args.quantization != "int4":
        raise ValueError("--moe-int4-backend segment requires --quantization int4")
    if args.tp != 1:
        raise ValueError("Nsight layer annotations currently require tensor parallel size 1")
    args.model = os.path.expanduser(args.model)

    engine = LLM(
        args.model,
        enforce_eager=True,
        execution_mode="mixed",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        int4_group_size=args.int4_group_size,
        int4_dense_path=not args.streaming_load,
        streaming_load=args.streaming_load,
        moe_int4_backend=args.moe_int4_backend,
        moe_int4_fuse_gate_up=not args.no_moe_int4_fuse_gate_up,
        multi_step_decode=False,
        kv_swap=False,
    )
    tokenizer = engine.tokenizer if args.prompts_file else None
    prompts, sampling_params = build_workload(args, tokenizer)
    _warmup(engine, prompts, sampling_params, args.warmup_seqs)

    for prompt, params in zip(prompts, sampling_params):
        engine.add_request(prompt, params)

    model_type = engine.config.hf_config.model_type
    model_label = _MODEL_LABELS.get(model_type, "Model")
    hooks, module_ranges, gpu_events = _install_nvtx_ranges(engine)
    print(
        f"[nsys] model={args.model}; requests={len(prompts)}; "
        f"prompt_tokens={sum(map(len, prompts))}; eager=True; "
        f"quantization={args.quantization}; int4_group_size={args.int4_group_size}; "
        f"streaming_load={args.streaming_load}; "
        f"moe_int4_backend={args.moe_int4_backend}; "
        f"moe_int4_fuse_gate_up={not args.no_moe_int4_fuse_gate_up}; "
        f"multi_step_decode=False; {model_label} module ranges={module_ranges}",
        flush=True,
    )

    steps = 0
    step_kinds: dict[str, int] = {}
    prefill_tokens = 0
    decode_rows = 0
    torch.cuda.synchronize()
    with nvtx_range("nsys_capture"):
        while not engine.is_finished():
            with nvtx_range(f"LLMEngine.step[{steps:04d}]"):
                _, kind, n_prefill, n_decode = engine.step()
            step_kinds[kind] = step_kinds.get(kind, 0) + 1
            prefill_tokens += n_prefill
            decode_rows += n_decode
            steps += 1
        torch.cuda.synchronize()
        gpu_event_summary = _summarize_gpu_events(gpu_events)
        gpu_event_phase_summary = _summarize_gpu_events_by_phase(
            gpu_event_summary, model_label)

    summary_path = os.path.expanduser(args.gpu_event_json)
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "model_type": engine.config.hf_config.model_type,
            "model_label_prefix": model_label,
            "device": torch.cuda.get_device_name(0),
            "execution_mode": "eager",
            "tensor_parallel_size": args.tp,
            "quantization": args.quantization,
            "int4_group_size": args.int4_group_size,
            "streaming_load": args.streaming_load,
            "moe_int4_backend": args.moe_int4_backend,
            "moe_int4_fuse_gate_up": not args.no_moe_int4_fuse_gate_up,
            "multi_step_decode": False,
            "num_requests": len(prompts),
            "prompt_tokens": sum(map(len, prompts)),
            "requested_output_tokens": [params.max_tokens for params in sampling_params],
            "engine_steps": steps,
            "step_kinds": step_kinds,
            "prefill_tokens": prefill_tokens,
            "decode_rows": decode_rows,
            "timing_source": "torch.cuda.Event on the current CUDA stream",
            "timing_semantics": (
                "stream elapsed time between event records; may include idle gaps "
                "while the host queues kernels; nested module ranges overlap"),
            "nested_module_ranges_overlap": True,
            "module_gpu_times": gpu_event_summary,
            "phase_gpu_times": gpu_event_phase_summary,
        }, file, indent=2)

    print(
        f"[nsys] capture complete: steps={steps}; kinds={step_kinds}; "
        f"prefill_tokens={prefill_tokens}; decode_rows={decode_rows}; "
        f"GPU event summary={summary_path}",
        flush=True,
    )
    for phase, timing in gpu_event_phase_summary.items():
        print(
            f"[cuda-events][{phase}] model_forwards={timing['model_forwards']} "
            f"layers={timing['decoder_layers']['ms_per_model_forward']:.3f}ms "
            f"attention={timing['attention_submodules']['ms_per_model_forward']:.3f}ms "
            f"mlp={timing['mlp_submodules']['ms_per_model_forward']:.3f}ms "
            f"shared_mlp={timing['shared_mlp_submodules']['ms_per_model_forward']:.3f}ms "
            f"lm_head={timing['lm_head']['ms_per_model_forward']:.3f}ms",
            flush=True,
        )
    # Keep the hook handles alive until the capture range has closed.
    _ = hooks


if __name__ == "__main__":
    main()
