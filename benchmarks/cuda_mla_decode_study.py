"""Compare CUDA and Triton absorbed MLA decode with the same paged KV inputs."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from statistics import median
from unittest.mock import patch

import torch

from benchmarks.deepseek_w4a16 import environment, write_json
from nanovllm.layers.attention_mla import mla_decode_attention
from nanovllm.layers.cuda_mla_decode import cuda_mla_decode_attention, load_cuda_mla_decode


def graph_times(call, *, calls: int = 32) -> list[float]:
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            call()
    graph.replay()
    return [time_graph(graph, calls) for _ in range(3)]


def time_graph(graph: torch.cuda.CUDAGraph, calls: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(4):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (4 * calls)


def make_operator_inputs(batch: int, length: int) -> tuple:
    pages_per_request = (length + 255) // 256
    pages = torch.randperm(batch * pages_per_request, device="cuda",
                           dtype=torch.int32).view(batch, pages_per_request)
    cache = torch.randn(batch * pages_per_request, 256, 576,
                        dtype=torch.bfloat16, device="cuda")
    q = torch.randn(batch, 16, 576, dtype=torch.float16, device="cuda")
    q_abs, q_pe = q.split((512, 64), dim=-1)
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    return q_abs, q_pe, cache, pages, lengths, 192**-0.5, 1.0, 1.0


def operator_cases() -> list[dict]:
    ext = load_cuda_mla_decode(torch.device("cuda"), required=True)
    cases = []
    for batch in (1, 8, 16, 32):
        for length in (128, 1024, 4096):
            args = make_operator_inputs(batch, length)

            def triton_call() -> torch.Tensor:
                return mla_decode_attention(*args)

            def cuda_call() -> torch.Tensor:
                return cuda_mla_decode_attention(*args, ext)

            ref, output = triton_call().float(), cuda_call().float()
            torch.testing.assert_close(output, ref, atol=2e-3, rtol=.008)
            triton_us = graph_times(triton_call)
            cuda_us = graph_times(cuda_call)
            cases.append(dict(batch=batch, context=length, triton_us=triton_us, cuda_us=cuda_us,
                              max_abs=(output - ref).abs().max().item(),
                              relative_l2=((output - ref).norm() / ref.norm()).item()))
            print(f"MLA B{batch:<2} L{length:<4} Triton {median(triton_us):7.2f} us "
                  f"CUDA {median(cuda_us):7.2f} us", flush=True)
    return cases


def model_cases(model: Path, data: Path, baseline: str = "triton",
                diagnose_original: bool = False) -> dict:
    from nanovllm import LLM
    from benchmarks.deepseek_fp4 import Greedy, select_texts, timed_batch
    from nanovllm.layers import attention_mla

    engine = LLM(str(model.expanduser()), quantization="int4", int4_group_size=64,
                 moe_int4_backend="compact", streaming_load=True, mla_decode_backend="cuda",
                 max_num_seqs=8, max_model_len=144, max_num_batched_tokens=1024,
                 multi_step_decode=False, mixed_cudagraph=False, kv_swap=False,
                 gpu_memory_utilization=.8)
    try:
        runner = engine.model_runner
        modules = runner._mla_modules
        assert modules and all(layer._cuda_mla_decode_ext is not None for layer in modules)
        extension = modules[0]._cuda_mla_decode_ext
        runner.sampler = Greedy()
        fields = ("graphs", "graph_vars", "graph_pool", "graph_bs")

        def graph_state() -> dict:
            return {field: getattr(runner, field) for field in fields}

        def capture() -> dict:
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(engine.config.hf_config.dtype)
                with torch.device("cuda"):
                    runner.capture_cudagraph()
            finally:
                torch.set_default_dtype(previous_dtype)
            return graph_state()

        def original_variant(name: str):
            if name == "original":
                return patch.object(attention_mla, "cuda_mla_decode_attention",
                                    partial(cuda_mla_decode_attention, variant="original"))
            return nullcontext()

        # Capture both CUDA variants after engine startup so the benchmark uses
        # the same default-device and compilation context for each graph.
        cuda_graphs = capture() if baseline == "original" else graph_state()
        if baseline == "triton":
            for layer in modules:
                layer._cuda_mla_decode_ext = None
        with original_variant(baseline):
            graph_states = {"cuda": cuda_graphs, baseline: capture()}

        def select(name: str) -> None:
            for field, value in graph_states[name].items():
                setattr(runner, field, value)
            for layer in modules:
                layer._cuda_mla_decode_ext = None if name == "triton" else extension

        def run(name: str, prompts: list[list[int]]) -> dict:
            select(name)
            with original_variant(name):
                return timed_batch(engine, prompts, 16)

        class CaptureGreedy(Greedy):
            def __init__(self) -> None:
                super().__init__()
                self.logits = []

            def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
                self.logits.append(logits.clone())
                return super().forward(logits, temperatures)

        def first_logit_difference(prompts: list[list[int]]) -> dict | None:
            records = {}
            for name in (baseline, "cuda"):
                runner.sampler = CaptureGreedy()
                run(name, prompts)
                records[name] = runner.sampler.logits
            runner.sampler = Greedy()
            for step, (baseline_logits, cuda_logits) in enumerate(
                    zip(records[baseline], records["cuda"])):
                if baseline_logits.shape != cuda_logits.shape:
                    raise RuntimeError("model logit shapes differ across the diagnostic pair")
                baseline_ids, cuda_ids = baseline_logits.argmax(-1), cuda_logits.argmax(-1)
                different = (baseline_ids != cuda_ids).nonzero().flatten()
                if different.numel():
                    row = int(different[0])
                    ids = [int(baseline_ids[row]), int(cuda_ids[row])]
                    return dict(step=step, row=row, candidate_ids=ids,
                                **{f"{baseline}_logits": baseline_logits[row, ids].tolist()},
                                cuda_logits=cuda_logits[row, ids].tolist(),
                                max_abs=(baseline_logits[row].float() - cuda_logits[row].float())
                                .abs().max().item())
            return None

        def compare_same_graph(prompts: list[list[int]]) -> dict:
            counts = []

            def compare(*args) -> torch.Tensor:
                original = cuda_mla_decode_attention(*args, variant="original")
                if args[0].shape[0] == len(prompts) and torch.cuda.is_current_stream_capturing():
                    optimized = cuda_mla_decode_attention(*args)
                    counts.append((original.view(torch.int16) != optimized.view(torch.int16)).count_nonzero())
                return original

            select("cuda")
            with patch.object(attention_mla, "cuda_mla_decode_attention", compare):
                graph_states["shadow"] = capture()
            if len(counts) != len(modules):
                raise RuntimeError(f"expected {len(modules)} MLA comparisons, recorded {len(counts)}")

            class ShadowGreedy(Greedy):
                def __init__(self) -> None:
                    super().__init__()
                    self.rows = []
                    self.step = 0

                def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
                    if self.step:
                        self.rows.append(torch.stack(counts).tolist())
                    self.step += 1
                    return super().forward(logits, temperatures)

            runner.sampler = ShadowGreedy()
            try:
                result = run("shadow", prompts)
                assert result["decode_graph_replays"] == len(runner.sampler.rows) == 15
                return dict(mismatched_elements_per_step_layer=runner.sampler.rows,
                            output_token_ids=result["output_token_ids"])
            finally:
                runner.sampler = Greedy()
                del graph_states["shadow"]
                select("cuda")

        prompts = [p for _, p in select_texts(data, engine.tokenizer, 8, 128, 42)]
        results = {}
        for batch in (1, 8):
            result = {"cuda": [], baseline: []}
            for name in result:
                run(name, prompts[:batch])
            for rep in range(3):
                for name in ((baseline, "cuda") if rep % 2 == 0 else ("cuda", baseline)):
                    result[name].append(run(name, prompts[:batch]))
            for runs in result.values():
                assert all(r["decode_graph_replays"] == 15 for r in runs)
            result["tokens_equal"] = all(
                x["output_token_ids"] == y["output_token_ids"]
                for x, y in zip(result["cuda"], result[baseline]))
            if not result["tokens_equal"]:
                result["first_logit_difference"] = first_logit_difference(prompts[:batch])
                if diagnose_original and baseline == "original":
                    result["same_graph"] = compare_same_graph(prompts[:batch])
            results[str(batch)] = result
            print(f"Model B{batch} tokens_equal={result['tokens_equal']}", flush=True)
        return results
    finally:
        engine.exit()


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, help="Optional DeepSeek-V2-Lite checkpoint for model replay")
    p.add_argument("--model-baseline", choices=("triton", "original"), default="triton")
    p.add_argument("--diagnose-original", action="store_true",
                   help="Compare both CUDA outputs within one graph if greedy tokens differ")
    p.add_argument("--data", type=Path,
                   default=Path("results/supplemental/sharegpt_500_heldout250_20261016.jsonl"))
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a new output path")
    torch.manual_seed(42)
    data = dict(environment=environment(), operators=operator_cases())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, data)
    if args.model is not None:
        torch.cuda.empty_cache()
        data["model"] = model_cases(args.model, args.data, args.model_baseline,
                                    args.diagnose_original)
        write_json(args.output, data)


if __name__ == "__main__":
    main()
