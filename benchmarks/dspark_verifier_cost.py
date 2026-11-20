"""Measure a verifier step-cost curve for DSpark prefix scheduling."""

import argparse
import json
from pathlib import Path
from random import Random
from statistics import median

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context


@torch.inference_mode()
def _measure(runner: ModelRunner, seqs: list[Sequence]) -> list[float]:
    inputs, positions = runner.prepare_spec(seqs)
    try:
        temperatures = runner.prepare_sample(seqs)

        def verify() -> tuple[torch.Tensor, torch.Tensor]:
            logits, hidden = runner.run_model(inputs, positions, "spec", return_hidden=True)
            return runner.sampler(logits, temperatures), hidden

        for _ in range(3):
            verify()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tokens, features = verify()
        graph.replay()
        samples = []
        launches = 16
        for _ in range(5):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(launches):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / launches)
        return samples
    finally:
        reset_context()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.home() / "huggingface/Qwen3-0.6B")
    parser.add_argument("--dspark-path", default="results/dspark_qwen3_0.6b/verified")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.context_len <= 251:
        parser.error("--context-len must fit the anchor and four candidates in one 256-token page")

    llm = LLM(str(args.model.expanduser()), speculative="dspark", dspark_path=args.dspark_path,
              max_model_len=1024, max_num_seqs=32, max_draft_len=4, gpu_memory_utilization=.5,
              slo_aware_scheduling=False)
    observations = []
    try:
        runner = llm.model_runner
        runner.sampler.forward = lambda logits, _: logits.argmax(dim=-1)
        rng = Random(17)
        for _ in range(32):
            llm.add_request(rng.choices(range(1_000, 32_000), k=args.context_len),
                            SamplingParams(max_tokens=32, ignore_eos=True))
        llm.step()
        active = list(llm.scheduler.running)
        if len(active) != 32 or any(seq.num_cached_tokens != len(seq) - 1 for seq in active):
            raise RuntimeError("verifier profiling requires all 32 synthetic prefixes to be cached")
        for seq in active:
            seq.is_prefill = False

        shapes = [(batch, depth) for batch in (1, 2, 4, 8, 16, 24, 32) for depth in range(5)]
        rng.shuffle(shapes)
        for batch, depth in shapes:
            seqs = active[:batch]
            for seq in seqs:
                seq.draft_tokens = [1000 + i for i in range(depth)]
                seq.num_scheduled_tokens = depth + 1
            times = _measure(runner, seqs)
            observations.append({"batch": batch, "draft_len": depth, "rows": batch * (depth + 1),
                                 "graph_gpu_ms": times, "median_ms": median(times)})
    finally:
        llm.exit()

    # Fit a smooth nondecreasing curve to row-shape medians. Isolated timing
    # noise or a single kernel selector boundary would otherwise stop admission early.
    row_counts = sorted({sample["rows"] for sample in observations})
    points = [(rows, median(s["median_ms"] for s in observations if s["rows"] == rows))
              for rows in row_counts]
    slopes = [(yb - ya) / (xb - xa) for i, (xa, ya) in enumerate(points)
              for xb, yb in points[i + 1:]]
    slope = max(0.0, median(slopes))
    intercept = median(y - slope * x for x, y in points)
    if intercept + slope <= 0:
        raise RuntimeError("could not fit a positive verifier step curve")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": str(args.model), "dspark_path": args.dspark_path,
              "device": torch.cuda.get_device_name(), "linear_backend": llm.config.linear_backend,
              "context_len": args.context_len, "timing": "steady_cuda_graph_replay",
              "observations": observations,
              "intercept_ms": intercept, "slope_ms_per_row": slope,
              "step_ms": [intercept + slope * rows for rows in range(1, row_counts[-1] + 1)]}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output}: {intercept:.3f} + {slope:.4f} ms/row", flush=True)


if __name__ == "__main__":
    main()
