"""Compare native W4A16 expert launches on the same DeepSeek weights and routes."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import permutations
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from benchmarks.deepseek_w4a16 import ModelTrace, environment, error_metrics, make_engine, model_batch, write_json
from nanovllm.layers.cuda_int4 import set_cuda_int4_expert_fusion


MODES = ("none", "gate_up", "full")
LAUNCHES = dict(zip(MODES, (5, 2, 1)))
ORDERS = tuple(permutations(MODES))


def expert_timing(model, experts: list, x: torch.Tensor, repeats: int) -> dict:
    graphs, errors = {}, {}
    expected = [expert(x) for expert in experts]
    launches_per_graph = 32

    def work():
        for index in range(launches_per_graph):
            experts[index % len(experts)](x)

    for mode in MODES:
        set_cuda_int4_expert_fusion(model, mode)
        actual = [expert(x) for expert in experts]
        errors[mode] = [error_metrics(a, b) for a, b in zip(actual, expected)]
        if any(row["relative_l2"] > .005 for row in errors[mode]):
            raise AssertionError(f"expert output differs for {mode}, M={x.size(0)}")
        work()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            work()
        graphs[mode] = graph
        graph.replay()
    torch.cuda.synchronize()
    event_us = {mode: [] for mode in MODES}
    eager_us = {mode: [] for mode in MODES}
    for repeat in range(repeats):
        for mode in ORDERS[repeat % len(ORDERS)]:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(5):
                graphs[mode].replay()
            end.record()
            end.synchronize()
            event_us[mode].append(begin.elapsed_time(end) * 1000 / (5 * launches_per_graph))
        for mode in ORDERS[repeat % len(ORDERS)]:
            set_cuda_int4_expert_fusion(model, mode)
            started = perf_counter()
            for _ in range(3):
                work()
            torch.cuda.synchronize()
            eager_us[mode].append((perf_counter() - started) * 1e6 / (3 * launches_per_graph))
    set_cuda_int4_expert_fusion(model, "none")
    return dict(graph_us=event_us, eager_us=eager_us, errors=errors)


def model_timing(engine, reference: dict, repeats: int, capture: bool = False) -> dict:
    model = engine.model_runner.model
    trace = ModelTrace(engine)
    forced = torch.tensor(reference["tokens"], device="cuda").T.contiguous()
    routes = {key: tuple(t.cuda() for t in pair) for key, pair in reference["routes"].items()}
    prompts = reference["prompts"]
    rows = Counter()

    def count_expert(_module, args):
        rows[(trace.step, args[0].size(0))] += 1

    hooks = [expert.register_forward_pre_hook(count_expert)
             for _, layer in trace.moes for expert in layer.experts]
    try:
        trace.begin(tokens=forced, routes=routes)
        model_batch(engine, prompts)
    finally:
        for hook in hooks:
            hook.remove()
    timings, logits, naturally_sampled = [], {}, {}
    if not capture:
        for mode in MODES:
            count = set_cuda_int4_expert_fusion(model, mode)
            if mode != "none" and count != sum(len(layer.experts) for _, layer in trace.moes):
                raise RuntimeError(f"installed {count} fused experts")
            trace.begin(tokens=forced, capture=True)
            torch.manual_seed(42)
            _, tokens = model_batch(engine, prompts)
            if tokens != reference["tokens"]:
                raise AssertionError("controlled model input token stream changed")
            logits[mode] = torch.stack(trace.logits).cpu()
            trace.begin()
            torch.manual_seed(42)
            _, naturally_sampled[mode] = model_batch(engine, prompts)

    for repeat in range(repeats):
        for mode in ORDERS[repeat % len(ORDERS)]:
            set_cuda_int4_expert_fusion(model, mode)
            trace.begin(tokens=forced, routes=routes)
            torch.manual_seed(42)
            label = f"moe_fusion_model/{mode}" if capture else None
            row, tokens = model_batch(engine, prompts, label)
            if tokens != reference["tokens"]:
                raise AssertionError("controlled model input token stream changed")
            print("model", repeat, mode, round(row["wall_ms"], 2), flush=True)
            timings.append(dict(mode=mode, repeat=repeat, **row))
    launch_counts = {}
    for phase, prefill in (("prefill", True), ("decode", False)):
        launch_counts[phase] = {
            mode: sum((LAUNCHES[mode] if size <= 4 else LAUNCHES["none"]) * count
                      for (step, size), count in rows.items() if size and (step == 0) == prefill)
            for mode in MODES
        }
    return dict(timings=timings, kernel_launches=launch_counts,
                expert_rows=[dict(step=step, m=m, calls=count)
                             for (step, m), count in sorted(rows.items())],
                logits={mode: error_metrics(value, logits["none"]) for mode, value in logits.items()},
                sampled_tokens=naturally_sampled, kv_blocks=engine.config.num_kvcache_blocks)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path,
                        default=Path("profiles/deepseek_w4a16_paired_20261117/snapshot.pt"))
    parser.add_argument("--model", type=Path, default=Path("~/huggingface/DeepSeek-V2-Lite"))
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    reference = torch.load(args.snapshot, weights_only=True)
    engine = make_engine(args, "cuda", cuda_int4_expert_fusion="gate_up")
    model = engine.model_runner.model
    case = reference["cases"]["expert_gate_prefill"]
    name = case["name"].rsplit(".", 1)[0]
    expert = model.get_submodule(name)
    if getattr(expert, "_cuda_int4_fusion", None) is not False:
        raise RuntimeError("model configuration did not install gate/up fusion")
    set_cuda_int4_expert_fusion(model, "none")
    expected_packed, expected_scales = reference["weights"][case["name"]]
    if (not torch.equal(expert.gate_proj.w_int4.cpu(), expected_packed)
            or not torch.equal(expert.gate_proj.w_int4_scale.cpu(), expected_scales)):
        raise RuntimeError("checkpoint no longer matches the saved expert")
    source = case["x"].cuda()
    layer = model.get_submodule(name.rsplit(".", 1)[0])
    if args.trace:
        from benchmarks.nsys_profile import nvtx_range

        for mode in MODES:
            set_cuda_int4_expert_fusion(model, mode)
            for m in (1, 2, 3, 4):
                x = source[:m].contiguous()
                expert(x)
                torch.cuda.synchronize()
                with nvtx_range(f"moe_fusion_expert/{mode}/{m}"):
                    expert(x)
                    torch.cuda.synchronize()
        set_cuda_int4_expert_fusion(model, "none")
        result = model_timing(engine, reference, 1, capture=True)
        write_json(args.root / "model_nsys.json", dict(environment=environment(), **result))
        return
    operators = {}
    for working_set, experts in (("resident", [expert]), ("rotating", list(layer)[:16])):
        operators[working_set] = {}
        for rows in (1, 2, 3, 4):
            x = source.repeat((rows + len(source) - 1) // len(source), 1)[:rows].contiguous()
            timing = expert_timing(model, experts, x, args.repeats)
            operators[working_set][rows] = timing
            print(working_set, rows, {k: round(median(v), 3) for k, v in timing["graph_us"].items()},
                  flush=True)
    write_json(args.root / "operators.json", dict(environment=environment(), operators=operators))
    result = model_timing(engine, reference, args.repeats)
    write_json(args.root / "model.json", dict(environment=environment(), **result))


if __name__ == "__main__":
    main()
