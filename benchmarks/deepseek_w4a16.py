"""DeepSeek-V2-Lite INT4 baseline and same-buffer CUDA comparison."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from math import isfinite
from pathlib import Path
import random
import statistics
from time import perf_counter

import torch


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def memory() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(), device_used=total - free)


def fingerprint(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def environment() -> dict:
    return dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                cuda=torch.version.cuda, capability=torch.cuda.get_device_capability())


def make_engine(args: argparse.Namespace, backend: str | None = "triton", **backend_options):
    from nanovllm import LLM

    engine = LLM(str(args.model.expanduser()), quantization="int4", int4_group_size=64,
                 streaming_load=True, int4_dense_path=False, moe_int4_backend="loop",
                 int4_kernel_backend=backend,
                 enforce_eager=True, execution_mode="mixed", multi_step_decode=False,
                 kv_swap=False, max_num_seqs=4, max_model_len=256, max_num_batched_tokens=256,
                 slo_aware_scheduling=False, tpot_aware_scheduling=False,
                 latency_aware_scheduling=False, cache_affinity_admission=False,
                 gpu_memory_utilization=.85, **backend_options)
    if engine.config.hf_config.model_type != "deepseek_v2":
        raise ValueError("this study requires a DeepSeek-V2-Lite checkpoint")
    return engine


def make_workload(engine, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    vocab = engine.config.hf_config.vocab_size
    return [[rng.randrange(1, vocab - 10) for _ in range(64)] for _ in range(4)]


def model_batch(engine, prompts: list[list[int]], label: str | None = None) -> tuple[dict, list]:
    from benchmarks.nsys_profile import nvtx_range
    from nanovllm import SamplingParams

    torch.cuda.synchronize()
    before = memory()
    torch.cuda.reset_peak_memory_stats()
    engine.reset_benchmark_metrics()
    started = perf_counter()
    for prompt in prompts:
        engine.add_request(prompt, SamplingParams(max_tokens=4, temperature=.8, ignore_eos=True))
    steps = []
    tokens = {}
    with nvtx_range(label) if label else nullcontext():
        while not engine.is_finished():
            with nvtx_range(f"step/{len(steps)}") if label else nullcontext():
                step_start = perf_counter()
                outputs, kind, prefill, decode = engine.step()
                steps.append(dict(kind=kind, prefill=prefill, decode=decode,
                                  ms=(perf_counter() - step_start) * 1000))
            tokens.update(outputs)
        torch.cuda.synchronize()
    wall_ms = (perf_counter() - started) * 1000
    after = memory()
    metrics = engine.collect_metrics()
    shape = [(s["kind"], s["prefill"], s["decode"]) for s in steps]
    if shape != [("prefill", 256, 0), *[("decode", 0, 4)] * 3]:
        raise RuntimeError(f"unexpected model-call shapes: {shape}")
    if sum(engine._last_step_prefix_hits.values()) or metrics["num_preemptions"]:
        raise RuntimeError("matched batch unexpectedly reused a prefix or preempted")
    requests = metrics["per_request"]
    return dict(
        wall_ms=wall_ms, steps=steps, memory_before=before, memory_after=after,
        ttft_ms=statistics.median((r["t_first_token"] - r["t_submitted"]) * 1000 for r in requests),
        tpot_ms=statistics.median((r["t_completed"] - r["t_first_token"]) * 1000 / 3
                                  for r in requests),
    ), [tokens[key] for key in sorted(tokens)]


class ModelTrace:
    """Freeze model inputs, and optionally routes, while retaining router/sampler work."""

    def __init__(self, engine):
        from nanovllm.layers.moe import MoE

        model = engine.model_runner.model
        self.sampler = engine.model_runner.sampler
        self.original_sample = self.sampler.forward
        self.moes = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, MoE)]
        self.routes = {}
        self.observed_routes = {}
        self.logits = []
        self.step = 0
        self.capture = False
        self.forced = None
        self.sampler.forward = self.sample
        for name, moe in self.moes:
            moe._route = self.route_wrapper(name, moe._route)

    def route_wrapper(self, name, original):
        def route(x):
            result = original(x)
            if self.capture:
                self.observed_routes[(self.step, name)] = result
            return self.routes.get((self.step, name), result)
        return route

    def sample(self, logits, temperatures):
        result = self.original_sample(logits, temperatures)
        if self.capture:
            self.logits.append(logits.detach())
        if self.forced is not None:
            result = self.forced[self.step]
            if result.shape != (logits.size(0),):
                raise RuntimeError("forced token shape differs from sampler output")
        self.step += 1
        return result

    def begin(self, tokens=None, routes=None, capture=False):
        self.step = 0
        self.logits = []
        self.observed_routes = {}
        self.capture = capture
        self.forced = tokens
        self.routes = routes or {}


def capture_operators(engine, trace: ModelTrace, prompts: list[list[int]]) -> dict:
    cases, weights = {}, {}
    hooks = []
    modules = dict(engine.model_runner.model.named_modules())
    names = {
        "model.layers.0.self_attn.q_proj": "q",
        "model.layers.0.mlp.gate_proj": "dense_gate",
        "model.layers.0.mlp.down_proj": "dense_down",
        "model.layers.1.mlp.shared_experts.gate_proj": "shared_gate",
    }
    names.update({name: "expert_" + name.rsplit(".", 1)[1].removesuffix("_proj")
                  for name in modules if name.startswith("model.layers.1.mlp.experts.")
                  and name.endswith((".gate_proj", ".up_proj", ".down_proj"))})

    def attach(name, role):
        def capture(module, inputs):
            x = inputs[0]
            phase = "prefill" if trace.step == 0 else "decode"
            case = f"{role}_{phase}"
            if case not in cases and x.size(0) > 0:
                if name not in weights:
                    weights[name] = (module.w_int4.cpu(), module.w_int4_scale.cpu())
                cases[case] = dict(name=name, x=x.detach().contiguous().cpu())
        return modules[name].register_forward_pre_hook(capture)

    for name, role in names.items():
        hooks.append(attach(name, role))
    try:
        trace.begin(capture=True)
        torch.manual_seed(42)
        _, tokens = model_batch(engine, prompts)
    finally:
        for hook in hooks:
            hook.remove()
    if len(cases) != 14:
        raise RuntimeError(f"missing real model projection captures: {sorted(cases)}")
    # Additional row counts use the first captured real activations, without
    # changing any packed data. Each measured backend receives the same slice.
    for role, sizes in (("q", (1, 8, 16, 64, 128)), ("dense_down", (1, 16, 64)),
                        ("expert_gate", (1, 4, 8, 16, 32)),
                        ("expert_down", (1, 4, 8, 16, 32))):
        case = cases[f"{role}_prefill"]
        source = case["x"]
        for m in sizes:
            x = source.repeat((m + len(source) - 1) // len(source), 1)[:m].contiguous().clone()
            cases[f"{role}_m{m}"] = dict(name=case["name"], x=x)
    return dict(cases=cases, weights=weights, prompts=prompts, tokens=tokens,
                routes={key: tuple(value.cpu() for value in pair)
                        for key, pair in trace.observed_routes.items()})


def model_info(engine) -> dict:
    runner = engine.model_runner
    linears = [mod for mod in runner.model.modules() if getattr(mod, "int4", False)]
    buffers = dict(packed=sum(mod.w_int4.nbytes for mod in linears),
                   scales=sum(mod.w_int4_scale.nbytes for mod in linears),
                   dense_copy=sum(mod.w_deq.nbytes for mod in linears if hasattr(mod, "w_deq")))
    return dict(model=engine.config.model, kv_blocks=engine.config.num_kvcache_blocks,
                kv_bytes=runner.kv_cache.nbytes, dtype=str(engine.config.hf_config.dtype),
                int4_weight_bytes=buffers, environment=environment())


def set_backend(engine, backend: str) -> int:
    if backend == "cuda":
        from nanovllm.layers.cuda_int4 import install_cuda_int4
        return install_cuda_int4(engine.model_runner.model)
    model = engine.model_runner.model
    for mod in model.modules():
        if getattr(mod, "_cuda_int4_ext", None) is not None:
            del mod._cuda_int4_ext
    return 0


def profile_model_projections(engine, trace: ModelTrace, reference: dict,
                              backend: str, cases: list[str], root: Path) -> None:
    from benchmarks.nsys_profile import nvtx_range

    modules = dict(engine.model_runner.model.named_modules())
    started, hooks = [], []

    def attach(case: str):
        item = reference["cases"][case]
        is_prefill = case.endswith("_prefill")

        active = []

        def enter(_module, inputs):
            matched = ((trace.step == 0) == is_prefill and inputs[0].shape == item["x"].shape
                       and case not in started)
            active.append(matched)
            if matched:
                started.append(case)
                print(f"{len(started) - 1} {backend} {case} M={inputs[0].size(0)}", flush=True)
                torch.cuda.nvtx.range_push("w4a16_projection")

        def exit_range(_module, _inputs, _output):
            if active.pop():
                torch.cuda.nvtx.range_pop()

        module = modules[item["name"]]
        return module.register_forward_pre_hook(enter), module.register_forward_hook(exit_range)

    for case in cases:
        hooks.extend(attach(case))
    try:
        torch.manual_seed(42)
        torch.cuda.profiler.start()
        with nvtx_range(f"ncu_model/{backend}"):
            timing, outputs = model_batch(engine, reference["prompts"])
    finally:
        torch.cuda.profiler.stop()
        for hook in hooks:
            hook.remove()
    if set(started) != set(cases) or outputs != reference["tokens"]:
        raise RuntimeError(f"incomplete in-model counters: {started}")
    write_json(root / f"ncu_model_{backend}.json", dict(**model_info(engine), cases=started,
                                                       capture_memory=timing["memory_after"]))


def run_model_study(args: argparse.Namespace, root: Path) -> None:
    backend = args.backend if args.mode == "smoke" else "triton"
    engine = make_engine(args, None if backend == "default" else backend)
    torch.cuda.synchronize()
    initialized = memory()
    prompts = make_workload(engine, 42)
    trace = ModelTrace(engine)
    if args.mode == "smoke":
        trace.begin()
        row, tokens = model_batch(engine, prompts)
        count = engine.model_runner.cuda_int4_count
        resolved_backend = engine.config.int4_kernel_backend
        expected_backend = "cuda" if args.backend == "default" else args.backend
        if resolved_backend != expected_backend:
            raise RuntimeError(f"expected INT4 backend {expected_backend}, got {resolved_backend}")
        if resolved_backend == "cuda" and count == 0:
            raise RuntimeError("native INT4 backend did not install any linears")
        write_json(root / f"{args.backend}_smoke.json",
                   dict(**model_info(engine), row=row, tokens=tokens, adapters=count,
                        int4_kernel_backend=resolved_backend))
        return
    path = root / "snapshot.pt"
    if args.mode == "baseline":
        reference = capture_operators(engine, trace, prompts)
        if path.exists():
            raise FileExistsError(path)
        torch.save(reference, path)
    else:
        reference = torch.load(path, weights_only=True)
    prompts = reference["prompts"]
    forced = torch.tensor(reference["tokens"], device="cuda").T.contiguous()
    routes = {key: tuple(t.cuda() for t in pair) for key, pair in reference["routes"].items()}
    if args.mode == "ncu_model":
        set_backend(engine, args.backend)
        trace.begin(tokens=forced, routes=routes)
        profile_model_projections(engine, trace, reference, args.backend, args.cases, root)
        return
    backends = ["triton"] if args.mode == "baseline" else ["triton", "cuda"]
    diagnostics, timings, natural_timings = {}, [], []
    logits_by_backend = {}
    hooks = []
    if args.trace or args.mode == "nsys":
        from benchmarks.nsys_profile import _install_nvtx_ranges
        hooks, _, _ = _install_nvtx_ranges(engine, cuda_events=False)
    for backend in backends:
        adapters = set_backend(engine, backend)
        trace.begin(tokens=forced, capture=True)
        _, result = model_batch(engine, prompts)
        if result != reference["tokens"]:
            raise RuntimeError("measured token inputs differ from the baseline replay")
        logits = torch.stack(trace.logits).cpu()
        logits_by_backend[backend] = logits
        diagnostics[backend] = dict(
            cuda_adapters=adapters, logits_sha256=fingerprint(logits),
            matching_route_indices=sum(torch.equal(trace.observed_routes[key][1], pair[1])
                                       for key, pair in routes.items()),
            route_calls=len(routes))
    if len(backends) == 2:
        diagnostics["logit_comparison"] = error_metrics(logits_by_backend["cuda"],
                                                       logits_by_backend["triton"])
        diagnostics["logit_comparison"]["argmax_matches"] = int((
            logits_by_backend["cuda"].argmax(-1) == logits_by_backend["triton"].argmax(-1)).sum())
        for repeat in range(3 if args.mode == "pair" else 1):
            for backend in backends[::(-1 if repeat % 2 else 1)]:
                set_backend(engine, backend)
                trace.begin()
                torch.manual_seed(42)
                row, natural = model_batch(engine, prompts)
                diagnostics[backend]["sampled_tokens"] = natural
                natural_timings.append(dict(repeat=repeat, backend=backend, tokens=natural, **row))
    for repeat in range(args.repeats if args.mode != "nsys" else 1):
        for backend in backends[::(-1 if repeat % 2 else 1)]:
            set_backend(engine, backend)
            trace.begin(tokens=forced, routes=routes)
            torch.manual_seed(42)
            label = f"model_capture/{backend}" if args.trace or args.mode == "nsys" else None
            row, output = model_batch(engine, prompts, label)
            if output != reference["tokens"]:
                raise RuntimeError("forced replay tokens differ")
            timings.append(dict(repeat=repeat, backend=backend, **row))
            print(args.mode, repeat, backend, round(row["wall_ms"], 2), flush=True)
    for hook in hooks:
        hook.remove()
    write_json(root / f"model_{args.mode}.json", dict(**model_info(engine),
               initialized=initialized, prompts=prompts, forced_tokens=reference["tokens"],
               diagnostics=diagnostics, timings=timings, natural_timings=natural_timings))


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.float(), expected.float()
    diff = actual - expected
    return dict(bitwise_equal=bool(torch.equal(actual, expected)),
                max_abs=float(diff.abs().max()), mae=float(diff.abs().mean()),
                relative_l2=float(diff.norm() / expected.norm().clamp_min(1e-12)))


def worker(args: argparse.Namespace, root: Path) -> None:
    from benchmarks.nsys_profile import nvtx_range
    from nanovllm.layers.linear import int4_gemm

    snapshot = torch.load(root / "snapshot.pt", weights_only=True)
    if args.mode == "ncu":
        cases = args.cases
        if not cases:
            raise ValueError("specify one or more --cases for Nsight Compute")
        backends = [args.backend]
    else:
        cases = args.cases or sorted(snapshot["cases"])
        backends = ["triton", "cuda"]
    ext = None
    if "cuda" in backends:
        from nanovllm.layers.cuda_int4 import load_cuda_int4
        ext = load_cuda_int4()
    results = {}
    for index, case in enumerate(cases):
        data = snapshot["cases"][case]
        x = data["x"].cuda()
        packed, scales = (t.cuda() for t in snapshot["weights"][data["name"]])
        m, k = x.shape
        n = packed.size(0)
        group = k // scales.size(1)

        def triton_call():
            return int4_gemm(x, packed, scales)

        def cuda_call():
            out = torch.empty((m, n), device=x.device, dtype=x.dtype)
            ext.gemm_out(x, packed, scales, out)
            return out

        ops = {"triton": triton_call, "cuda": cuda_call}
        for backend in backends:
            for _ in range(10):
                ops[backend]()
        torch.cuda.synchronize()
        if args.mode == "ncu":
            print(f"{index} {args.backend} {case} M={m} N={n} K={k}", flush=True)
            torch.cuda.profiler.start()
            with nvtx_range("op_capture/" + case):
                ops[args.backend]()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            continue
        w = packed.to(torch.int32)
        unpacked = torch.stack(((w & 15) - 8, ((w >> 4) & 15) - 8), -1).view(n, k)
        dense = (unpacked.float().view(n, -1, group) * scales.float().unsqueeze(-1))
        ref = x.double() @ dense.view(n, k).to(x.dtype).double().T
        triton, cuda = triton_call(), cuda_call()
        ref = ref.to(x.dtype)
        errors = dict(cuda_vs_triton=error_metrics(cuda, triton),
                      cuda_vs_fp64=error_metrics(cuda, ref), triton_vs_fp64=error_metrics(triton, ref))
        if not all(isfinite(row["relative_l2"]) and row["relative_l2"] <= .005
                   for row in errors.values()):
            raise AssertionError(f"CUDA mismatch for {case}: {errors}")
        del w, unpacked, dense, ref, triton, cuda
        torch.cuda.synchronize()
        graph = {}
        mem = {}
        for backend in backends:
            torch.cuda.empty_cache()
            before = memory()
            torch.cuda.reset_peak_memory_stats()
            out = ops[backend]()
            torch.cuda.synchronize()
            after = memory()
            mem[backend] = dict(before=before, after=after, output_bytes=out.nbytes,
                                extra_peak_bytes=after["peak_allocated"] - before["allocated"] - out.nbytes)
            del out
        launches = 32
        for backend in backends:
            graph[backend] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph[backend]):
                for _ in range(launches):
                    ops[backend]()
            graph[backend].replay()
        torch.cuda.synchronize()
        samples = {backend: [] for backend in backends}
        for repeat in range(args.repeats):
            for backend in backends[::(-1 if (repeat + index) % 2 else 1)]:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(5):
                    graph[backend].replay()
                end.record()
                end.synchronize()
                samples[backend].append(begin.elapsed_time(end) * 1000 / (5 * launches))
        results[case] = dict(m=m, n=n, k=k, name=data["name"], errors=errors, memory=mem,
                             input_sha256=fingerprint(data["x"]),
                             packed_sha256=fingerprint(packed), scales_sha256=fingerprint(scales),
                             packed_bytes=packed.nbytes, scale_bytes=scales.nbytes, graph_us=samples)
        print(case, m, n, k, {key: round(statistics.median(value), 3) for key, value in samples.items()},
              flush=True)
    if results:
        write_json(root / "operators.json", dict(environment=environment(), cases=results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "pair", "nsys", "smoke", "operators", "ncu", "ncu_model"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("~/huggingface/DeepSeek-V2-Lite"))
    parser.add_argument("--backend", choices=("triton", "cuda", "default"), default="triton")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--trace", action="store_true", help="label the baseline under Nsight Systems")
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.mode == "ncu_model" and not args.cases:
        parser.error("ncu_model requires --cases")
    if args.backend == "default" and args.mode != "smoke":
        parser.error("--backend default is only used by the smoke check")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.mode in ("operators", "ncu"):
        worker(args, args.root)
    else:
        run_model_study(args, args.root)


if __name__ == "__main__":
    main()
