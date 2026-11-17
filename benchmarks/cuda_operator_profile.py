"""Custom CUDA operators only: unprofiled timings, nsys timelines and NCU counters."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys


MODULE = "benchmarks.cuda_operator_profile"
ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_cases(suite: str, dtypes: list[str], fp8: bool) -> list[dict]:
    """DeepSeek dimensions are standalone floating-point probes, not INT4 serving."""
    rows = []
    shapes = []
    if suite in ("qwen", "all"):
        shapes += [("qwen", role, m, n, k) for m in (1, 8, 512)
                   for role, n, k in (("qkv", 4096, 1024), ("o", 1024, 2048),
                                      ("gate_up", 6144, 1024), ("down", 1024, 3072))]
    if suite in ("deepseek", "all"):
        shapes += [("deepseek", role, m, n, k) for m in (1, 8, 128)
                   for role, n, k in (("q_proj", 3072, 2048),
                                      ("expert_gate_up", 2816, 2048),
                                      ("expert_down", 2048, 1408))]
    for dtype in dtypes:
        for model, role, m, n, k in shapes:
            counter = ((model == "qwen" and dtype == "bf16" and
                        ((m == 1 and role == "qkv") or m == 8 or (m == 512 and role == "gate_up")))
                       or (model == "qwen" and dtype == "fp16" and m == 512 and role == "gate_up")
                       or (model == "deepseek" and dtype == "bf16" and m == 8 and role == "expert_gate_up"))
            rows.append(dict(id=f"{model}_{dtype}_m{m}_{role}", model=model, role=role,
                             dtype=dtype, shape=[m, n, k], counter=counter,
                             scope="deployed Qwen shape" if model == "qwen" else "standalone float shape; not deployed INT4"))
    if fp8 and suite in ("qwen", "all"):
        for role, n in (("qkv", 4096), ("gate_up", 6144)):
            rows.append(dict(id=f"qwen_fp8_m128_{role}", model="qwen", role=role,
                             dtype="fp8", shape=[128, n, 1024], counter=role == "gate_up",
                             scope="experimental FP8 pack_rows + GEMM; not default backend"))
    return rows


def resolve_choice(case: dict, device: str) -> dict:
    """Mirror CudaLinear dispatch without calling its allocation-heavy adapter."""
    m, n, k = case["shape"]
    if case["model"] == "qwen":
        filename = {"bf16": "cuda_decode_sm120.json", "fp16": "cuda_decode_sm120_fp16.json",
                    "fp8": "cuda_fp8_sm120.json"}[case["dtype"]]
        path = ROOT / "nanovllm/layers" / filename
        bundled = json.loads(path.read_text()) if path.exists() else {}
        if device == bundled.get("device"):
            selected = bundled["selectors"].get(f"{m},{n},{k}")
            if selected:
                return selected
    if case["dtype"] == "fp8":
        return dict(mode="fp8", config=3)
    return dict(mode="gemv" if m <= 4 else "gemm", config=0 if m <= 4 else 10)


@contextmanager
def nvtx(label: str):
    import torch
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class Workload:
    """Preallocate inputs/outputs and pack weights outside every measured range."""

    def __init__(self, case: dict, regime: str, seed: int, choice: dict | None = None) -> None:
        import torch
        from nanovllm.layers.cuda_linear import load_extension, load_decode_extension
        self.case, self.regime = case, regime
        self.choice = choice or resolve_choice(case, torch.cuda.get_device_name())
        m, n, k = case["shape"]
        torch.manual_seed(seed)
        dtype = torch.float16 if case["dtype"] == "fp16" else torch.bfloat16
        self.x = torch.randn(m, k, device="cuda", dtype=dtype)
        count = 1 if regime == "resident" else 28
        self.weights = [torch.randn(n, k, device="cuda", dtype=dtype) * .02 for _ in range(count)]
        self.outputs = [torch.empty(m, n, device="cuda", dtype=dtype) for _ in range(count)]
        self.functions = []
        self.packed = []
        self.scales = []
        mode, config = self.choice["mode"], self.choice["config"]
        x = self.x
        if mode == "fp8":
            ext = load_extension(True)
            self.activation = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            self.row_scales = torch.empty(m, device=x.device, dtype=torch.float32)
            a, scales = self.activation, self.row_scales
            for w, out in zip(self.weights, self.outputs):
                scale = max(w.abs().max().item() / 448, 1e-12)
                packed = (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                self.packed.append(packed)
                self.scales.append(scale)
                def run(packed=packed, out=out, scale=scale):
                    ext.pack_rows_out(x, a, scales)
                    ext.gemm_rows_out(a, packed, out, scales, config, scale)
                self.functions.append(run)
        else:
            ext = load_decode_extension() if mode == "decode" else load_extension()
            for w, out in zip(self.weights, self.outputs):
                if mode == "decode":
                    entry = ext.decode_no_swizzle_out if self.choice.get("no_swizzle") else ext.decode_out
                    self.functions.append(lambda w=w, out=out: entry(x, w, out, config))
                elif mode == "gemv":
                    self.functions.append(lambda w=w, out=out: ext.linear_gemv_out(x, w, out))
                else:
                    packed = w.T.contiguous()
                    self.packed.append(packed)
                    self.functions.append(lambda packed=packed, out=out: ext.gemm_out(x, packed, out, config))
        self.operations = 32 if regime == "resident" else 28

    def run(self) -> None:
        if self.regime == "resident":
            for _ in range(self.operations):
                self.functions[0]()
        else:
            for fn in self.functions:
                fn()

    def validate(self) -> dict:
        import torch
        m = self.case["shape"][0]
        rows = sorted(set(range(min(4, m))) | set(range(max(0, m-4), m)))
        idx = torch.tensor(rows, device=self.x.device)
        errors = []
        quantization_errors = []
        for i in sorted({0, len(self.weights)-1}):
            self.functions[i]()
            original = self.x[idx].double() @ self.weights[i].double().T
            expected = original
            if self.choice["mode"] == "fp8":
                expected = (self.activation[idx].double() @ self.packed[i].double().T)
                expected *= self.row_scales[idx].double()[:, None] * self.scales[i]
                quantization_errors.append(((expected-original).norm()/original.norm().clamp_min(1e-12)).item())
            actual = self.outputs[i][idx].double()
            error = ((actual-expected).norm()/expected.norm().clamp_min(1e-12)).item()
            limit = .001 if self.case["dtype"] == "fp16" else .006
            if not torch.isfinite(self.outputs[i]).all() or error > limit:
                raise AssertionError((self.case["id"], i, error, limit))
            errors.append(error)
        return dict(relative_l2_max=max(errors), checked_rows=rows,
                    checked_weights=sorted({0, len(self.weights)-1}),
                    fp8_quantization_l2_max=max(quantization_errors, default=None))


def environment() -> dict:
    import torch
    def command(cmd):
        run = subprocess.run(cmd, capture_output=True, text=True)
        return dict(returncode=run.returncode, output=(run.stdout + run.stderr).strip())
    source_files = [ROOT / "nanovllm/layers/csrc" / name for name in
                    ("cuda_gemm.cu", "cuda_decode.cu", "cuda_fp8_gemm.cu")]
    return dict(date_utc=datetime.now(timezone.utc).isoformat(), device=torch.cuda.get_device_name(),
                sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
                torch=torch.__version__, cuda=torch.version.cuda,
                git_head=command(["git", "rev-parse", "HEAD"]), git_status=command(["git", "status", "--short"]),
                gpu=command(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.free", "--format=csv,noheader"]),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})


def timing(args: argparse.Namespace) -> None:
    import torch
    path = args.output / "timing.json"
    if path.exists():
        if not args.resume:
            raise ValueError("preserve timing.json; use a fresh output or --resume")
        data = json.loads(path.read_text())
        current = environment()
        if any(data["environment"][key] != current[key] for key in ("source_sha256", "device", "sm_count", "torch", "cuda")):
            raise ValueError("CUDA sources/device/software changed since timing started; use a fresh output")
    else:
        data = dict(environment=environment(), method="custom kernels only; 3 seeds; randomized resident/rotating28 order; 9 graph rounds; preallocated outputs", rows=[])
    completed = {(r["case"]["id"], r["seed"], r["regime"]) for r in data["rows"]}
    if len(completed) != len(data["rows"]):
        raise ValueError("duplicate independent timing rows")
    rng = random.Random(711)
    for case in selected_cases(args):
        for seed in (17, 42, 99):
            regimes = ["resident", "rotating28"]
            rng.shuffle(regimes)
            for regime in regimes:
                if (case["id"], seed, regime) in completed:
                    continue
                workload = Workload(case, regime, seed)
                check = workload.validate()
                for _ in range(3):
                    workload.run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    workload.run()
                graph.replay()
                torch.cuda.synchronize()
                samples = []
                for _ in range(9):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(); graph.replay(); end.record(); end.synchronize()
                    samples.append(begin.elapsed_time(end)*1000/workload.operations)
                row = dict(case=case, choice=workload.choice, regime=regime, seed=seed,
                           samples_us=samples, median_us=statistics.median(samples), accuracy=check)
                data["rows"].append(row)
                write_json(path, data)
                print("TIME", case["id"], regime, seed, round(row["median_us"], 3), flush=True)
                del graph, workload
                gc.collect()


def trace(args: argparse.Namespace) -> None:
    import torch
    rows = []
    for case in selected_cases(args):
        for regime in ("resident", "rotating28"):
            workload = Workload(case, regime, 42)
            check = workload.validate()
            for _ in range(3):
                workload.run()
            torch.cuda.synchronize()
            graph = None
            if args.execution == "graph":
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    workload.run()
                graph.replay(); torch.cuda.synchronize()
            with nvtx(f"op_capture/{case['id']}/{regime}/{args.execution}"):
                for _ in range(3):
                    if graph is not None:
                        graph.replay()
                    else:
                        workload.run()
                torch.cuda.synchronize()
            rows.append(dict(case=case, regime=regime, choice=workload.choice, accuracy=check,
                             invocations=3*workload.operations))
            print("TRACE", case["id"], regime, flush=True)
            del graph, workload
            gc.collect()
    write_json(args.output / f"nsys_{args.execution}.meta.json", dict(rows=rows, environment=environment()))


def probe(args: argparse.Namespace) -> None:
    import torch
    cases = selected_cases(args)
    if len(cases) != 1:
        raise ValueError("probe requires exactly one --case")
    workload = Workload(cases[0], "resident", 42)
    check = workload.validate()
    for _ in range(10):
        workload.functions[0]()
    torch.cuda.synchronize()
    if args.meta:
        write_json(args.meta, dict(case=cases[0], choice=workload.choice, accuracy=check))
    torch.cuda.profiler.start()
    with nvtx("operator_probe"):
        workload.functions[0]()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def selected_cases(args: argparse.Namespace) -> list[dict]:
    cases = make_cases(args.suite, args.dtypes, args.fp8)
    if args.case:
        selected = [c for c in cases if c["id"] in args.case]
        if {c["id"] for c in selected} != set(args.case):
            raise ValueError(f"unknown --case: {set(args.case)-{c['id'] for c in selected}}")
        return selected
    return cases


def common_args(args: argparse.Namespace) -> list[str]:
    result = ["--suite", args.suite, "--dtypes", *args.dtypes,
              "--output", str(args.output), "--fp8" if args.fp8 else "--no-fp8"]
    return result


def run_logged(command: list[str], prefix: Path) -> None:
    """Save failures as well as successful exact commands; never hide stderr."""
    run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    prefix.with_suffix(".log").write_text(run.stdout, encoding="utf-8")
    write_json(prefix.with_suffix(".status.json"), dict(command=command, returncode=run.returncode))
    if run.returncode:
        raise RuntimeError(run.stdout[-3500:])


def collect_nsys(args: argparse.Namespace) -> None:
    from benchmarks.cuda_operator_profile_report import analyze_nsys
    for execution in ("eager", "graph"):
        prefix = args.output / f"nsys_{execution}"
        if prefix.with_suffix(".nsys-rep").exists():
            if not args.resume:
                raise ValueError("preserve existing Nsight Systems report; use --resume")
        else:
            command = [args.nsys, "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
                       "--cpuctxsw=none", "--cuda-graph-trace=node", "-o", str(prefix),
                       sys.executable, "-m", MODULE, "--mode", "trace", "--execution", execution,
                       *common_args(args)]
            if args.case:
                command += ["--case", *args.case]
            print("NSYS", execution, flush=True)
            run_logged(command, prefix)
        sqlite = prefix.with_suffix(".sqlite")
        if not sqlite.exists():
            run_logged([args.nsys, "export", "--type=sqlite", "-o", str(sqlite),
                        str(prefix.with_suffix(".nsys-rep"))], args.output / f"export_{execution}")
        write_json(args.output / f"nsys_{execution}.analysis.json", analyze_nsys(sqlite))


def collect_ncu(args: argparse.Namespace) -> None:
    from benchmarks.cuda_gemm_profile import select_ncu
    from benchmarks.cuda_operator_profile_report import read_ncu
    tool = select_ncu(args.ncu)
    folder = args.output / "ncu"
    folder.mkdir(exist_ok=True)
    cases = [c for c in selected_cases(args) if args.case or c["counter"]]
    for case in cases:
        for cache in ("none", "all"):
            for rep in range(args.counter_repeats):
                prefix = folder / f"{case['id']}_{cache}_r{rep}"
                native = list(folder.glob(prefix.name + ".ncu-rep*"))
                status = prefix.with_suffix(".status.json")
                if native:
                    if not args.resume or len(native) != 1 or not status.exists() or json.loads(status.read_text())["returncode"]:
                        raise ValueError(f"preserve existing NCU reports: {prefix}")
                else:
                    command = [tool, "--target-processes", "all", "--profile-from-start", "off",
                               "--replay-mode", "kernel", "--clock-control", "none", "--cache-control", cache]
                    for section in ("LaunchStats", "Occupancy", "SpeedOfLight", "SchedulerStats",
                                    "WarpStateStats", "MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Tables", "ComputeWorkloadAnalysis"):
                        command += ["--section", section]
                    command += ["--metrics", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
                                "-o", str(prefix), sys.executable, "-m", MODULE, "--mode", "probe",
                                *common_args(args), "--case", case["id"], "--meta", str(prefix)+".meta.json"]
                    print("NCU", prefix.name, flush=True)
                    run_logged(command, prefix)
                    native = list(folder.glob(prefix.name + ".ncu-rep*"))
                if len(native) != 1:
                    raise RuntimeError(f"missing or ambiguous native report: {prefix}")
                csv = prefix.with_suffix(".csv")
                if not csv.exists():
                    run = subprocess.run([tool, "--import", str(native[0]), "--page", "raw", "--csv"], capture_output=True, text=True, check=True)
                    csv.write_text(run.stdout, encoding="utf-8")
                rows = read_ncu(csv)
                write_json(prefix.with_suffix(".analysis.json"), dict(case=case, cache=cache, repeat=rep,
                           native=str(native[0]), kernels=rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "timing", "nsys", "ncu", "trace", "probe", "report"), default="all")
    parser.add_argument("--output", type=Path, default=Path("profiles") / ("cuda_operator_"+datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--suite", choices=("qwen", "deepseek", "all"), default="all")
    parser.add_argument("--dtypes", nargs="+", choices=("bf16", "fp16"), default=["bf16", "fp16"])
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--case", nargs="+")
    parser.add_argument("--counter-repeats", type=int, default=3)
    parser.add_argument("--nsys", default="/usr/local/bin/nsys")
    parser.add_argument("--ncu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    parser.add_argument("--meta", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.counter_repeats < 1:
        parser.error("counter-repeats must be positive")
    if args.mode in ("all", "timing", "nsys", "ncu"):
        sources = [Path(__file__), ROOT / "benchmarks/cuda_operator_profile_report.py",
                   *sorted((ROOT / "nanovllm/layers").glob("cuda_*sm120*.json")),
                   *sorted((ROOT / "nanovllm/layers/csrc").glob("cuda_*.cu"))]
        manifest = dict(cases=selected_cases(args), counter_repeats=args.counter_repeats,
                        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        manifest_path = args.output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("measurement configuration/sources changed; use a fresh output")
        else:
            write_json(manifest_path, manifest)
    if args.mode in ("all", "timing"):
        timing(args)
    if args.mode in ("all", "nsys"):
        collect_nsys(args)
    if args.mode in ("all", "ncu"):
        collect_ncu(args)
    if args.mode == "trace":
        trace(args)
    if args.mode == "probe":
        probe(args)
    if args.mode in ("all", "report"):
        from benchmarks.cuda_operator_profile_report import generate
        generate(args.output)


if __name__ == "__main__":
    main()
