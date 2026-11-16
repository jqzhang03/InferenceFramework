"""FP16/BF16 handwritten CUDA GEMM vs direct cuBLAS GemmEx.

Run in WSL: bash benchmarks/run_cuda_gemm.sh --output results/cuda_gemm.json
Tuning and final measurements use different inputs. Both backends write into
preallocated buffers; CUDA Graph replay removes Python/launch starvation. This
is a warm-cache kernel benchmark, not end-to-end serving performance.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
from types import ModuleType
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import torch


CONFIGS = [
    (64, 128, 32, 32, 64, 3), (64, 128, 64, 32, 64, 3),
    (128, 128, 32, 32, 64, 3), (128, 128, 64, 32, 64, 3),
    (64, 64, 32, 32, 32, 3), (64, 128, 32, 32, 64, 4),
    (128, 128, 32, 32, 64, 4), (128, 256, 32, 64, 64, 3),
    (64, 256, 32, 32, 64, 3), (128, 64, 32, 32, 32, 3),
    (64, 128, 32, 32, 64, 2), (64, 64, 32, 32, 32, 2),
    (128, 64, 32, 32, 32, 2), (128, 128, 32, 32, 64, 2),
    (128, 64, 32, 32, 32, 4), (64, 128, 32, 16, 64, 3),
    (64, 128, 32, 64, 32, 3), (64, 128, 32, 64, 32, 2),
    (64, 64, 32, 64, 32, 3), (64, 64, 32, 64, 32, 4),
    (64, 128, 32, 16, 64, 4),
]


def load_extension() -> ModuleType:
    """Build only this standalone extension, without importing the engine."""
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError("CUDA GEMM requires SM80 or newer")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    return load(
        name="nanovllm_cuda_gemm",
        sources=[str(Path(__file__).with_name("cuda_gemm.cu"))],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        extra_ldflags=["-lcublas"],
        verbose=os.environ.get("GEMM_BUILD_VERBOSE") == "1",
    )


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected.double()).norm()
            / expected.double().norm().clamp_min(1e-30)).item()


def accuracy(a: torch.Tensor, b: torch.Tensor, custom: torch.Tensor,
             baseline: torch.Tensor, source_a: torch.Tensor | None = None,
             source_b: torch.Tensor | None = None) -> dict:
    """FP64 reference uses exactly the represented FP16/BF16 inputs.

    Small matrices: full output. Large matrices: reproducible 64x64 output
    submatrix, using all K elements in every sampled dot product.
    """
    m, k = a.shape
    n = b.shape[1]
    generator = torch.Generator().manual_seed(1701)
    rows = torch.randperm(m, generator=generator)[:min(m, 64)].to(a.device)
    cols = torch.randperm(n, generator=generator)[:min(n, 64)].to(a.device)
    full = max(m, n, k) <= 512
    aa = a.double() if full else a[rows].double()
    bb = b.double() if full else b[:, cols].double()
    reference = aa @ bb
    result = {"reference": "fp64 on rounded input", "full_reference": full,
              "reference_shape": list(reference.shape)}
    limit = 8e-4 if a.dtype == torch.float16 else 6e-3
    for name, output in (("cuda", custom), ("cublas", baseline)):
        sampled = output if full else output[rows[:, None], cols[None, :]]
        error = sampled.double() - reference
        rel = relative_l2(sampled, reference)
        if not torch.isfinite(output).all() or rel > limit:
            raise AssertionError(f"{name} failed accuracy: relative L2 {rel} > {limit}")
        result[name] = {"relative_l2": rel, "max_abs": error.abs().max().item(),
                        "rmse": error.square().mean().sqrt().item()}
    result["cuda_vs_cublas_relative_l2_full"] = relative_l2(custom, baseline)
    if result["cuda_vs_cublas_relative_l2_full"] > 2 * limit:
        raise AssertionError("custom CUDA and cuBLAS disagree on full output")
    if source_a is not None and source_b is not None:
        sa = source_a.double() if full else source_a[rows].double()
        sb = source_b.double() if full else source_b[:, cols].double()
        source_reference = sa @ sb
        result["including_input_rounding"] = {
            "reference": "FP64 GEMM of common original FP32 inputs",
            "cuda_relative_l2": relative_l2(custom if full else custom[rows[:,None],cols[None,:]], source_reference),
            "cublas_relative_l2": relative_l2(baseline if full else baseline[rows[:,None],cols[None,:]], source_reference),
        }
    return result


def capture(fn: Callable[[], object], iterations: int) -> torch.cuda.CUDAGraph:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            fn()
    return graph


def elapsed_graph(graph: torch.cuda.CUDAGraph, iterations: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def tune(ext: ModuleType, a: torch.Tensor, b: torch.Tensor, candidates: list[int]) -> tuple[int, dict]:
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    baseline = torch.empty_like(out)
    ext.cublas_out(a, b, baseline)
    timings = {}
    for config in candidates:
        fn = lambda: ext.gemm_out(a, b, out, config)
        fn()
        # Correctness gates every candidate before any performance selection.
        accuracy(a, b, out, baseline)
        graph = capture(fn, 20)
        graph.replay()
        torch.cuda.synchronize()
        timings[config] = statistics.median(elapsed_graph(graph, 20) for _ in range(3))
        del graph
    return min(timings, key=timings.get), timings


def correctness_checks(ext: ModuleType) -> dict:
    """Tail tiles, fallback, zeros, cancellation, streams, graph and API guards."""
    cases = []
    for dtype in (torch.float16, torch.bfloat16):
        for m, n, k in ((1, 8, 8), (17, 136, 40), (129, 264, 72),
                        (7, 13, 19), (3, 8, 0), (33, 64, 256)):
            a = torch.randn(m, k, device="cuda", dtype=dtype)
            b = torch.randn(k, n, device="cuda", dtype=dtype)
            out = torch.empty(m, n, device="cuda", dtype=dtype)
            ref = a.double() @ b.double()
            for config in range(len(CONFIGS)):
                ext.gemm_out(a, b, out, config)
                err = relative_l2(out, ref)
                if not torch.isfinite(out).all() or err > (8e-4 if dtype == torch.float16 else 6e-3):
                    raise AssertionError(f"{dtype}, {(m,n,k)}, config={config}: {err}")
            cases.append({"dtype": str(dtype), "shape_mnk": [m,n,k], "configs": len(CONFIGS)})
        for pattern in ("zeros", "identity", "cancellation", "dynamic_range"):
            a = torch.randn(64, 128, device="cuda", dtype=dtype)
            b = torch.randn(128, 128, device="cuda", dtype=dtype)
            if pattern == "zeros":
                a.zero_()
            elif pattern == "identity":
                b.copy_(torch.eye(128, device="cuda", dtype=dtype))
            elif pattern == "cancellation":
                a[:, 1::2] = -a[:, ::2]
                b[1::2] = b[::2]
            else:
                scale = torch.logspace(-2, 2, 128, device="cuda")
                a = (a.float() * scale).to(dtype)
                b = (b.float() / scale[:, None]).to(dtype)
            out, baseline = torch.empty(64,128,device="cuda",dtype=dtype), torch.empty(64,128,device="cuda",dtype=dtype)
            ext.cublas_out(a,b,baseline)
            for config in range(len(CONFIGS)):
                ext.gemm_out(a,b,out,config)
                if pattern == "cancellation":
                    if out.abs().max() > 1e-4:
                        raise AssertionError("cancellation failed")
                else:
                    accuracy(a,b,out,baseline)
            cases.append({"dtype": str(dtype), "pattern": pattern, "configs": len(CONFIGS)})
        # Non-default stream producers and consumers; capture both entry points.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            a = torch.randn(64,64,device="cuda",dtype=dtype)
            b = torch.randn(64,64,device="cuda",dtype=dtype)
            c, d = torch.empty_like(a), torch.empty_like(a)
            graph = capture(lambda: (ext.gemm_out(a,b,c,0), ext.cublas_out(a,b,d)), 2)
            graph.replay()
            ok = relative_l2(c,d) < (2e-3 if dtype == torch.float16 else 1e-2)
        stream.synchronize()
        if not ok:
            raise AssertionError("non-default stream / CUDA graph failed")
        # Empty outputs should be a safe no-op.
        ext.gemm_out(torch.empty(0,8,device="cuda",dtype=dtype),
                     torch.empty(8,8,device="cuda",dtype=dtype),
                     torch.empty(0,8,device="cuda",dtype=dtype),0)
    a = torch.randn(16,16,device="cuda",dtype=torch.float16)
    invalid = [(a.t(),a,torch.empty_like(a),0), (a,a,a,0),
               (a,a,torch.empty_like(a),-1), (a.float(),a,torch.empty_like(a),0),
               (a[:,:8].contiguous(),a,torch.empty_like(a),0)]
    for args in invalid:
        try:
            ext.gemm_out(*args)
        except RuntimeError:
            continue
        raise AssertionError("invalid input was accepted")
    torch.cuda.synchronize()
    return {"passed": True, "cases": cases, "invalid_input_checks": len(invalid),
            "stream_graph_and_empty_checks": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="64,4096,4096;256,4096,4096;1024,1024,1024;4096,4096,4096;128,11008,4096;512,4096,11008;8192,4096,4096",
                        help="semicolon-separated M,N,K triples")
    parser.add_argument("--configs", default=",".join(map(str,range(len(CONFIGS)))))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tuning-report", type=Path,
                        help="freeze per-dtype/shape configs from an earlier tuning JSON")
    parser.add_argument("--output", type=Path, default=Path("results/cuda_gemm.json"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.repeats <= 0:
        parser.error("iterations/repeats must be positive")
    ext = load_extension()
    print(f"GPU: {torch.cuda.get_device_name()} / torch {torch.__version__}", flush=True)
    torch.manual_seed(args.seed)
    checks = correctness_checks(ext)
    print("Correctness / boundaries / stream / graph / API guards: PASS", flush=True)
    if args.check_only:
        return
    shapes = [tuple(map(int, entry.split(","))) for entry in args.shapes.split(";")]
    candidates = list(map(int,args.configs.split(",")))
    if not candidates or any(c < 0 or c >= len(CONFIGS) for c in candidates):
        parser.error("config outside supported range")
    fixed_configs = {}
    if args.tuning_report:
        tuning_report = json.loads(args.tuning_report.read_text(encoding="utf-8"))
        if tuning_report["configs"] != [list(config) for config in CONFIGS]:
            raise ValueError("tuning report configuration definitions differ")
        fixed_configs = {(r["dtype"],tuple(r["shape_mnk"])):r["config"] for r in tuning_report["results"]}
    props = torch.cuda.get_device_properties(0)
    git_head = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    report = {"date_utc": datetime.now(timezone.utc).isoformat(), "git_head": git_head,
              "git_worktree_dirty": bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip()),
              "gpu": props.name, "sm": f"{props.major}.{props.minor}",
              "sm_count": props.multi_processor_count, "torch": torch.__version__,
              "cuda": torch.version.cuda, "platform": platform.platform(),
              "nvcc": subprocess.check_output(["nvcc","--version"],text=True).strip(),
              "nvidia_smi": subprocess.check_output(["nvidia-smi","--query-gpu=driver_version,temperature.gpu,clocks.sm,power.draw","--format=csv,noheader"],text=True).strip(),
              "timing": "CUDA events around graph replay; preallocated output; warm cache; tuning excluded",
              "cublas": "direct cublasGemmEx; CUBLAS_COMPUTE_32F; reduced precision reduction disabled",
              "config_fields": ["BM","BN","BK","WM","WN","stages"],
              "configs": CONFIGS, "seed": args.seed, "iterations": args.iterations,
              "tuning_report": str(args.tuning_report) if args.tuning_report else None,
              "repeats": args.repeats, "checks": checks, "results": []}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rng = random.Random(args.seed)
    for dtype in (torch.float16, torch.bfloat16):
        for m,n,k in shapes:
            if min(m,n,k) <= 0:
                raise ValueError("benchmark shapes must be positive")
            torch.manual_seed(args.seed + 1)
            a = torch.randn(m,k,device="cuda",dtype=torch.float32).to(dtype)
            b = torch.randn(k,n,device="cuda",dtype=torch.float32).to(dtype)
            if args.tuning_report:
                best, tuning = fixed_configs[(str(dtype),(m,n,k))], {}
            else:
                best, tuning = tune(ext,a,b,candidates)
            # New inputs for final correctness and final paired timing.
            torch.manual_seed(args.seed)
            source_a = torch.randn(m,k,device="cuda",dtype=torch.float32)
            source_b = torch.randn(k,n,device="cuda",dtype=torch.float32)
            a.copy_(source_a); b.copy_(source_b)
            c = torch.empty(m,n,device="cuda",dtype=dtype)
            d = torch.empty_like(c)
            ext.gemm_out(a,b,c,best); ext.cublas_out(a,b,d)
            errors = accuracy(a,b,c,d,source_a,source_b)
            graphs = {"cuda":capture(lambda:ext.gemm_out(a,b,c,best),args.iterations),
                      "cublas":capture(lambda:ext.cublas_out(a,b,d),args.iterations)}
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            times = {key:[] for key in graphs}
            for _ in range(args.repeats):
                order = list(graphs)
                rng.shuffle(order)
                for key in order:
                    times[key].append(elapsed_graph(graphs[key],args.iterations))
            us = {key:statistics.median(value) for key,value in times.items()}
            row = {"dtype":str(dtype),"shape_mnk":[m,n,k],"config":best,
                   "tuning_us":tuning,"accuracy":errors,"time_us":us,"samples_us":times,
                   "tflops":{key:2*m*n*k/(value*1e6) for key,value in us.items()},
                   "cuda_over_cublas":us["cublas"]/us["cuda"]}
            report["results"].append(row)
            args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
            print(f"{str(dtype):15s} {m:5d}x{n:5d}x{k:5d} cfg={best} "
                  f"CUDA={us['cuda']:.2f} us ({row['tflops']['cuda']:.2f} TF/s) "
                  f"cuBLAS={us['cublas']:.2f} us ratio={row['cuda_over_cublas']:.1%} "
                  f"relL2={errors['cuda']['relative_l2']:.3g}",flush=True)
            del graphs,a,b,c,d,source_a,source_b
    print(f"Saved {args.output}",flush=True)


if __name__ == "__main__":
    main()
