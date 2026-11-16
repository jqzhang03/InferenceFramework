"""Native SM120 FP8 E4M3 GEMM versus tuned cuBLASLt on identical packed inputs."""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import torch


CONFIGS=[(64,64,64,32,32,2),(64,64,64,32,32,3),(64,64,64,32,32,4),
         (64,128,64,32,64,2),(64,128,64,32,64,3),(64,128,64,32,64,4),
         (128,128,64,32,64,2),(128,128,64,32,64,3),(128,128,64,32,64,4),
         (64,128,128,32,64,2),(64,128,128,32,64,3),(128,128,128,32,64,2),
         (128,256,64,64,64,2),(128,256,64,64,64,3),
         (64,128,64,64,32,2),(64,128,64,64,32,3),(32,64,64,16,32,3),(64,64,128,32,32,3)]
CONFIGS=[(*cfg,1) for cfg in CONFIGS]+[
    (64,64,64,32,32,2,8),(64,128,64,32,64,2,8),(128,128,64,32,64,2,8),
    (64,128,128,32,64,2,8),(128,128,128,32,64,2,8),
    (128,64,64,32,32,2,8),(128,64,64,32,32,3,8),(128,64,128,32,32,2,8),
    (128,64,64,64,32,2,8),(32,64,64,16,32,3,8)]


def load_extension() -> ModuleType:
    from torch.utils.cpp_extension import load

    if torch.cuda.get_device_capability()!=(12,0):
        raise RuntimeError("FP8 extension targets RTX Blackwell SM120 only")
    os.environ.setdefault("CXX","/usr/bin/g++-13")
    previous_arch=os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"]="12.0a"
    try:
        return load(name="nanovllm_cuda_fp8_gemm",sources=[str(Path(__file__).with_name("cuda_fp8_gemm.cu"))],
                    extra_cuda_cflags=["-O3","-lineinfo","--ptxas-options=-v","-ccbin=/usr/bin/g++-13"],
                    extra_ldflags=["-lcublasLt"],verbose=os.environ.get("GEMM_BUILD_VERBOSE")=="1")
    finally:
        if previous_arch is None: os.environ.pop("TORCH_CUDA_ARCH_LIST",None)
        else: os.environ["TORCH_CUDA_ARCH_LIST"]=previous_arch


def pack(x: torch.Tensor) -> tuple[torch.Tensor,float]:
    scale=max(x.abs().max().item()/448.0,1e-12) if x.numel() else 1.0
    return (x/scale).clamp(-448,448).to(torch.float8_e4m3fn),scale


def relative_l2(a: torch.Tensor,b: torch.Tensor) -> float:
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30)).item()


def accuracy(a: torch.Tensor,w: torch.Tensor,out: torch.Tensor,alpha: float,
             source_a: torch.Tensor | None=None,source_w: torch.Tensor | None=None) -> dict:
    m,k=a.shape;n=w.shape[0]
    af,wf=a.float(),w.float()
    full=m*n*k<=8_000_000
    rows=torch.arange(m,device="cuda") if full else torch.linspace(0,m-1,min(64,m),device="cuda").long()
    cols=torch.arange(n,device="cuda") if full else torch.linspace(0,n-1,min(64,n),device="cuda").long()
    ref=(af[rows].double()@wf[cols].double().T)*alpha
    actual=out[rows[:,None],cols[None,:]].double()
    error=relative_l2(actual,ref)
    if not torch.isfinite(out).all() or error>4e-3:
        raise AssertionError(f"FP8 arithmetic error {error}; limit 0.004")
    result={"arithmetic_relative_l2":error,"max_abs":(actual-ref).abs().max().item(),
            "reference":"FP64 of decoded FP8 inputs, including alpha; full" if full else "FP64 of decoded FP8 inputs, 64x64 sampled output, all K"}
    if source_a is not None:
        source_ref=source_a[rows].double()@source_w[cols].double().T
        result["including_quantization_relative_l2"]=relative_l2(actual,source_ref)
    return result


def correctness_checks(ext: ModuleType) -> dict:
    torch.manual_seed(123)
    cases=0
    for m,n,k in ((16,16,16),(31,47,80),(64,64,64),(65,129,144),(128,256,256),(1,17,32),(577,145,80)):
        x=torch.randn(m,k,device="cuda");y=torch.randn(n,k,device="cuda")
        a,sa=pack(x);w,sw=pack(y);out=torch.empty(m,n,device="cuda",dtype=torch.bfloat16)
        for config in range(len(CONFIGS)):
            ext.gemm_out(a,w,out,config,sa*sw)
            accuracy(a,w,out,sa*sw)
            cases+=1
    # Exact identities and cancellation catch fragment permutation and invalid tails.
    for value in (0.,1.,-1.):
        a=torch.full((33,48),value,device="cuda").to(torch.float8_e4m3fn)
        w=torch.ones(17,48,device="cuda").to(torch.float8_e4m3fn)
        out=torch.empty(33,17,device="cuda",dtype=torch.bfloat16)
        for cfg in range(len(CONFIGS)):
            ext.gemm_out(a,w,out,cfg,0.5)
            torch.testing.assert_close(out.float(),torch.full((33,17),value*24,device="cuda"),rtol=0,atol=0)
            cases+=1
    # E5M2 is an additional native-kernel correctness check, not a measured Lt baseline.
    a=torch.randn(33,80,device="cuda").to(torch.float8_e5m2)
    w=torch.randn(19,80,device="cuda").to(torch.float8_e5m2)
    out=torch.empty(33,19,device="cuda",dtype=torch.bfloat16)
    for config in range(len(CONFIGS)):
        ext.gemm_out(a,w,out,config,1.)
        accuracy(a,w,out,1.)
        cases+=1
    a=torch.eye(32,device="cuda").to(torch.float8_e4m3fn)
    w=torch.randn(32,32,device="cuda").to(torch.float8_e4m3fn)
    out=torch.empty(32,32,device="cuda",dtype=torch.bfloat16)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph=torch.cuda.CUDAGraph()
        ext.gemm_out(a,w,out,0,1.)
        with torch.cuda.graph(graph,stream=stream):
            ext.gemm_out(a,w,out,0,1.)
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(out,w.float().T.to(torch.bfloat16),rtol=0,atol=0)
    # K=0 and empty M/N are explicit safe paths.
    zero=torch.ones(3,4,device="cuda",dtype=torch.bfloat16)
    ext.gemm_out(torch.empty(3,0,device="cuda",dtype=a.dtype),torch.empty(4,0,device="cuda",dtype=a.dtype),zero,0,1.)
    assert not zero.count_nonzero()
    ext.gemm_out(a[:0],w,torch.empty(0,32,device="cuda",dtype=torch.bfloat16),0,1.)
    invalid=[(a.T,w,out,0,1.),(a,w,out,-1,1.),(a,w,out,0,float("nan")),
             (a,w,out.float(),0,1.),(a[:,:17].contiguous(),w[:,:17].contiguous(),out,0,1.)]
    for args in invalid:
        try: ext.gemm_out(*args)
        except RuntimeError: continue
        raise AssertionError("invalid input accepted")
    return {"passed":True,"config_shape_pattern_cases":cases,"identity_stream_graph":True,"invalid_cases":len(invalid)}


def benchmark(args: argparse.Namespace,ext: ModuleType) -> None:
    from benchmarks.cuda_gemm_bench import capture,elapsed_graph

    shapes=[tuple(map(int,s.split(','))) for s in args.shapes.replace('+',';').split(';')]
    if any(len(s)!=3 or min(s)<=0 or s[2]%16 for s in shapes):
        raise ValueError("positive M,N,K; K divisible by 16 required")
    root=args.output;root.parent.mkdir(parents=True,exist_ok=True)
    report={"date_utc":datetime.now(timezone.utc).isoformat(),"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
            "configs":CONFIGS,"layout":"A[M,K], W[N,K], C[M,N], row major; E4M3/E4M3 -> BF16; FP32 accumulation",
            "method":"prepacked common inputs; no quantization/transpose in timings; 100-node graphs; paired randomized rounds; cuBLASLt heuristics tuned outside validation", "results":[]}
    frozen={}
    if args.tuning_report:
        prior=json.loads(args.tuning_report.read_text())
        if prior["configs"]!=[list(c) for c in CONFIGS]: raise ValueError("config definitions changed")
        frozen={tuple(r["shape_mnk"]):r for r in prior["results"]}
    rng=random.Random(args.seed)
    # Warm GPU/clocks on a separate medium shape, outside all measurement inputs.
    warm_a=torch.randn(1024,1024,device="cuda").to(torch.float8_e4m3fn)
    warm_c=torch.empty(1024,1024,device="cuda",dtype=torch.bfloat16)
    warm_graph=capture(lambda:ext.gemm_out(warm_a,warm_a,warm_c,4,1.),100)
    for _ in range(20): warm_graph.replay()
    torch.cuda.synchronize()
    for m,n,k in shapes:
        # Tuning uses a separate seed and buffers. Freeze both selectors for validation.
        torch.manual_seed(20261116)
        a,_=pack(torch.randn(m,k,device="cuda"));w,_=pack(torch.randn(n,k,device="cuda"))
        c=torch.empty(m,n,device="cuda",dtype=torch.bfloat16)
        plan=ext.Plan(a,w,c)
        if frozen:
            config=frozen[m,n,k]["config"];algo=frozen[m,n,k]["cublas_algorithm"]
            signature=frozen[m,n,k]["cublas_algorithm_info"]
            matches=[i for i in range(plan.count()) if dict(plan.algorithm_info(i))==signature]
            if not matches: raise RuntimeError("frozen cuBLASLt algorithm unavailable")
            algo=matches[0]
            tuning={}
        else:
            tuning={}
            for cfg in range(len(CONFIGS)):
                fn=lambda cfg=cfg:ext.gemm_out(a,w,c,cfg,1.)
                fn();accuracy(a,w,c,1.)
                g=capture(fn,20)
                tuning[f"cuda_{cfg}"]=statistics.median(elapsed_graph(g,20) for _ in range(3))
            config=min(range(len(CONFIGS)),key=lambda v:tuning[f"cuda_{v}"])
            for idx in range(plan.count()):
                try:
                    fn=lambda idx=idx:plan.run_out(a,w,c,idx,1.)
                    fn();accuracy(a,w,c,1.)
                    g=capture(fn,20)
                    tuning[f"cublas_{idx}"]=statistics.median(elapsed_graph(g,20) for _ in range(3))
                except RuntimeError:
                    continue
            candidates=[idx for idx in range(plan.count()) if f"cublas_{idx}" in tuning]
            if not candidates: raise RuntimeError("no validated cuBLASLt candidate")
            algo=min(candidates,key=lambda v:tuning[f"cublas_{v}"])
        signature=dict(plan.algorithm_info(algo))
        del plan,a,w,c
        seeds=[]
        for seed in args.seeds:
            torch.manual_seed(seed)
            x=torch.randn(m,k,device="cuda");y=torch.randn(n,k,device="cuda")
            a,sa=pack(x);w,sw=pack(y);alpha=sa*sw
            c=torch.empty(m,n,device="cuda",dtype=torch.bfloat16);d=torch.empty_like(c)
            plan=ext.Plan(a,w,d)
            matches=[i for i in range(plan.count()) if dict(plan.algorithm_info(i))==signature]
            if not matches: raise RuntimeError("cuBLASLt algorithm changed across inputs")
            algo=matches[0]
            ext.gemm_out(a,w,c,config,alpha);plan.run_out(a,w,d,algo,alpha)
            checks={"cuda":accuracy(a,w,c,alpha,x,y),"cublas":accuracy(a,w,d,alpha,x,y),
                    "cuda_vs_cublas_relative_l2":relative_l2(c,d)}
            if checks["cuda_vs_cublas_relative_l2"]>4e-3: raise AssertionError("full output mismatch")
            graphs={"cuda":capture(lambda:ext.gemm_out(a,w,c,config,alpha),args.iterations),
                    "cublas":capture(lambda:plan.run_out(a,w,d,algo,alpha),args.iterations)}
            samples={name:[] for name in graphs}
            for _ in range(args.repeats):
                names=list(graphs);rng.shuffle(names)
                for name in names: samples[name].append(elapsed_graph(graphs[name],args.iterations))
            seeds.append({"seed":seed,"samples_us":samples,"us":{name:statistics.median(t) for name,t in samples.items()},"accuracy":checks})
            del graphs,plan,x,y,a,w,c,d
        row={"shape_mnk":[m,n,k],"config":config,"cublas_algorithm":algo,"cublas_algorithm_info":signature,"tuning_us":tuning,"seeds":seeds,
             "median_us":{name:statistics.median(s["us"][name] for s in seeds) for name in ("cuda","cublas")},
             "relative_throughput":statistics.median(s["us"]["cublas"]/s["us"]["cuda"] for s in seeds)}
        report["results"].append(row)
        root.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(f"{m}x{n}x{k} cfg{config} Lt#{algo}: {row['median_us']} efficiency={row['relative_throughput']:.3f}",flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--trace",action="store_true",help="one validated profiled launch; requires frozen tuning report")
    parser.add_argument("--backend",choices=("cuda","cublas"),default="cuda")
    parser.add_argument("--shapes",default="64,4096,4096;256,4096,4096;1024,1024,1024;2048,2048,2048;4096,4096,4096;128,11008,4096;512,4096,11008;8192,4096,4096")
    parser.add_argument("--seeds",type=int,nargs="+",default=[7,17,42])
    parser.add_argument("--seed",type=int,default=7,help="timing order seed")
    parser.add_argument("--iterations",type=int,default=100)
    parser.add_argument("--repeats",type=int,default=7)
    parser.add_argument("--tuning-report",type=Path)
    parser.add_argument("--output",type=Path,default=Path("results/cuda_fp8_gemm_20261116.json"))
    args=parser.parse_args()
    if min(args.iterations,args.repeats)<=0: parser.error("positive iterations and repeats required")
    ext=load_extension()
    if args.trace:
        if not args.tuning_report: parser.error("trace requires --tuning-report")
        shape=tuple(map(int,args.shapes.split(',')))
        m,n,k=shape
        row=next(r for r in json.loads(args.tuning_report.read_text())["results"] if tuple(r["shape_mnk"])==shape)
        torch.manual_seed(42)
        a,sa=pack(torch.randn(m,k,device="cuda"));w,sw=pack(torch.randn(n,k,device="cuda"))
        out=torch.empty(m,n,device="cuda",dtype=torch.bfloat16)
        plan=ext.Plan(a,w,out)
        signature=row["cublas_algorithm_info"]
        idx=next(i for i in range(plan.count()) if dict(plan.algorithm_info(i))==signature)
        fn=(lambda:ext.gemm_out(a,w,out,row["config"],sa*sw)) if args.backend=="cuda" else (lambda:plan.run_out(a,w,out,idx,sa*sw))
        fn();accuracy(a,w,out,sa*sw)
        for _ in range(10): fn()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        with torch.cuda.nvtx.range(f"fp8/{args.backend}/{m}x{n}x{k}"): fn()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        print(f"trace validated: {args.backend} {shape}; extension={ext.__file__}",flush=True)
        return
    checks=correctness_checks(ext)
    print(json.dumps(checks),flush=True)
    if not args.check_only: benchmark(args,ext)


if __name__=="__main__":
    main()
