"""Controlled CUDA GEMM diagnostics; see --help for individual stages."""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ncu_installations() -> dict:
    candidates = set(glob.glob("/opt/nvidia/nsight-compute/*/ncu")
                     + glob.glob("/usr/local/cuda*/nsight-compute*/ncu")
                     + glob.glob("/usr/local/cuda*/bin/ncu")
                     + glob.glob("/mnt/*/NsightCompute/target/linux-desktop-glibc_2_11_3-x64/ncu")
                     + glob.glob("/mnt/c/Program Files/NVIDIA Corporation/Nsight Compute */target/linux-desktop-glibc_2_11_3-x64/ncu"))
    if shutil.which("ncu"):
        candidates.add(shutil.which("ncu"))
    result = {}
    for candidate in sorted(candidates):
        try:
            run = subprocess.run([candidate,"--version"],capture_output=True,text=True)
        except OSError:
            continue
        if run.returncode==0:
            result[candidate] = run.stdout + run.stderr
    return result


def select_ncu(explicit: str | None = None) -> str:
    """Prefer the newest installed Linux CLI; permit an explicit override."""
    override=explicit or os.environ.get("NCU_BIN")
    if override:
        resolved=shutil.which(override)
        if not resolved:
            raise FileNotFoundError(f"NCU_BIN/--ncu is not executable: {override}")
        return resolved
    versions=[]
    for path,description in ncu_installations().items():
        match=re.search(r"Version\s+(\d+)\.(\d+)\.(\d+)",description)
        if match:
            versions.append((tuple(map(int,match.groups())),path))
    if not versions:
        raise FileNotFoundError("No Linux Nsight Compute CLI found; set NCU_BIN")
    return max(versions)[1]


def tool_inventory() -> dict:
    result=ncu_installations()
    for tool in ("nsys", "nvidia-smi", "cuobjdump"):
        result[tool] = shutil.which(tool)
    return result


def suite(output_dir: Path, row_only: bool = False, stage_only: bool = False,
          layout_only: bool = False) -> None:
    """Unprofiled paired timing and controlled changes, separate from tracing."""
    import torch
    from benchmarks.cuda_gemm_bench import CONFIGS, accuracy, capture, elapsed_graph, load_extension

    ext = load_extension()
    output_dir.mkdir(parents=True,exist_ok=True)
    report = {"date_utc":datetime.now(timezone.utc).isoformat(), "gpu":torch.cuda.get_device_name(),
              "torch":torch.__version__,"configs":CONFIGS,"resources":[],"timings":[],
              "method":"30 nodes per graph; 5 paired randomized rounds; 3 seeds; preallocated buffers; warm cache",
              "nvidia_smi":subprocess.check_output(["nvidia-smi","--query-gpu=clocks.sm,temperature.gpu,power.draw","--format=csv,noheader"],text=True).strip()}
    rng=random.Random(20261115)
    filename="layout_ablation.json" if layout_only else "stage_ablation.json" if stage_only else "row_grid.json" if row_only else "ablation.json"
    def save() -> None:
        (output_dir/filename).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    def run(dtype: torch.dtype, m: int, n: int, k: int, variants: list[tuple[str,int,int]], group: str) -> None:
        # Identical seed/shape data are reused across all variants.
        rows=[]
        for seed in (7,17,42):
            torch.manual_seed(seed)
            a=torch.randn(m,k,device="cuda",dtype=dtype)
            b=torch.randn(k,n,device="cuda",dtype=dtype)
            ref=torch.empty(m,n,device="cuda",dtype=dtype)
            ext.cublas_out(a,b,ref)
            graphs={}
            outputs={}
            for name,cfg,pad in variants:
                out=torch.empty_like(ref)
                outputs[name]=out
                if cfg<0:
                    fn=lambda out=out:ext.cublas_out(a,b,out)
                elif cfg>=100:
                    fn=lambda cfg=cfg,pad=pad,out=out:ext.gemm_diagnostic_out(a,b,out,cfg-100,pad)
                else:
                    fn=lambda cfg=cfg,pad=pad,out=out:ext.gemm_out_smem(a,b,out,cfg,pad)
                fn()
                accuracy(a,b,out,ref)
                graphs[name]=capture(fn,30)
                graphs[name].replay()
            torch.cuda.synchronize()
            times={name:[] for name in graphs}
            for _ in range(5):
                order=list(graphs)
                rng.shuffle(order)
                for name in order:
                    times[name].append(elapsed_graph(graphs[name],30))
            medians={name:statistics.median(samples) for name,samples in times.items()}
            rows.append({"seed":seed,"us":medians,"samples_us":times})
            del graphs,outputs,a,b,ref,out,fn
        item={"group":group,"dtype":str(dtype),"shape_mnk":[m,n,k],"variants":variants,"seeds":rows,
              "median_us":{name:statistics.median(row["us"][name] for row in rows) for name,_,_ in variants}}
        report["timings"].append(item)
        save()
        print(f"{group} {dtype} {m}x{n}x{k}: {item['median_us']}",flush=True)
    empty=capture(ext.launch_empty,100)
    report["empty_node_us"]=[elapsed_graph(empty,100) for _ in range(7)]
    del empty
    for dtype in (torch.float16,torch.bfloat16):
        bf=dtype==torch.bfloat16
        for config in (0,5,10,16,17):
            for size in (1024,4096):
                report["resources"].append({"dtype":str(dtype),"config":config,"size":size,"padding":0,
                    "info":dict(ext.kernel_info(bf,config,size,size,0))})
        if layout_only:
            # Bank audit is an address-model prediction, not a hardware counter.
            report["a_ldmatrix_bank_groups"]={
                "original":[((r*32+((r%4)*8))//8)%8 for r in range(8)],
                "alternative":[((r*32+(((r//2)%4)*8))//8)%8 for r in range(8)],
                "interpretation":"16-byte row segments; original repeats 4 groups, alternative covers 8"}
            variants=[("cublas",-1,0),("cfg10",10,0),("cfg10_altA",112,0),
                      ("cfg17",17,0),("cfg17_altA",113,0),
                      ("tile64_s2",100,0),("tile64_s2_altA",106,0),
                      ("tile64_s3",101,0),("tile64_s3_altA",107,0),
                      ("tile64_s6",104,0),("tile64_s6_altA",110,0)]
            for variant in (0,1,4,6,7,10,12,13):
                report["resources"].append({"dtype":str(dtype),"diagnostic_variant":variant,"padding":0,
                    "info":dict(ext.diagnostic_info(bf,variant,1024,1024,0))})
            for size in (1024,4096):
                run(dtype,size,size,size,variants,"layout")
            continue
        if stage_only:
            natural=[("cfg10",10,0),("cublas",-1,0)]
            equalized=[("cublas",-1,0)]
            for variant,stages in enumerate((2,3,4,5,6,8)):
                natural.append((f"stages{stages}",100+variant,0))
                report["resources"].append({"dtype":str(dtype),"diagnostic_variant":variant,"stages":stages,"padding":0,
                    "info":dict(ext.diagnostic_info(bf,variant,1024,1024,0))})
                if stages<=6:
                    pad=49152-(64+64)*32*stages*2
                    equalized.append((f"stages{stages}",100+variant,pad))
                    report["resources"].append({"dtype":str(dtype),"diagnostic_variant":variant,"stages":stages,"padding":pad,
                        "equalized":True,"info":dict(ext.diagnostic_info(bf,variant,1024,1024,pad))})
            for size in (1024,4096):
                run(dtype,size,size,size,natural,"stage_natural")
                run(dtype,size,size,size,equalized,"stage_equalized_48k")
            continue
        if row_only:
            # N is fixed and all tiles are full: no edge-mask waste confound.
            for m in (896,960,1024,1088,1152,1216,1280,1408,1536,2048,4096):
                report["resources"].append({"dtype":str(dtype),"config":10,"shape_mn":[m,1024],"padding":0,
                    "info":dict(ext.kernel_info(bf,10,m,1024,0))})
                run(dtype,m,1024,1024,[("cfg10",10,0),("cfg17",17,0),("cublas",-1,0)],"row_grid")
            continue
        # Same block shape/K tile, changing only warp tile and pipeline stages.
        for size in (1024,4096):
            run(dtype,size,size,size,[("cublas",-1,0),("cfg10",10,0),("cfg17",17,0),
                                     ("stage3",0,0),("stage4",5,0),("warp64_stage3",16,0)],"configuration")
        # Same instruction stream, only unused dynamic shared memory changes.
        for cfg in (10,17):
            variants=[("base",cfg,0)]
            for pad in (8*1024,16*1024,32*1024):
                meta=dict(ext.kernel_info(bf,cfg,1024,1024,pad))
                report["resources"].append({"dtype":str(dtype),"config":cfg,"size":1024,"padding":pad,"info":meta})
                variants.append((f"pad{pad//1024}k",cfg,pad))
            for size in (1024,4096):
                run(dtype,size,size,size,variants,f"residency_cfg{cfg}")
        # Fix the exact kernel and grid while sweeping reduction length.
        for size in (1024,4096):
            for k in (256,512,1024,2048,4096):
                run(dtype,size,size,k,[("cfg10",10,0),("cublas",-1,0)],"k_sweep")
        # Tail/wave control: keep K fixed; output grid crosses resident wave bounds.
        for size in (896,960,1024,1088,1152,1280):
            run(dtype,size,size,1024,[("cfg10",10,0),("cublas",-1,0)],"grid_sweep")
    print(output_dir/filename,flush=True)


def inspect_timeline(output_dir: Path) -> None:
    with sqlite3.connect(output_dir/"timeline.sqlite") as conn:
        names=[row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for name in names:
            if any(key in name for key in ("KERNEL","NVTX","StringIds","RUNTIME")):
                print(name,conn.execute(f'PRAGMA table_info("{name}")').fetchall())
                print(conn.execute(f'SELECT * FROM "{name}" LIMIT 1').fetchall())


def analyze_timeline(output_dir: Path) -> dict:
    with sqlite3.connect(output_dir/"timeline.sqlite") as conn:
        conn.row_factory=sqlite3.Row
        strings={r["id"]:r["value"] for r in conn.execute("SELECT * FROM StringIds")}
        ranges=list(conn.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'gemm/%' AND end IS NOT NULL"))
        groups={}
        detail=[]
        for nv in ranges:
            kernels=list(conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=? ORDER BY start",(nv["start"],nv["end"])))
            if not kernels:
                raise RuntimeError(f"no GPU kernels in NVTX range {nv['text']}")
            expected=100 if nv["text"]=="gemm/empty/graph" else 20
            durations=[(r["end"]-r["start"])/1000 for r in kernels]
            gaps=[(b["start"]-a["end"])/1000 for a,b in zip(kernels,kernels[1:])]
            key=nv["text"].rsplit("/rep",1)[0]
            row={"range":nv["text"],"kernel_count":len(kernels),"calls":expected,
                 "kernel_us_per_call":sum(durations)/expected,
                 "span_us_per_call":(kernels[-1]["end"]-kernels[0]["start"])/1000/expected,
                 "gap_mean_us":statistics.mean(gaps),"gap_median_us":statistics.median(gaps),
                 "launches":[],"range_us_per_call":(nv["end"]-nv["start"])/1000/expected}
            seen=set()
            for kernel in kernels:
                name=strings[kernel["demangledName"]]
                if name in seen: continue
                seen.add(name)
                row["launches"].append({"name":name,"grid":[kernel[f"grid{d}"] for d in "XYZ"],
                    "block":[kernel[f"block{d}"] for d in "XYZ"],"registers_per_thread":kernel["registersPerThread"],
                    "static_shared":kernel["staticSharedMemory"],"dynamic_shared":kernel["dynamicSharedMemory"],
                    "local_bytes_per_thread":kernel["localMemoryPerThread"]})
            detail.append(row)
            groups.setdefault(key,[]).append(row)
        summary=[]
        for key,rows in groups.items():
            row={"range":key,"repeats":len(rows),"kernel_count":rows[0]["kernel_count"],"launches":rows[0]["launches"]}
            for field in ("kernel_us_per_call","span_us_per_call","gap_mean_us","gap_median_us","range_us_per_call"):
                row[field]=statistics.median(r[field] for r in rows)
            summary.append(row)
    result={"source":"Nsight Systems 2026.5.1 CUDA kernel timestamps, NVTX ranges include synchronization",
            "summary":summary,"ranges":detail}
    (output_dir/"timeline_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for row in summary:
        print(f"{row['range']}: kernel={row['kernel_us_per_call']:.3f} us span={row['span_us_per_call']:.3f} us gap={row['gap_mean_us']:.3f} us",flush=True)
    return result


def trace_all() -> None:
    import torch
    from benchmarks.cuda_gemm_bench import capture,load_extension
    ext=load_extension()
    torch.cuda.profiler.start()
    for dtype_name,dtype in (("fp16",torch.float16),("bf16",torch.bfloat16)):
        for size,cfg in ((1024,10),(1024,17),(4096,10),(4096,17)):
            a=torch.randn(size,size,device="cuda",dtype=dtype)
            b=torch.randn_like(a)
            c=torch.empty_like(a)
            for backend in ("cuda","cublas"):
                fn=(lambda:ext.gemm_out(a,b,c,cfg)) if backend=="cuda" else (lambda:ext.cublas_out(a,b,c))
                graph=capture(fn,20)
                graph.replay(); torch.cuda.synchronize()
                for rep in range(3):
                    for method in ("graph","eager"):
                        label=f"gemm/{backend}/{dtype_name}/{size}/cfg{cfg}/{method}/rep{rep}"
                        with torch.cuda.nvtx.range(label):
                            if method=="graph":
                                graph.replay()
                            else:
                                for _ in range(20): fn()
                            torch.cuda.synchronize()
                del graph
            del a,b,c
    graph=capture(ext.launch_empty,100)
    with torch.cuda.nvtx.range("gemm/empty/graph"):
        graph.replay(); torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def nsys(output_dir: Path) -> None:
    output_dir.mkdir(parents=True,exist_ok=True)
    prefix=output_dir/"timeline"
    cmd=[shutil.which("nsys"),"profile","--force-overwrite=true","--trace=cuda,nvtx",
         "--sample=none","--cpuctxsw=none","--cuda-graph-trace=node","-o",str(prefix),
         sys.executable,"-m","benchmarks.cuda_gemm_profile","--mode","trace-all"]
    run=subprocess.run(cmd,capture_output=True,text=True)
    (output_dir/"nsys.log").write_text(run.stdout+run.stderr,encoding="utf-8")
    print(run.stdout+run.stderr,flush=True)
    if run.returncode:
        raise RuntimeError("Nsight Systems capture failed")
    subprocess.run([shutil.which("nsys"),"export","--type=sqlite","--force-overwrite=true",
                    "-o",str(prefix)+".sqlite",str(prefix)+".nsys-rep"],check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["tools", "trace", "trace-all", "ncu", "suite", "rows", "stages", "layout", "nsys", "inspect", "analyze"], default="tools")
    parser.add_argument("--dtype", choices=["fp16","bf16"], default="fp16")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--shape", type=int, nargs=3, metavar=("M","N","K"),
                        help="rectangular trace/counter shape; overrides --size")
    parser.add_argument("--config", type=int, default=10)
    parser.add_argument("--backend", choices=["cuda","cublas"], default="cuda")
    parser.add_argument("--ncu", help="override auto-selected newest Linux ncu (also NCU_BIN)")
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cuda_gemm_profile"))
    args = parser.parse_args()
    if args.size<=0 or args.calls<=0 or args.config<0 or args.config>=21:
        parser.error("positive size/calls and config in [0,21) required")
    if args.shape and min(args.shape)<=0:
        parser.error("shape dimensions must be positive")
    if args.mode == "tools":
        print(json.dumps(tool_inventory(),indent=2),flush=True)
    elif args.mode == "suite":
        suite(args.output_dir)
    elif args.mode == "rows":
        suite(args.output_dir,row_only=True)
    elif args.mode == "stages":
        suite(args.output_dir,stage_only=True)
    elif args.mode == "layout":
        suite(args.output_dir,layout_only=True)
    elif args.mode == "inspect":
        inspect_timeline(args.output_dir)
    elif args.mode == "analyze":
        analyze_timeline(args.output_dir)
    elif args.mode == "trace-all":
        trace_all()
    elif args.mode == "nsys":
        nsys(args.output_dir)
    elif args.mode == "trace":
        import torch
        from benchmarks.cuda_gemm_bench import accuracy, load_extension

        ext = load_extension()
        dtype = torch.float16 if args.dtype=="fp16" else torch.bfloat16
        torch.manual_seed(42)
        m,n,k=args.shape or (args.size,args.size,args.size)
        a = torch.randn(m,k,device="cuda",dtype=dtype)
        b = torch.randn(k,n,device="cuda",dtype=dtype)
        c = torch.empty(m,n,device="cuda",dtype=dtype)
        reference=torch.empty_like(c)
        ext.cublas_out(a,b,reference)
        fn = (lambda:ext.gemm_out(a,b,c,args.config)) if args.backend=="cuda" else (lambda:ext.cublas_out(a,b,c))
        fn()
        accuracy(a,b,c,reference)
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        trace_shape="x".join(map(str,args.shape)) if args.shape else str(args.size)
        with torch.cuda.nvtx.range(f"gemm/{args.backend}/{args.dtype}/{trace_shape}/cfg{args.config}"):
            for _ in range(args.calls):
                fn()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        print("trace completed",flush=True)
    elif args.mode == "ncu":
        tool = select_ncu(args.ncu)
        version=subprocess.check_output([tool,"--version"],text=True).strip()
        match=re.search(r"Version\s+(\d+)\.(\d+)\.(\d+)",version)
        tag="_".join(match.groups()) if match else "unknown"
        directory=args.output_dir/f"ncu_{tag}"
        directory.mkdir(parents=True,exist_ok=True)
        shape_tag="x".join(map(str,args.shape)) if args.shape else str(args.size)
        prefix = directory / f"ncu_{args.backend}_{args.dtype}_{shape_tag}_cfg{args.config}"
        command = [tool,"--target-processes","all","--profile-from-start","off",
                   "--launch-count","1","--clock-control","none","--cache-control","none",
                   "--section","LaunchStats","--section","Occupancy",
                   "--section","SpeedOfLight","--section","SchedulerStats",
                   "--section","WarpStateStats","--section","MemoryWorkloadAnalysis",
                   "--section","MemoryWorkloadAnalysis_Tables",
                   "--section","ComputeWorkloadAnalysis","--force-overwrite",
                   "-o",str(prefix),sys.executable,"-m","benchmarks.cuda_gemm_profile",
                   "--mode","trace","--backend",args.backend,"--dtype",args.dtype,
                   "--size",str(args.size),"--config",str(args.config),"--calls","1"]
        if args.shape:
            command += ["--shape",*map(str,args.shape)]
        run = subprocess.run(command,capture_output=True,text=True)
        Path(str(prefix)+".log").write_text(run.stdout+run.stderr,encoding="utf-8")
        status={"tool":tool,"version":version,"command":command,"returncode":run.returncode}
        if not run.returncode:
            reports=list(directory.glob(prefix.name+".ncu-rep*"))
            if not reports:
                raise RuntimeError("ncu returned success without producing a report")
            report=max(reports,key=lambda p:p.stat().st_mtime_ns)
            exported=subprocess.run([tool,"--import",str(report),"--page","raw","--csv"],capture_output=True,text=True)
            status.update(report=str(report),csv=str(prefix)+".csv",export_returncode=exported.returncode)
            Path(str(prefix)+".csv").write_text(exported.stdout,encoding="utf-8")
            if exported.stderr:
                Path(str(prefix)+".export.log").write_text(exported.stderr,encoding="utf-8")
        Path(str(prefix)+".status.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
        print(f"Selected ncu: {tool}",flush=True)
        print(run.stdout+run.stderr,flush=True)
        print(f"NCU exit code: {run.returncode}",flush=True)
        if run.returncode:
            raise SystemExit(run.returncode)
        if status.get("export_returncode"):
            raise SystemExit(status["export_returncode"])


if __name__ == "__main__":
    main()
