"""Paired unprofiled timings and repeated ncu comparisons; keep datasets separate."""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def read_ncu_csv(path: Path) -> dict:
    """Retain raw metric names/units; do not treat stall ratios as percentages."""
    with path.open(encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=2 or "Kernel Name" not in rows[1]:
        raise ValueError(f"Expected units plus one kernel in {path}")
    return {"kernel":rows[1]["Kernel Name"],"units":rows[0],"metrics":rows[1]}


def timings(root: Path) -> None:
    import torch
    from benchmarks.cuda_gemm_bench import accuracy, capture, elapsed_graph, load_extension

    ext=load_extension()
    rng=random.Random(20261116)
    report={"date_utc":datetime.now(timezone.utc).isoformat(),"gpu":torch.cuda.get_device_name(),
            "torch":torch.__version__,"method":"100 graph nodes, 7 randomized paired rounds, seeds 7/17/42; fixed configs, warm cache", "rows":[]}
    shapes=[(mn,mn,k) for mn in (1024,4096) for k in (1024,4096)]
    shapes += [(1024,1024,k) for k in (256,512,2048)]
    shapes += [(m,1024,1024) for m in (896,1152,1216,1536)]
    for dtype in (torch.float16,torch.bfloat16):
        for m,n,k in shapes:
            seeds=[]
            for seed in (7,17,42):
                torch.manual_seed(seed)
                a=torch.randn(m,k,device="cuda",dtype=dtype)
                b=torch.randn(k,n,device="cuda",dtype=dtype)
                ref=torch.empty(m,n,device="cuda",dtype=dtype)
                ext.cublas_out(a,b,ref)
                outputs={name:torch.empty_like(ref) for name in ("cfg10","cfg17","cublas")}
                fns={"cfg10":lambda:ext.gemm_out(a,b,outputs["cfg10"],10),
                     "cfg17":lambda:ext.gemm_out(a,b,outputs["cfg17"],17),
                     "cublas":lambda:ext.cublas_out(a,b,outputs["cublas"])}
                checks={}
                graphs={}
                for name,fn in fns.items():
                    fn()
                    checks[name]=accuracy(a,b,outputs[name],ref)
                    graphs[name]=capture(fn,100)
                    graphs[name].replay()
                torch.cuda.synchronize()
                samples={name:[] for name in graphs}
                for _ in range(7):
                    order=list(graphs)
                    rng.shuffle(order)
                    for name in order:
                        samples[name].append(elapsed_graph(graphs[name],100))
                seeds.append({"seed":seed,"samples_us":samples,"accuracy":checks,
                              "us":{name:statistics.median(v) for name,v in samples.items()}})
                del graphs,fns,outputs,a,b,ref,fn
            row={"dtype":str(dtype),"shape_mnk":[m,n,k],"seeds":seeds,
                 "median_us":{name:statistics.median(s["us"][name] for s in seeds) for name in samples}}
            report["rows"].append(row)
            (root/"timings.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
            print(f"timing {dtype} {m}x{n}x{k}: {row['median_us']}",flush=True)


def counters(root: Path, repeats: int) -> None:
    """Profile serially, alternating backend order between independent captures."""
    cases=[]
    for dtype in ("fp16","bf16"):
        for size,cfg in ((1024,10),(4096,17)):
            for repeat in range(repeats):
                for backend in (("cuda","cublas") if repeat%2==0 else ("cublas","cuda")):
                    cases.append((dtype,(size,size,size),cfg,backend,repeat,"primary"))
        # Same cfg10 separates reduction length from output-grid capacity.
        for shape in ((1024,1024,4096),(4096,4096,1024),(4096,4096,4096),
                      (1152,1024,1024),(1216,1024,1024)):
            cases.append((dtype,shape,10,"cuda",0,"control"))
    result={"date_utc":datetime.now(timezone.utc).isoformat(),"repeats":repeats,"rows":[]}
    for dtype,shape,cfg,backend,repeat,group in cases:
        folder=root/"captures"/f"{group}_rep{repeat}"
        command=[sys.executable,"-m","benchmarks.cuda_gemm_profile","--mode","ncu",
                 "--backend",backend,"--dtype",dtype,"--config",str(cfg),
                 "--shape",*map(str,shape),"--output-dir",str(folder)]
        print(f"counter {group} {dtype} {shape} {backend} cfg{cfg} repeat{repeat}",flush=True)
        run=subprocess.run(command,capture_output=True,text=True)
        folder.mkdir(parents=True,exist_ok=True)
        log=folder/f"{backend}_{dtype}_{'x'.join(map(str,shape))}_cfg{cfg}.driver.log"
        log.write_text(run.stdout+run.stderr,encoding="utf-8")
        if run.returncode:
            raise RuntimeError(f"counter capture failed: {log}")
        statuses=list(folder.glob(f"ncu_*/ncu_{backend}_{dtype}_{'x'.join(map(str,shape))}_cfg{cfg}.status.json"))
        if len(statuses)!=1:
            raise RuntimeError(f"ambiguous status files: {statuses}")
        status=json.loads(statuses[0].read_text())
        row={"group":group,"dtype":dtype,"shape_mnk":shape,"config":cfg,"backend":backend,
             "repeat":repeat,"status":status,"status_path":str(statuses[0])}
        row.update(read_ncu_csv(Path(status["csv"])))
        result["rows"].append(row)
        (root/"counters.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(f"counter study: {len(result['rows'])} successful captures",flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("all","timings","counters"),default="all")
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--output-dir",type=Path,default=Path("results/cuda_gemm_counter_20261116"))
    args=parser.parse_args()
    if args.repeats<1:
        parser.error("positive repeats required")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.mode in ("all","timings"):
        timings(args.output_dir)
    if args.mode in ("all","counters"):
        counters(args.output_dir,args.repeats)


if __name__=="__main__":
    main()
