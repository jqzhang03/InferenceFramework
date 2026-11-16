"""Isolate grouped CTA scheduling from tile changes on matched FP8 inputs."""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import torch


def main() -> None:
    from benchmarks.cuda_fp8_gemm_bench import accuracy,load_extension,pack
    from benchmarks.cuda_gemm_bench import capture,elapsed_graph

    ext=load_extension();rng=random.Random(20261116)
    report={"method":"same-input graph timing; three seeds, 7 randomized paired rounds, 100 nodes; group-only pairs 0/18,3/19,11/22", "rows":[]}
    output=Path("results/cuda_fp8_group_ablation_20261116.json")
    for m,n,k in ((128,11008,4096),(512,4096,11008),(4096,4096,4096)):
        seeds=[]
        for seed in (7,17,42):
            torch.manual_seed(seed)
            a,sa=pack(torch.randn(m,k,device="cuda"));w,sw=pack(torch.randn(n,k,device="cuda"))
            configs=(0,18,3,19,11,22,25)
            outputs={cfg:torch.empty(m,n,device="cuda",dtype=torch.bfloat16) for cfg in configs}
            graphs={}
            for cfg in configs:
                fn=lambda cfg=cfg:ext.gemm_out(a,w,outputs[cfg],cfg,sa*sw)
                fn();accuracy(a,w,outputs[cfg],sa*sw)
                graphs[cfg]=capture(fn,100)
            for base,grouped in ((0,18),(3,19),(11,22)):
                if not torch.equal(outputs[base],outputs[grouped]): raise AssertionError("grouping changed output")
            samples={cfg:[] for cfg in configs}
            for _ in range(7):
                order=list(configs);rng.shuffle(order)
                for cfg in order: samples[cfg].append(elapsed_graph(graphs[cfg],100))
            seeds.append({"seed":seed,"samples_us":samples,"us":{cfg:statistics.median(t) for cfg,t in samples.items()},"group_pairs_bitwise_equal":True})
            del graphs,outputs,a,w,fn
        row={"shape_mnk":[m,n,k],"seeds":seeds,
             "median_us":{cfg:statistics.median(s["us"][cfg] for s in seeds) for cfg in configs},
             "paired_speedup":{f"{base}_to_{group}":statistics.median(s["us"][base]/s["us"][group] for s in seeds) for base,group in ((0,18),(3,19),(11,22))}}
        report["rows"].append(row)
        output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(f"{m}x{n}x{k}: {row['median_us']}; grouped speedup {row['paired_speedup']}",flush=True)


if __name__=="__main__":
    main()
