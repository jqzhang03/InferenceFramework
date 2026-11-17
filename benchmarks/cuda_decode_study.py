"""Decode kernel tuning, rotating-weight validation, model replay and counters."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import torch
from torch.nn import functional as F
from nanovllm.layers.cuda_linear import CudaLinear,load_extension,load_decode_extension
from benchmarks.qwen_cuda_linear import latency


def stream_times(x: torch.Tensor,weights: list[torch.Tensor],ext,new,old_choice: dict,configs: list[int]) -> dict:
    outputs=[torch.empty(x.size(0),w.size(0),device=x.device,dtype=x.dtype) for w in weights]
    key=f'{x.size(0)},{weights[0].size(0)},{x.size(1)}'
    adapters=[CudaLinear(w,'cuda',ext,selectors={key:old_choice}) for w in weights]
    def run(name):
        for i,w in enumerate(weights):
            if name=='torch':F.linear(x,w)
            elif name=='old':adapters[i](x)
            else:new.decode_out(x,w,outputs[i],int(name),)
    graphs={}
    for name in ['torch','old']+[str(c) for c in configs]:
        run(name);run(name);torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):run(name)
        graphs[name]=g
    samples={key:[] for key in graphs};rng=random.Random(33)
    for _ in range(9):
        order=list(graphs);rng.shuffle(order)
        for name in order:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();graphs[name].replay();end.record();end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000/len(weights))
    return {key:dict(us=statistics.median(values),samples_us=values) for key,values in samples.items()}


def tune(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'tuning.json').exists():
        raise ValueError('preserve previous tuning; choose a fresh --output directory')
    ext=load_extension();new=load_decode_extension()
    previous=json.loads(Path('results/qwen_cuda_bf16_tune.json').read_text())
    selectors=dict(previous['selectors']);rows=[]
    for m in (1,4,8,16):
        for n,k in ((4096,1024),(1024,2048),(6144,1024),(1024,3072)):
            key=f'{m},{n},{k}';torch.manual_seed(17)
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*0.02
            expected=x.double()@w.double().T;out=torch.empty(m,n,device='cuda',dtype=x.dtype)
            candidates=[]
            for cfg in range(23):
                new.decode_out(x,w,out,cfg)
                error=((out.double()-expected).norm()/expected.norm()).item()
                if not torch.isfinite(out).all() or error>0.006:raise AssertionError((key,cfg,error))
                candidates.append(dict(config=cfg,us=latency(lambda:new.decode_out(x,w,out,cfg),10),error=error))
            # Keep the best MMA, split-K and SIMT families for streaming selection.
            shortlist=set(min([c for c in candidates if a<=c['config']<b],key=lambda c:c['us'])['config']
                          for a,b in ((0,6),(6,14),(14,23)))
            weights=[torch.randn_like(w)*0.02 for _ in range(28)]
            stream=stream_times(x,weights,ext,new,previous['selectors'][key],sorted(shortlist))
            cfg=min(shortlist,key=lambda c:stream[str(c)]['us'])
            selectors[key]=dict(mode='decode',config=cfg)
            del weights
            torch.manual_seed(99)
            weights=[torch.randn_like(w)*0.02 for _ in range(28)]
            heldout=stream_times(x,weights,ext,new,previous['selectors'][key],[cfg])
            row=dict(shape=[m,n,k],config=cfg,candidates=candidates,stream=stream,heldout=heldout)
            rows.append(row)
            print(key,'cfg',cfg,'heldout', {a:round(b['us'],2) for a,b in heldout.items()},flush=True)
            result=dict(device=torch.cuda.get_device_name(),backend='cuda_decode',selectors=selectors,rows=rows)
            (args.output/'tuning.json').write_text(json.dumps(result,indent=2))
            del weights


def freeze(args: argparse.Namespace) -> None:
    report=json.loads((args.output/'tuning.json').read_text())
    old=json.loads(Path('results/qwen_cuda_bf16_tune.json').read_text())
    for row in report['rows']:
        key=','.join(map(str,row['shape']));cfg=str(row['config'])
        # M=1 is already bandwidth limited; use only training measurements to gate.
        if row['shape'][0]==1 or row['stream'][cfg]['us']>row['stream']['old']['us']*0.98:
            report['selectors'][key]=old['selectors'][key]
    report['selection_policy']='M=1 retains old GEMV; M>1 enables new path only if training rotating-weight time improves >=2%'
    (args.output/'deployment.json').write_text(json.dumps(report,indent=2))


def replay(args: argparse.Namespace) -> None:
    variants=[('torch',''),('cuda','results/qwen_cuda_bf16_tune.json'),('cuda_decode',str(args.output/'deployment.json'))]
    for rep in range(2):
        for idx in (range(3) if rep==0 else reversed(range(3))):
            backend,tuning=variants[idx]
            command=[sys.executable,'-m','benchmarks.qwen_cuda_linear','--mode','infer','--backend',backend,
                     '--output',str(args.output/f'model_v{idx}_rep{rep}.json')]
            if tuning:command+=['--tuning',tuning]
            subprocess.run(command,check=True)


def probe(args: argparse.Namespace) -> None:
    new=load_decode_extension();torch.manual_seed(99)
    n,k=(1024,2048) if args.config==5 else (1024,3072) if args.config in (8,23) else (4096,1024)
    x=torch.randn(8,k,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(n,k,device='cuda',dtype=x.dtype)
    out=torch.empty(8,n,device='cuda',dtype=x.dtype)
    new.decode_out(x,w,out,args.config);torch.cuda.synchronize()
    reference=x.double()@w.double().T
    if ((out.double()-reference).norm()/reference.norm()).item()>0.006:
        raise AssertionError('counter target failed correctness')
    new.decode_out(x,w,out,args.config);torch.cuda.synchronize()


def counters(args: argparse.Namespace) -> None:
    from benchmarks.cuda_gemm_profile import select_ncu
    ncu=select_ncu();folder=args.output/'counters_swizzle';folder.mkdir(exist_ok=True)
    if any(folder.glob('*.csv')):raise ValueError('preserve previous counters; use a fresh output directory')
    metrics=['gpu__time_duration.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
             'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
             'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum']
    for cfg in (5,8,21,23):
        command=[ncu,'--target-processes','all','--kernel-name','regex:decode_(mma|simt)_kernel',
                 '--launch-skip','1','--launch-count','1','--clock-control','none',
                 '--metrics',','.join(metrics),'--csv','--log-file',str(folder/f'cfg{cfg}.csv'),
                 sys.executable,'-m','benchmarks.cuda_decode_study','--mode','probe','--config',str(cfg)]
        run=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (folder/f'cfg{cfg}.log').write_text(run.stdout)
        (folder/f'cfg{cfg}.command.json').write_text(json.dumps(dict(command=command,status=run.returncode),indent=2))
        if run.returncode:raise RuntimeError(run.stdout[-2000:])
        print('ncu cfg',cfg,'completed',flush=True)


def nsys(args: argparse.Namespace) -> None:
    from benchmarks.qwen_cuda_nsys import analyze
    folder=args.profile_output
    folder.mkdir(parents=True,exist_ok=True)
    if any(folder.glob('*.nsys-rep')):raise ValueError('preserve existing traces; choose another --profile-output')
    for name,backend,tuning in [('old_b8','cuda','results/qwen_cuda_bf16_tune.json'),
                               ('reuse_b8','cuda_decode',str(args.output/'deployment.json'))]:
        prefix=folder/name
        command=['/usr/local/bin/nsys','profile','--force-overwrite=true','--trace=cuda,nvtx,osrt,cublas',
                 '--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','-o',str(prefix),sys.executable,
                 '-m','benchmarks.qwen_cuda_nsys','--mode','trace','--backend',backend,'--batch','8',
                 '--length','128','--tuning',tuning,'--output',str(prefix)+'.json']
        run=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        Path(str(prefix)+'.log').write_text(run.stdout)
        Path(str(prefix)+'.command.json').write_text(json.dumps(dict(command=command,status=run.returncode),indent=2))
        if run.returncode:raise RuntimeError(run.stdout[-2000:])
        subprocess.run(['/usr/local/bin/nsys','export','--type=sqlite','--force-overwrite=true','-o',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],check=True,stdout=subprocess.DEVNULL)
        result=analyze(Path(str(prefix)+'.sqlite'))
        Path(str(prefix)+'.analysis.json').write_text(json.dumps(result,indent=2))
        print('nsys',name,'kernel ms',result['capture']['kernel_sum_ms'],flush=True)


def bank_timing(args: argparse.Namespace) -> None:
    ext=load_extension();new=load_decode_extension();torch.manual_seed(101)
    x=torch.randn(8,3072,device='cuda',dtype=torch.bfloat16)
    weights=[torch.randn(1024,3072,device='cuda',dtype=x.dtype)*0.02 for _ in range(28)]
    report=stream_times(x,weights,ext,new,dict(mode='gemm',config=23),[8,23])
    (args.output/'swizzle_timing.json').write_text(json.dumps(report,indent=2))
    print('swizzle timing', {k:round(v['us'],3) for k,v in report.items()},flush=True)


def sanitize(args: argparse.Namespace) -> None:
    command=['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','1',
             sys.executable,'-m','pytest','tests/test_cuda_decode.py','-q']
    run=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (args.output/'memcheck.log').write_text(run.stdout)
    if run.returncode:raise RuntimeError(run.stdout[-3000:])
    print(run.stdout[-1200:],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['tune','freeze','replay','probe','counters','nsys','sanitize','bank-timing'],default='tune')
    p.add_argument('--output',type=Path,default=Path('results/cuda_decode_reuse_20261116'))
    p.add_argument('--profile-output',type=Path,default=Path('profiles/cuda_decode_reuse_20261116'))
    p.add_argument('--config',type=int,default=1);args=p.parse_args()
    {'tune':tune,'freeze':freeze,'replay':replay,'probe':probe,'counters':counters,'nsys':nsys,'sanitize':sanitize,'bank-timing':bank_timing}[args.mode](args)
