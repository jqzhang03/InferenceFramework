"""Profile cfg22 at Qwen QKV/gate-up shapes; preserve raw reports and units."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import torch
from torch.nn import functional as F
from nanovllm.layers.cuda_linear import load_extension


def buffers(m: int,n: int,k: int,seed: int):
    torch.manual_seed(seed)
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*0.02
    return x,w,w.T.contiguous(),torch.empty(m,n,device='cuda',dtype=x.dtype)


def operation(ext,backend: str,x,w,b,out):
    if backend=='cfg22':return lambda:ext.gemm_out(x,b,out,22)
    if backend=='alt_a':return lambda:ext.gemm_diagnostic_out(x,b,out,14,0)
    return lambda:F.linear(x,w)


def trace(args: argparse.Namespace) -> None:
    ext=load_extension();m,n,k=args.shape
    x,w,b,out=buffers(m,n,k,42);fn=operation(ext,args.backend,x,w,b,out)
    result=fn();actual=out if result is None else result
    expected=x.double()@w.double().T
    error=((actual.double()-expected).norm()/expected.norm()).item()
    if not torch.isfinite(actual).all() or error>0.006:raise AssertionError(error)
    for _ in range(10):fn()
    torch.cuda.synchronize()
    metadata=dict(backend=args.backend,shape=args.shape,relative_l2=error,device=torch.cuda.get_device_name(),
                  torch_version=torch.__version__,cuda=torch.version.cuda,
                  bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    if args.backend!='torch':
        metadata['resources']=dict(ext.kernel_info(True,22,m,n,0) if args.backend=='cfg22' else ext.diagnostic_info(True,14,m,n,0))
    if args.meta:args.meta.write_text(json.dumps(metadata,indent=2))
    torch.cuda.profiler.start()
    with torch.cuda.nvtx.range(f'cfg22/{args.backend}/{m}x{n}x{k}'):
        fn()
    torch.cuda.synchronize();torch.cuda.profiler.stop()


def graph(fn):
    fn();fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    return g


def timing(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True,exist_ok=True)
    timing_path=args.output/args.timing_name
    if timing_path.exists():raise ValueError('use fresh timing output')
    ext=load_extension();rows=[];rng=random.Random(31)
    for m,n,k in ((8,4096,1024),(8,6144,1024),(16,4096,1024),(8,8192,1024)):
        for seed in (17,42,99):
            x,w,b,out=buffers(m,n,k,seed)
            expected=x.double()@w.double().T
            errors={}
            for backend in ('cfg22','alt_a','torch'):
                value=operation(ext,backend,x,w,b,out)()
                actual=out if value is None else value
                error=((actual.double()-expected).norm()/expected.norm()).item()
                if not torch.isfinite(actual).all() or error>0.006:raise AssertionError((backend,m,n,k,seed,error))
                errors[backend]=error
            # One resident weight and a rotating 28-layer working set.
            for regime,count in (('resident',1),('rotating28',28)):
                weights=[w]+[torch.randn_like(w)*0.02 for _ in range(count-1)]
                packed=[v.T.contiguous() for v in weights]
                outputs=[torch.empty_like(out) for _ in weights]
                repeats=32 if count==1 else 1
                def run(backend):
                    for _ in range(repeats):
                        for ww,bb,oo in zip(weights,packed,outputs):operation(ext,backend,x,ww,bb,oo)()
                graphs={key:graph(lambda key=key:run(key)) for key in ('cfg22','alt_a','torch')}
                samples={key:[] for key in graphs}
                for _ in range(9):
                    order=list(graphs);rng.shuffle(order)
                    for key in order:
                        a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record();graphs[key].replay();z.record();z.synchronize()
                        samples[key].append(a.elapsed_time(z)*1000/(count*repeats))
                row=dict(shape=[m,n,k],seed=seed,regime=regime,samples_us=samples,
                         median_us={key:statistics.median(v) for key,v in samples.items()},accuracy_relative_l2=errors)
                rows.append(row);print(m,n,regime,seed,row['median_us'],flush=True)
                timing_path.write_text(json.dumps(dict(method='9 randomized paired graph rounds; 3 seeds; resident 32-node graph or rotating28 one-pass graph',rows=rows),indent=2))
                del graphs,weights,packed,outputs


def capture(args: argparse.Namespace) -> None:
    from benchmarks.cuda_gemm_profile import select_ncu
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.glob('*.ncu-rep*')) and not args.resume:raise ValueError('preserve captures; choose fresh output or --resume')
    tool=select_ncu();version=subprocess.check_output([tool,'--version'],text=True)
    jobs=[]
    shapes=[(8,4096,1024),(8,6144,1024)]
    for cache in ('none','all'):
        for shape in shapes:
            for rep in range(args.repeats):
                for backend in (('cfg22','alt_a','torch') if rep%2==0 else ('torch','alt_a','cfg22')):
                    jobs.append((shape,backend,cache,rep,'primary'))
    # Same cfg22: valid-row filling and grid-size controls, not assumed bottlenecks.
    for shape in ((16,4096,1024),(8,8192,1024)):
        jobs.append((shape,'cfg22','none',0,'control'))
    manifest=[]
    for shape,backend,cache,rep,group in jobs:
        prefix=args.output/f'{group}_{backend}_{"x".join(map(str,shape))}_{cache}_r{rep}'
        command=[tool,'--target-processes','all','--profile-from-start','off','--launch-count','1',
                 '--clock-control','none','--cache-control',cache]
        for section in ('LaunchStats','Occupancy','SpeedOfLight','SchedulerStats','WarpStateStats',
                        'MemoryWorkloadAnalysis','MemoryWorkloadAnalysis_Tables','ComputeWorkloadAnalysis'):
            command+=['--section',section]
        command+=['--metrics','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
                  '-o',str(prefix),sys.executable,'-m','benchmarks.cfg22_ncu','--mode','trace',
                  '--shape',*map(str,shape),'--backend',backend,'--meta',str(prefix)+'.meta.json']
        native=list(args.output.glob(prefix.name+'.ncu-rep*'))
        status_path=Path(str(prefix)+'.status.json')
        if args.resume and native and status_path.exists() and json.loads(status_path.read_text())['returncode']==0:
            print('REUSE',prefix.name,flush=True);status=json.loads(status_path.read_text())
        else:
            print('NCU',prefix.name,flush=True)
            run=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            Path(str(prefix)+'.log').write_text(run.stdout,encoding='utf-8')
            status=dict(shape=shape,backend=backend,cache_control=cache,repeat=rep,group=group,
                        command=command,returncode=run.returncode,tool_version=version)
            status_path.write_text(json.dumps(status,indent=2))
            if run.returncode:raise RuntimeError(run.stdout[-2500:])
            native=list(args.output.glob(prefix.name+'.ncu-rep*'))
        if len(native)!=1:raise RuntimeError(f'ambiguous native report: {native}')
        export=subprocess.run([tool,'--import',str(native[0]),'--page','raw','--csv'],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        status.update(native_report=str(native[0]),export_returncode=export.returncode)
        status_path.write_text(json.dumps(status,indent=2))
        Path(str(prefix)+'.csv').write_text(export.stdout,encoding='utf-8')
        if export.returncode:raise RuntimeError(export.stdout[-2000:]+export.stderr)
        manifest.append(status);(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))


def model_trace(args: argparse.Namespace) -> None:
    from nanovllm import LLM,SamplingParams
    engine=LLM('/home/zjq/huggingface/Qwen3-0.6B',linear_backend='cuda_decode',enforce_eager=True,
               max_num_seqs=8,max_num_batched_tokens=2048,max_model_len=512,gpu_memory_utilization=0.55,
               multi_step_decode=False,mixed_cudagraph=False,kv_swap=False,
               latency_aware_scheduling=False,slo_aware_scheduling=False,tpot_aware_scheduling=False)
    sp=SamplingParams(temperature=0.00001,max_tokens=3,ignore_eos=True)
    rng=random.Random(42);prompts=[[rng.randrange(100,10000) for _ in range(128)] for _ in range(8)]
    engine.generate([[t+1 for t in p] for p in prompts],sp,use_tqdm=False)
    for prompt in prompts:engine.add_request(prompt,sp)
    _,kind,_,_=engine.step()
    if kind!='prefill':raise AssertionError(kind)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    engine.step()
    torch.cuda.synchronize();torch.cuda.profiler.stop()
    engine.exit()


def model_capture(args: argparse.Namespace) -> None:
    from benchmarks.cuda_gemm_profile import select_ncu
    tool=select_ncu();folder=args.output/'model_context';folder.mkdir(exist_ok=True)
    if any(folder.glob('*.ncu-rep*')) and not args.resume:raise ValueError('preserve model context reports; use --resume')
    metrics=['gpu__time_duration.sum','dram__bytes_op_read.sum','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
             'lts__t_sector_hit_rate.pct','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
             'sm__warps_active.avg.pct_of_peak_sustained_active','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum']
    for offset,name in ((28,'layer14_qkv'),(29,'layer14_gate_up')):
        prefix=folder/name
        status_path=Path(str(prefix)+'.status.json')
        if args.resume and status_path.exists() and json.loads(status_path.read_text())['returncode']==0 and Path(str(prefix)+'.csv').exists():
            print('REUSE',name,flush=True)
            continue
        command=[tool,'--target-processes','all','--profile-from-start','off','--kernel-name','regex:gemm_kernel',
                 '--launch-skip',str(offset),'--launch-count','1','--replay-mode','application',
                 '--clock-control','none','--cache-control','none','--metrics',','.join(metrics),'-o',str(prefix),
                 sys.executable,'-m','benchmarks.cfg22_ncu','--mode','model-trace']
        print('MODEL NCU',name,flush=True)
        run=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        Path(str(prefix)+'.log').write_text(run.stdout)
        Path(str(prefix)+'.status.json').write_text(json.dumps(dict(command=command,returncode=run.returncode),indent=2))
        if run.returncode:raise RuntimeError(run.stdout[-3000:])
        native=list(folder.glob(name+'.ncu-rep*'))
        export=subprocess.run([tool,'--import',str(native[0]),'--page','raw','--csv'],capture_output=True,text=True)
        Path(str(prefix)+'.csv').write_text(export.stdout)
        if export.returncode:raise RuntimeError(export.stdout+export.stderr)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['trace','timing','capture','model-trace','model-capture'],default='capture')
    p.add_argument('--shape',nargs=3,type=int,default=[8,4096,1024])
    p.add_argument('--backend',choices=['cfg22','alt_a','torch'],default='cfg22')
    p.add_argument('--output',type=Path,default=Path('results/cfg22_ncu_20261116'))
    p.add_argument('--meta',type=Path);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--timing-name',default='timing_amortized.json')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();{'trace':trace,'timing':timing,'capture':capture,'model-trace':model_trace,'model-capture':model_capture}[args.mode](args)
