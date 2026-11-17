"""Qwen decoder projection tuning and end-to-end replay."""
from __future__ import annotations
import argparse
import json
import random
import statistics
import time
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
import torch
from torch.nn import functional as F
from nanovllm.layers.cuda_linear import CudaLinear, load_extension


def latency(fn: Callable, iterations: int = 20) -> float:
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iterations): fn()
    times=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();g.replay();b.record();b.synchronize()
        times.append(a.elapsed_time(b)*1000/iterations)
    return statistics.median(times)


def tune(args: argparse.Namespace) -> None:
    ext=load_extension(); fp8=load_extension(True) if 'fp8' in args.backend else None
    rows=[]; selectors={}
    for m in (1,4,8,16,32,128,512,1024):
        for n,k in ((4096,1024),(1024,2048),(6144,1024),(1024,3072)):
            torch.manual_seed(17)
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*0.02
            adapter=CudaLinear(w,args.backend,ext,fp8)
            ref=F.linear(x,w)
            base=latency(lambda:F.linear(x,w))
            key=f'{m},{n},{k}'
            candidates=[]
            for mode,configs in [('gemv',[0]),('gemm',range(24))]+([('fp8',range(28))] if fp8 else []):
                if mode=='gemv' and m>16: continue
                for cfg in configs:
                    adapter.selectors={key:dict(mode=mode,config=cfg)}
                    out=adapter(x)
                    error=((out.float()-ref.float()).norm()/ref.float().norm()).item()
                    if not torch.isfinite(out).all() or error>(0.05 if mode=='fp8' else 0.008):
                        raise AssertionError((key,mode,cfg,error))
                    us=latency(lambda:adapter(x),10)
                    candidates.append(dict(mode=mode,config=cfg,us=us,error=error))
            best=min(candidates,key=lambda a:a['us'])
            adapter.selectors={key:best}
            torch.manual_seed(99)
            validate_x=torch.randn_like(x)
            expected=F.linear(validate_x,w)
            actual=adapter(validate_x)
            validation_error=((actual.float()-expected.float()).norm()/expected.float().norm()).item()
            if validation_error>(0.05 if best['mode']=='fp8' else 0.008):
                raise AssertionError((key,'heldout',validation_error))
            selectors[key]=dict(mode='torch') if args.backend.endswith('auto') and best['us']>base*0.98 else best
            rows.append(dict(shape=[m,n,k],torch_us=base,best=best,candidates=candidates,validation_error=validation_error))
            print(key,round(base,2),best,flush=True)
    result=dict(device=torch.cuda.get_device_name(),backend=args.backend,selectors=selectors,rows=rows)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2))


def infer(args: argparse.Namespace) -> None:
    from nanovllm import LLM, SamplingParams
    engine=LLM(args.model,linear_backend=args.backend,cuda_linear_tuning=args.tuning,
               max_num_seqs=16,max_num_batched_tokens=2048,max_model_len=2048,
               gpu_memory_utilization=0.55,enforce_eager=args.eager,
               multi_step_decode=False,mixed_cudagraph=False,
               latency_aware_scheduling=False,slo_aware_scheduling=False,
               tpot_aware_scheduling=False,max_prefill_chunk_tokens=2048)
    cases=[]
    for batch,length in ((1,128),(8,128),(1,1024)):
        gen=random.Random(42)
        prompts=[[gen.randrange(100,10000) for _ in range(length)] for _ in range(batch)]
        params=SamplingParams(temperature=0.00001,max_tokens=32,ignore_eos=True)
        engine.generate(prompts,params,use_tqdm=False)
        samples=[];tokens=[]
        for repeat in range(5):
            gen=random.Random(1000+repeat)
            prompts=[[gen.randrange(100,10000) for _ in range(length)] for _ in range(batch)]
            torch.manual_seed(123)
            torch.cuda.synchronize();start=time.perf_counter()
            outputs=engine.generate(prompts,params,use_tqdm=False)
            torch.cuda.synchronize();samples.append(time.perf_counter()-start)
            tokens.append([o['token_ids'] for o in outputs])
        cases.append(dict(batch=batch,input_length=length,output_length=32,seconds=samples,
                          median_s=statistics.median(samples),tokens=tokens))
        print(args.backend,batch,length,statistics.median(samples),flush=True)
    logits=[]
    for length in (128,1024):
        gen=random.Random(555)
        prompt=[gen.randrange(100,10000) for _ in range(length)]
        engine.generate([prompt],SamplingParams(temperature=0.00001,max_tokens=1,ignore_eos=True),
                        use_tqdm=False,collect_logits=True)
        logits.append(engine.collected_logits[-1][1].float().cpu())
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    torch.save(logits,Path(args.output).with_suffix('.pt'))
    Path(args.output).write_text(json.dumps(dict(backend=args.backend,eager=args.eager,cases=cases,device=torch.cuda.get_device_name(),
                                              installed_linears=engine.model_runner.cuda_linear_count),indent=2))
    engine.exit()


def suite(args: argparse.Namespace) -> None:
    """Serial GPU measurements, reverse backend order in the second round."""
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    bf16=json.loads(Path('results/qwen_cuda_bf16_tune.json').read_text())
    # The same-precision auto variant is independently visible from FP8.
    hybrid=dict(bf16,backend='cuda_auto',selectors={
        key: (dict(mode='torch') if row['best']['us']>row['torch_us']*0.98 else row['best'])
        for key,row in zip(bf16['selectors'],bf16['rows'])})
    (root/'cuda_auto_tuning.json').write_text(json.dumps(hybrid,indent=2))
    variants=[('torch',''),('cuda',''),('cuda','results/qwen_cuda_bf16_tune.json'),
              ('cuda_auto',str(root/'cuda_auto_tuning.json')),('cuda_fp8_auto','results/qwen_cuda_tune.json')]
    for rep in range(2):
        for idx in (range(len(variants)) if rep==0 else reversed(range(len(variants)))):
            backend,tuning=variants[idx]
            cmd=[sys.executable,'-m','benchmarks.qwen_cuda_linear','--mode','infer','--backend',backend,
                 '--model',args.model,'--output',str(root/f'v{idx}_rep{rep}.json')]
            if tuning: cmd+=['--tuning',tuning]
            if args.eager: cmd+=['--eager']
            subprocess.run(cmd,check=True)


def prefill_suite(args: argparse.Namespace) -> None:
    root=Path(args.output)
    fp8=json.loads(Path('results/qwen_cuda_tune.json').read_text())
    bf16=json.loads((root/'cuda_auto_tuning.json').read_text())
    for key in fp8['selectors']:
        if int(key.split(',')[0])<32:
            fp8['selectors'][key]=bf16['selectors'].get(key,dict(mode='torch'))
    tuning=root/'fp8_prefill_tuning.json'
    tuning.write_text(json.dumps(fp8,indent=2))
    for rep in range(2):
        subprocess.run([sys.executable,'-m','benchmarks.qwen_cuda_linear','--mode','infer',
                        '--backend','cuda_fp8_auto','--tuning',str(tuning),'--model',args.model,
                        '--output',str(root/f'v5_rep{rep}.json')],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['tune','infer','suite','prefill-suite'],default='tune')
    p.add_argument('--backend',default='cuda');p.add_argument('--tuning',default='')
    p.add_argument('--model',default='/home/zjq/huggingface/Qwen3-0.6B')
    p.add_argument('--eager',action='store_true');p.add_argument('--output',required=True)
    args=p.parse_args();{'tune':tune,'infer':infer,'suite':suite,'prefill-suite':prefill_suite}[args.mode](args)
