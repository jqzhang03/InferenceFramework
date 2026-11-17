"""Collect full-process Nsight traces; analyze only the marked inference window."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time


@contextmanager
def nvtx(label: str):
    import torch
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def trace(args: argparse.Namespace) -> None:
    import torch
    from nanovllm import LLM, SamplingParams
    from benchmarks.nsys_profile import _wrap_method

    options={} if args.framework_defaults else dict(
        multi_step_decode=False,mixed_cudagraph=False,kv_swap=False,
        latency_aware_scheduling=False,slo_aware_scheduling=False,
        tpot_aware_scheduling=False,max_prefill_chunk_tokens=2048)
    init_start=time.perf_counter()
    with nvtx('startup/model_initialization'):
        engine=LLM(args.model,linear_backend=args.backend,cuda_linear_tuning=args.tuning,
               enforce_eager=args.eager,max_num_seqs=16,max_num_batched_tokens=2048,
               max_model_len=2048,gpu_memory_utilization=0.55,
               **options)
    initialization_s=time.perf_counter()-init_start
    params=SamplingParams(temperature=0.00001,max_tokens=32,ignore_eos=True)

    def prompts(seed: int) -> list[list[int]]:
        rng=random.Random(seed)
        return [[rng.randrange(100,10000) for _ in range(args.length)] for _ in range(args.batch)]

    warm_start=time.perf_counter()
    with nvtx('startup/inference_warmup'):
        for seed in (40,41):
            engine.generate(prompts(seed),params,use_tqdm=False)
        torch.cuda.synchronize()
    warmup_s=time.perf_counter()-warm_start
    runner=engine.model_runner
    for obj,method,label,kind in (
        (engine,'step','LLMEngine.step',None),
        (engine,'add_request','LLMEngine.add_request',None),
        (engine.scheduler,'schedule','Scheduler.schedule',None),
        (engine.scheduler,'postprocess','Scheduler.postprocess',None),
        (runner,'run','ModelRunner.run',1),
        (runner,'run_model','ModelRunner.run_model',2),
        (runner,'prepare_prefill','ModelRunner.prepare_prefill',None),
        (runner,'prepare_decode','ModelRunner.prepare_decode',None),
        (runner,'prepare_sample','ModelRunner.prepare_sample',None),
        (runner.sampler,'forward','Sampler.forward',None),
        (runner.model,'compute_logits','Qwen3.compute_logits',None),
    ):
        _wrap_method(obj,method,label,kind)
    hooks=[]
    if args.eager:
        for name,module in runner.model.named_modules():
            if type(module).__name__ not in ('Qwen3DecoderLayer','Qwen3Attention','Qwen3MLP',
                                            'QKVParallelLinear','RowParallelLinear','MergedColumnParallelLinear',
                                            'RMSNorm','RotaryEmbedding','SiluAndMul','Attention','VocabParallelEmbedding') and name!='lm_head':
                continue
            range_label='shared_rotary_emb' if type(module).__name__=='RotaryEmbedding' else name
            def enter(mod,inputs,label=range_label):
                detail=''
                if inputs and hasattr(inputs[0],'shape') and hasattr(mod,'input_size'):
                    detail=f' input={tuple(inputs[0].shape)}'
                torch.cuda.nvtx.range_push('Qwen3.'+label+detail)
            def leave(mod,inputs,out):
                torch.cuda.nvtx.range_pop()
            hooks.extend((module.register_forward_pre_hook(enter),module.register_forward_hook(leave)))
    telemetry=[]
    torch.cuda.synchronize()
    with nvtx('nsys_capture'):
        for rep in range(3):
            torch.manual_seed(123)
            start=time.perf_counter()
            with nvtx(f'request_batch/rep{rep}/B{args.batch}/L{args.length}/{args.backend}'):
                outputs=engine.generate(prompts(1000+rep),params,use_tqdm=False)
                torch.cuda.synchronize()
            telemetry.append(dict(rep=rep,seconds=time.perf_counter()-start,
                                  tokens=[o['token_ids'] for o in outputs],step_stats=dict(engine._step_stats),
                                  decode_graph_replays=runner.decode_cudagraph_replay_count,
                                  mixed_graph_replays=runner.mixed_cudagraph_replay_count))
    metadata=dict(backend=args.backend,batch=args.batch,input_length=args.length,output_length=32,
                  eager=args.eager,installed_linears=runner.cuda_linear_count,
                  decode_graph_replays=runner.decode_cudagraph_replay_count,
                  device=torch.cuda.get_device_name(),torch_version=torch.__version__,
                  cuda_version=torch.version.cuda,tuning=args.tuning,telemetry=telemetry,
                  framework_defaults=args.framework_defaults,initialization_s=initialization_s,warmup_s=warmup_s,
                  policy={key:getattr(engine.config,key) for key in (
                      'multi_step_decode','max_decode_steps','mixed_cudagraph','latency_aware_scheduling',
                      'slo_aware_scheduling','tpot_aware_scheduling','max_prefill_chunk_tokens','kv_swap',
                      'max_num_seqs','max_num_batched_tokens','max_model_len','gpu_memory_utilization')},
                  annotation='CPU NVTX ranges; no per-module CUDA Events; eager adds module hooks')
    args.output.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in metadata.items() if k!='telemetry'}),flush=True)
    for hook in hooks:hook.remove()
    engine.exit()


def merged_ns(intervals: list[tuple[int,int]]) -> int:
    end=-1;total=0
    for start,stop in sorted(intervals):
        total+=max(0,stop-max(start,end));end=max(end,stop)
    return total


def category(name: str) -> str:
    for pattern,label in (
        ('decode_mma_kernel','CUDA decode MMA'),('decode_simt_kernel','CUDA batched SIMT'),
        ('finish_kernel<','CUDA split-K reduction'),
        ('linear_gemv_kernel','CUDA BF16 GEMV'),('fp8_kernel<','CUDA native FP8 GEMM'),
        ('pack_rows_kernel','FP8 activation quantization'),('gemm_kernel<','CUDA BF16 GEMM'),
        ('store_kvcache','KV cache store'),('flash','FlashAttention'),('fmha','FlashAttention'),
        ('gemm','Library GEMM / LM head'),('gemv','Library GEMV / LM head'),
        ('rms','RMSNorm'),('rsqrt','RMSNorm'),('rotary','RoPE'),('silu','SiLU'),('embedding','Embedding'),
        ('softmax','Sampling softmax / argmax'),
        ('reduce','Reduction / sampling'),('elementwise','Elementwise'),
    ):
        if pattern.lower() in name.lower():return label
    return 'Other kernels'


def analyze(path: Path, kernel_labels: dict[str,str] | None = None) -> dict:
    db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    strings={r['id']:r['value'] for r in db.execute('SELECT * FROM StringIds')}
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    nv=[]
    for row in db.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        r=dict(row);r['label']=r.get('text') or strings.get(r.get('textId'),'')
        nv.append(r)
    window=[r for r in nv if r['label']=='nsys_capture']
    if len(window)!=1:raise RuntimeError(f'expected one nsys_capture, got {len(window)}')
    start,end=window[0]['start'],window[0]['end']
    kernels=[]
    for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=? ORDER BY start',(start,end)):
        r=dict(row);r['name']=strings.get(r.get('demangledName'),strings.get(r.get('shortName'),'unknown'))
        r['category']=category(r['name'])
        if r['category']=='Other kernels' and kernel_labels and r['name'] in kernel_labels:
            r['category']=kernel_labels[r['name']]
        kernels.append(r)
    if not kernels:raise RuntimeError('capture has no CUDA kernels')
    apis=[]
    for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=?',(start,end)):
        r=dict(row);r['name']=strings.get(r['nameId'],'unknown');apis.append(r)
    by_correlation={r['correlationId']:r for r in apis if r.get('correlationId')}
    attribution=[]
    for label,group in (('Qwen3.compute_logits','Library LM head'),('Sampler.forward','Sampling softmax / argmax')):
        ranges=sorted((r for r in nv if r['label']==label),key=lambda r:r['start'])
        attribution.append((group,ranges,[r['start'] for r in ranges]))
    leaf_ranges=defaultdict(list)
    for r in nv:
        name=r['label'].split(' input=')[0]
        if not name.startswith('Qwen3.'):continue
        group=None
        for suffix,label in (('.qkv_proj','QKV projection'),('.o_proj','O projection'),
                             ('.gate_up_proj','Gate/up projection'),('.down_proj','Down projection'),
                             ('.embed_tokens','Embedding'),('.act_fn','SiLU'),
                             ('.q_norm','RMSNorm'),('.k_norm','RMSNorm'),('.norm','RMSNorm'),
                             ('layernorm','RMSNorm'),('rotary_emb','RoPE'),('.attn','Attention + KV store')):
            if name.endswith(suffix):group=label;break
        if group:leaf_ranges[group].append(r)
    for group,ranges in leaf_ranges.items():
        ranges.sort(key=lambda r:r['start'])
        attribution.append((group,ranges,[r['start'] for r in ranges]))
    linked=Counter()
    for k in kernels:
        api=by_correlation.get(k.get('correlationId'))
        if api is None:continue
        for group,ranges,starts in attribution:
            idx=bisect_right(starts,api['start'])-1
            if idx>=0 and ranges[idx]['end']>=api['end']:
                k['module_group']=group;linked[group]+=1
                if group in ('Library LM head','Sampling softmax / argmax'):k['category']=group
                break
    copies=[]
    for table in ('CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        if table in tables:
            copies.extend(dict(r) for r in db.execute(f'SELECT * FROM {table} WHERE start>=? AND end<=?',(start,end)))

    def summarize(a: int,b: int) -> dict:
        ks=[k for k in kernels if k['start']>=a and k['end']<=b]
        cs=[k for k in copies if k['start']>=a and k['end']<=b]
        aa=[k for k in apis if k['start']>=a and k['end']<=b]
        counts=Counter();times=Counter();modules=Counter();names=defaultdict(lambda:dict(calls=0,ns=0))
        api_names=defaultdict(lambda:dict(calls=0,ns=0))
        for k in ks:
            d=k['end']-k['start'];counts[k['category']]+=1;times[k['category']]+=d
            if k.get('module_group'):modules[k['module_group']]+=d
            names[k['name']]['calls']+=1;names[k['name']]['ns']+=d
        for k in aa:
            api_names[k['name']]['calls']+=1;api_names[k['name']]['ns']+=k['end']-k['start']
        busy=merged_ns([(k['start'],k['end']) for k in ks+cs])
        result=dict(wall_ms=(b-a)/1e6,kernel_calls=len(ks),kernel_sum_ms=sum(times.values())/1e6,
                    gpu_busy_union_ms=busy/1e6,gpu_idle_ms=(b-a-busy)/1e6,
                    busy_pct=busy/(b-a)*100,copies=len(cs),copy_ms=sum(k['end']-k['start'] for k in cs)/1e6,
                    copy_bytes=sum(k.get('bytes',0) for k in cs),
                    categories=[dict(name=n,calls=counts[n],ms=d/1e6) for n,d in times.most_common()],
                    module_gpu_groups=[dict(name=n,ms=d/1e6) for n,d in modules.most_common()],
                    top_kernels=[dict(name=n,calls=r['calls'],ms=r['ns']/1e6) for n,r in sorted(names.items(),key=lambda kv:-kv[1]['ns'])[:15]],
                    top_apis=[dict(name=n,calls=r['calls'],ms=r['ns']/1e6) for n,r in sorted(api_names.items(),key=lambda kv:-kv[1]['ns'])[:12]])
        if ks:result.update(first_kernel=ks[0]['start'],last_kernel=max(k['end'] for k in ks))
        return result
    phases={}
    phase_rows=[]
    for kind in ('prefill','decode','mixed'):
        ranges=[r for r in nv if r['label']==f'ModelRunner.run [{kind}]' and r['start']>=start and r['end']<=end]
        if not ranges:continue
        rows=[summarize(r['start'],r['end']) for r in ranges]
        phase_rows.extend(dict(phase=kind,start=r['start'],end=r['end'],**s) for r,s in zip(ranges,rows))
        cat=Counter();modules=Counter()
        for s in rows:
            for item in s['categories']:cat[item['name']]+=item['ms']
            for item in s['module_gpu_groups']:modules[item['name']]+=item['ms']
        phases[kind]=dict(calls=len(rows),wall_ms=sum(r['wall_ms'] for r in rows),
                          kernel_sum_ms=sum(r['kernel_sum_ms'] for r in rows),
                          busy_ms=sum(r['gpu_busy_union_ms'] for r in rows),
                          categories=[dict(name=n,ms=d) for n,d in cat.most_common()],
                          module_gpu_groups=[dict(name=n,ms=d) for n,d in modules.most_common()])
    gaps=[]
    for rep in [r for r in nv if r['label'].startswith('request_batch/') and r['start']>=start and r['end']<=end]:
        dec=sorted((r for r in phase_rows if r['phase']=='decode' and r['start']>=rep['start'] and r['end']<=rep['end']),key=lambda r:r['start'])
        gaps.extend(max(0,b['first_kernel']-a['last_kernel'])/1e6 for a,b in zip(dec,dec[1:]) if 'first_kernel' in b and 'last_kernel' in a)
    annotations=defaultdict(lambda:dict(calls=0,ns=0))
    for r in nv:
        if r['start']>=start and r['end']<=end and r['label'].startswith(('LLMEngine.','Scheduler.','Sampler.','ModelRunner.prepare','Qwen3.compute_logits')):
            annotations[r['label']]['calls']+=1;annotations[r['label']]['ns']+=r['end']-r['start']
    module_map=defaultdict(set)
    for k in kernels:
        if k.get('module_group'):module_map[k['name']].add(k['module_group'])
    result=dict(sqlite=str(path),capture=summarize(start,end),phases=phases,phase_rows=phase_rows,
                kernel_module_map={name:sorted(groups) for name,groups in module_map.items()},
                supplementary_kernel_labels=kernel_labels or {},
                startup_nvtx=[dict(label=r['label'],wall_ms=(r['end']-r['start'])/1e6) for r in nv if r['label'].startswith('startup/')],
                direct_kernel_nvtx_links=dict(linked),
                between_decode_gpu_gaps_ms=gaps,
                cpu_nvtx=[dict(name=n,calls=r['calls'],ms=r['ns']/1e6) for n,r in sorted(annotations.items(),key=lambda x:-x[1]['ns'])])
    db.close()
    return result


def collect(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.glob('*.nsys-rep')):
        raise ValueError('output already contains traces; choose a new directory to preserve history')
    cases=[('torch_graph_b8','torch',8,128,False,''),
           ('cuda_graph_b1','cuda',1,128,False,'results/qwen_cuda_bf16_tune.json'),
           ('cuda_graph_b8','cuda',8,128,False,'results/qwen_cuda_bf16_tune.json'),
           ('cuda_graph_long','cuda',1,1024,False,'results/qwen_cuda_bf16_tune.json'),
           ('fp8_graph_b8','cuda_fp8_auto',8,128,False,'results/qwen_cuda_tune.json'),
           ('cuda_eager_b8','cuda',8,128,True,'results/qwen_cuda_bf16_tune.json')]
    if args.suite=='decode-current':
        cases=[('decode_default_b1','cuda_decode',1,128,False,''),
               ('decode_default_b8','cuda_decode',8,128,False,''),
               ('decode_default_long','cuda_decode',1,1024,False,''),
               ('decode_controlled_b8','cuda_decode',8,128,False,''),
               ('decode_eager_b8','cuda_decode',8,128,True,'')]
    manifest=[]
    for name,backend,batch,length,eager,tuning in cases:
        prefix=args.output/name
        cmd=[args.nsys,'profile','--force-overwrite=true','--trace=cuda,nvtx,osrt,cublas',
             '--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','-o',str(prefix),
             sys.executable,'-m','benchmarks.qwen_cuda_nsys','--mode','trace','--model',args.model,
             '--backend',backend,'--batch',str(batch),'--length',str(length),'--output',str(prefix)+'.json']
        if tuning:cmd+=['--tuning',tuning]
        if eager:cmd+=['--eager']
        if args.suite=='decode-current' and name!='decode_controlled_b8':cmd+=['--framework-defaults']
        print('CAPTURE',name,flush=True)
        run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        Path(str(prefix)+'.log').write_text(run.stdout,encoding='utf-8')
        manifest.append(dict(case=name,command=cmd,status=run.returncode))
        (args.output/'commands.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        if run.returncode:raise RuntimeError(f'{name} failed: {run.stdout[-2500:]}')
        run=subprocess.run([args.nsys,'export','--type=sqlite','--force-overwrite=true','-o',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        Path(str(prefix)+'.export.log').write_text(run.stdout,encoding='utf-8')
        if run.returncode:raise RuntimeError(run.stdout)
        result=analyze(Path(str(prefix)+'.sqlite'))
        Path(str(prefix)+'.analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(name,json.dumps({k:v for k,v in result['capture'].items() if k in ('wall_ms','kernel_sum_ms','busy_pct','kernel_calls')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['trace','collect','analyze'],default='collect')
    p.add_argument('--output',type=Path,default=Path('profiles/qwen_cuda_nsys_20261116'))
    p.add_argument('--model',default='/home/zjq/huggingface/Qwen3-0.6B')
    p.add_argument('--nsys',default='/usr/local/bin/nsys')
    p.add_argument('--backend',default='cuda');p.add_argument('--tuning',default='')
    p.add_argument('--suite',choices=['legacy','decode-current'],default='legacy')
    p.add_argument('--framework-defaults',action='store_true')
    p.add_argument('--batch',type=int,default=8);p.add_argument('--length',type=int,default=128)
    p.add_argument('--eager',action='store_true');args=p.parse_args()
    if args.mode=='trace':trace(args)
    elif args.mode=='collect':collect(args)
    else:print(json.dumps(analyze(args.output),indent=2))
