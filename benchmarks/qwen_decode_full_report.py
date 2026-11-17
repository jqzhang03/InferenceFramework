"""Full inference breakdown for the current cuda_decode backend."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from benchmarks.qwen_cuda_nsys import analyze


def main(root: Path) -> None:
    cases=['decode_default_b1','decode_default_b8','decode_default_long','decode_controlled_b8','decode_eager_b8']
    labels=dict(decode_default_b1='默认调度 B=1/L=128',decode_default_b8='默认调度 B=8/L=128',
                decode_default_long='默认调度 B=1/L=1024',decode_controlled_b8='固定预算对照 B=8/L=128',
                decode_eager_b8='eager 逐层诊断 B=8/L=128')
    eager=analyze(root/'decode_eager_b8.sqlite')
    # Only transfer unique non-linear kernel identities verified by eager launches.
    kernel_labels={name:groups[0] for name,groups in eager['kernel_module_map'].items()
                   if len(groups)==1 and groups[0] in ('RoPE','Embedding','RMSNorm','SiLU')}
    data={};metadata={};summary={};csv_rows=[]
    for name in cases:
        d=analyze(root/f'{name}.sqlite',kernel_labels)
        (root/f'{name}.analysis.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
        data[name]=d;metadata[name]=json.loads((root/f'{name}.json').read_text())
        summary[name]={k:d[k] for k in ('capture','phases','cpu_nvtx','startup_nvtx','direct_kernel_nvtx_links')}
        for phase,p in [('all',d['capture'])]+list(d['phases'].items()):
            for r in p['categories']:
                csv_rows.append(dict(case=name,phase=phase,component=r['name'],gpu_ms=r['ms'],
                                     gpu_ms_per_batch=r['ms']/3,kernel_share_pct=r['ms']/p['kernel_sum_ms']*100))
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (root/'components.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    lines=['# Qwen3-0.6B + cuda_decode：完整推理 Nsight Systems 分解（2026-11-16）','',
           '## 本次配置与范围','',
           '当前工作树、RTX 5060 Ti / SM120、BF16、TP=1、linear_backend=cuda_decode。'
           '自动加载打包的冻结选择器，112 个 decoder 投影 adapter；M=4/8/16 的 O/down 使用新 MMA，'
           '其他形状保留旧 CUDA 路径。LM head 调库，Attention 使用原 FlashAttention。'
           '没有额外 FP8 权重量化。每批固定 32 输出 token，合成 token 输入，三批不同输入；两批同形状预热。','',
           '主结果恢复当前默认调度开关：multi_step_decode=True（最多4步）、mixed_cudagraph=True、'
           'latency/SLO/TPOT-aware scheduling=True、KV swap=True。为控制采集规模，资源上限是 '
           'max_num_seqs=16、max_num_batched_tokens=2048、max_model_len=2048、GPU memory utilization=0.55。'
           '固定预算对照关闭上述动态配额/multi-step/mixed Graph；eager 轨迹用于模块归因，额外带 CPU NVTX hooks。','',
           'Nsight Systems 2026.5.1 全进程采集 CUDA/NVTX/OSRT/cuBLAS，Graph node tracing 开启。'
           '启动和预热另有 NVTX 标签，正式推理只统计 nsys_capture。'
           '以下 wall time 含插桩开销，不替代无 profiler 的性能结果；CPU 调用时间与 GPU 活动可能重叠。','',
           '## 启动、预热和正式推理','',
           '| 场景 | 初始化 ms | 两批预热 ms | 三批正式推理 ms | 每批平均 ms | Graph replay（decode/mixed） |',
           '|---|---:|---:|---:|---:|---|']
    for name in cases:
        m=metadata[name];c=data[name]['capture']
        replays=(sum(r['decode_graph_replays'] for r in m['telemetry']),sum(r['mixed_graph_replays'] for r in m['telemetry']))
        lines.append(f"| {labels[name]} | {m['initialization_s']*1000:.2f} | {m['warmup_s']*1000:.2f} | {c['wall_ms']:.2f} | {c['wall_ms']/3:.2f} | {replays[0]}/{replays[1]} |")
    lines+=['','初始化包含加载、CUDA/JIT 初始化和初始图捕获；预热还可能捕获 mixed 图。'
            'JIT/磁盘缓存状态会随采集顺序变化，不用初始化列比较不同 batch 的效率。'
            '这些成本已单列，正式推理没有重复提示带来的前缀命中。输入直接提供 token IDs，不包含输入文本分词；输出整理包含在正式 wall time 中。','',
            '## 整体 GPU 活动与间隙','',
            '| 场景 | GPU kernel 总 ms/批 | GPU 活动并集 ms/批 | 无 GPU 活动 ms/批 | GPU 活动比例 | kernel 数 |',
            '|---|---:|---:|---:|---:|---:|']
    for name in cases:
        c=data[name]['capture']
        lines.append(f"| {labels[name]} | {c['kernel_sum_ms']/3:.3f} | {c['gpu_busy_union_ms']/3:.3f} | {c['gpu_idle_ms']/3:.3f} | {c['busy_pct']:.1f}% | {c['kernel_calls']} |")
    lines+=['','GPU 活动是 kernel/memcpy/memset 时间区间的并集；空白区间包括主机提交、等待、调度等，'
            '不能全部叫作 CPU 计算。活动比例不是 achieved occupancy。kernel 时间求和不与 CPU NVTX 时间叠加。','',
            '## Prefill / Mixed / Decode 的 wall time 分区','',
            '| 场景 | Prefill 次数、ms/批 | Mixed 次数、ms/批 | Decode forward 次数、ms/批 | Runner 外 ms/批 |',
            '|---|---|---|---|---:|']
    for name in cases:
        d=data[name];cells=[]
        for phase in ('prefill','mixed','decode'):
            p=d['phases'].get(phase,{})
            cells.append(f"{p.get('calls',0)} / {p.get('wall_ms',0)/3:.3f}")
        residual=d['capture']['wall_ms']-sum(p['wall_ms'] for p in d['phases'].values())
        lines.append('| '+labels[name]+' | '+' | '.join(cells)+f' | {residual/3:.3f} |')
    lines+=['','次数统计整份轨迹（三批），时间为每批平均。阶段 wall time 来自互不嵌套的 ModelRunner.run NVTX；'
            'Runner 外包括请求提交、调度/postprocess、生成结果整理等。Decode forward 次数不同于 engine.step 次数，'
            'multi-step 会在一个 step 内执行多次 forward。Mixed 同时处理两类 token，不用人为比例把它拆成纯 Prefill/Decode。','',
            '## 全过程各 GPU 部分耗时（ms/批）','',
            '| 场景 | 线性投影 | Split-K归约 | Attention | LM head | RMSNorm | RoPE | SiLU | KV写入 | 采样 | 其他 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    projection=('CUDA BF16 GEMV','CUDA BF16 GEMM','CUDA decode MMA','CUDA batched SIMT','Library GEMM / LM head','Library GEMV / LM head')
    components=[projection,('CUDA split-K reduction',),('FlashAttention',),('Library LM head',),('RMSNorm',),
                ('RoPE',),('SiLU',),('KV cache store',),('Sampling softmax / argmax',)]
    for name in cases:
        mapping={r['name']:r['ms'] for r in data[name]['capture']['categories']}
        numbers=[sum(mapping.get(k,0) for k in keys)/3 for keys in components]
        other=data[name]['capture']['kernel_sum_ms']/3-sum(numbers)
        lines.append('| '+labels[name]+' | '+' | '.join(f'{x:.3f}' for x in numbers+[other])+' |')
    lines+=['','线性投影不含 LM head 和 Split-K 归约，避免重复计数。LM head/采样按 launch correlation ID 关联。'
            '其他类别按 kernel 名称；RoPE/Embedding 等仅在 eager 的 launch 归属唯一时，将核名映射用于 Graph 分类。'
            '其他包括未归类索引、embedding、逐元素等；0 表示该分类未找到匹配 kernel，不意味着某逻辑完全没有成本。','',
            '## 各阶段 GPU kernel 组成','',
            '| 场景/阶段 | forward 数 | kernel 总 ms/forward | 最耗时分类（占该阶段 kernel 时间） |',
            '|---|---:|---:|---|']
    for name in cases:
        for phase,p in data[name]['phases'].items():
            hot='；'.join(f"{r['name']}: {r['ms']/p['kernel_sum_ms']*100:.1f}%" for r in p['categories'][:4])
            lines.append(f"| {labels[name]} / {phase} | {p['calls']} | {p['kernel_sum_ms']/p['calls']:.4f} | {hot} |")
    lines+=['','## 默认 B=8 的主要热点','']
    d=data['decode_default_b8'];c=d['capture'];total=c['kernel_sum_ms']
    mapping={r['name']:r['ms'] for r in c['categories']}
    for label,keys in [('Decoder 线性投影',projection+('CUDA split-K reduction',)),('LM head',('Library LM head',)),
                       ('Attention',('FlashAttention',)),('RMSNorm',('RMSNorm',)),('采样',('Sampling softmax / argmax',))]:
        value=sum(mapping.get(k,0) for k in keys)
        lines.append(f'- {label}：{value/3:.3f} ms/批，GPU kernel 时间占比 {value/total*100:.1f}%。')
    lines+=['',f"GPU 活动比例 {c['busy_pct']:.1f}%，每批还有 {c['gpu_idle_ms']/3:.3f} ms 未记录到 GPU 活动。"
            '这些空白需对照主机轨道和 CUDA API 判断，不等于 scheduler 本身耗时。'
            '优化后仍应结合投影、词表层以及提交间隙分析；本次未采集硬件 stall/带宽计数器，不作 compute/memory-bound 定论。','',
            '## CPU 范围与真实拷贝','',
            '| 默认 B=8 CPU 范围/API | 次数 | 累计 ms |','|---|---:|---:|']
    for item in d['cpu_nvtx']+c['top_apis'][:8]:
        lines.append(f"| `{item['name']}` | {item['calls']} | {item['ms']:.3f} |")
    lines+=['',f"默认 B=8 实际 GPU copy/memset 总时长 {c['copy_ms']:.3f} ms、{c['copy_bytes']} bytes。"
            'cudaMemcpyAsync 的 CPU 长调用可能包含采样后 .tolist() 等待先前 GPU 计算；不能按 API 总时长认定 PCIe 瓶颈。'
            'LLMEngine.step 是包含模型运行和等待的外层范围，不是纯 CPU 时间。以上 CPU 范围有嵌套，不能相加。','',
            '## eager 逐模块 GPU 归因（仅诊断）','',
            '| 模块 | 三批 GPU kernel ms | ms/批 |','|---|---:|---:|']
    for item in data['decode_eager_b8']['capture']['module_gpu_groups']:
        lines.append(f"| {item['name']} | {item['ms']:.3f} | {item['ms']/3:.3f} |")
    lines+=['','eager 的模块 GPU 时间由 CPU NVTX 范围中的 launch correlation 归因，不是 NVTX 范围的 wall time。'
            '采用叶子模块归属，避免把 Attention 父范围与 QKV/O 再重复相加。RoPE 是共享模块，轨道名为 shared_rotary_emb。'
            '逐层 hooks 和 eager 提交会改变执行节奏，其绝对时间不能直接代替 Graph 版本。','',
            '## Windows 时间线与数据','']
    for name in cases:lines.append(f'- [{name}.nsys-rep](../{root.as_posix()}/{name}.nsys-rep)：{labels[name]}。')
    lines+=['',f'- [分项 CSV](../{root.as_posix()}/components.csv)；[汇总 JSON](../{root.as_posix()}/summary.json)。','',
            '使用 Nsight Systems 2026.5.1 或更新版本。先定位 startup/model_initialization、startup/inference_warmup，'
            '再放大 nsys_capture → request_batch/rep1。查看 LLMEngine.step 和 ModelRunner.run [prefill/mixed/decode]，'
            '展开 CUDA Graph 节点，对照 decode_mma_kernel、finish_kernel、gemm_kernel、LM head 和 FlashAttention。'
            'eager 版本可以展开 Qwen3.model.layers.* 各叶子模块。','',
            '原始 SQLite、命令/日志、配置/输出 JSON 和逐阶段 analysis.json 同目录保留。'
            '本次为合成 token 的离线分析，无线上到达/SLO 结论；未修改引擎或算子实现。','',
            '## 复现','',
            '```bash','bash benchmarks/run_qwen_cuda.sh --nsys --suite decode-current --output profiles/qwen_decode_full_NEW',
            'bash benchmarks/run_qwen_cuda.sh --full-nsys-report --root profiles/qwen_decode_full_NEW','```','',
            '请使用新目录保留历史。旧 CUDA/FP8 和旧优化测量仍保留各自报告，不与本次数据合并。']
    output='\n'.join(lines)+'\n'
    Path('benchmarks/_qwen_decode_full_nsys_20261116.md').write_text(output,encoding='utf-8')
    (root/'REPORT.md').write_text(output.replace(f'../{root.as_posix()}/',''),encoding='utf-8')
    print(json.dumps(dict(default_b8_kernel_ms_per_batch=c['kernel_sum_ms']/3,default_b8_busy_pct=c['busy_pct'],
                          phases={key:p['calls'] for key,p in d['phases'].items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('profiles/qwen_decode_full_nsys_20261116'))
    main(p.parse_args().root)
