"""Nsight Systems kernel/CPU analysis for the marked Qwen inference window."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from benchmarks.qwen_cuda_nsys import analyze


def main(root: Path) -> None:
    cases=['torch_graph_b8','cuda_graph_b1','cuda_graph_b8','cuda_graph_long','fp8_graph_b8','cuda_eager_b8']
    data={}
    for name in cases:
        result=analyze(root/(name+'.sqlite'))
        (root/(name+'.analysis.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        data[name]=result
    def ms(case: str,phase: str,category: str) -> float:
        return next((r['ms'] for r in data[case]['phases'][phase]['categories'] if r['name']==category),0)
    cuda_decode=data['cuda_graph_b8']['phases']['decode']
    cuda_proj=ms('cuda_graph_b8','decode','CUDA BF16 GEMV')+ms('cuda_graph_b8','decode','CUDA BF16 GEMM')
    lib_decode=data['torch_graph_b8']['phases']['decode']
    lib_proj=ms('torch_graph_b8','decode','Library GEMM / LM head')+ms('torch_graph_b8','decode','Library GEMV / LM head')
    quant=next(x['ms'] for x in data['fp8_graph_b8']['capture']['categories'] if x['name']=='FP8 activation quantization')
    labels=dict(torch_graph_b8='调库，B=8 / L=128，Graph',cuda_graph_b1='CUDA BF16，B=1 / L=128，Graph',
                cuda_graph_b8='CUDA BF16，B=8 / L=128，Graph',cuda_graph_long='CUDA BF16，B=1 / L=1024，Graph',
                fp8_graph_b8='FP8 混合，B=8 / L=128，Graph',cuda_eager_b8='CUDA BF16，B=8 / L=128，eager')
    lines=['# Qwen3-0.6B 新 CUDA 算子：Nsight Systems 分析（2026-11-16）','',
           '## 测量范围','',
           'RTX 5060 Ti / SM120，Nsight Systems 2026.5.1，TP=1、BF16 模型。'
           'CUDA 使用冻结的 BF16 调优配置；FP8 使用冻结的混合选择配置。'
           '每份轨迹包含三批不同但各后端一致的合成 token 请求；每请求固定输出 32 token，'
           '先完成两批同形状预热。每份主采集有 3 次 Prefill、93 次 Decode。'
           'Prefill 预算固定为 2048，关闭动态配额、多步 Decode 和 mixed Graph；'
           'Graph 版本保留 Decode CUDA Graph，eager 版本供逐层分析。','',
           '采用全进程采集（含初始化），**下表只统计 `nsys_capture` NVTX 范围**。'
           '开启 `--cuda-graph-trace=node`，可在 Windows 展开 Graph 内核。'
           '未添加逐模块 CUDA Event；eager 版本添加逐层/投影 CPU NVTX 标记。'
           'Nsight 插桩和 eager hooks 有开销，这些数字用于归因，不替代无 profiler 的吞吐数据。','',
           '## 时间分布','',
           'GPU 活动时间为 kernel/memcpy/memset 区间的并集；GPU 空闲指窗口内未记录到这些活动的时间，'
           '不等于硬件 occupancy。kernel 总时间单独求和，不能与嵌套 NVTX 模块时间相加。','',
           '| 场景 | 窗口 ms | kernel 总 ms | GPU 活动占比 | GPU 空闲 ms | kernel 数量 |',
           '|---|---:|---:|---:|---:|---:|']
    for name in cases:
        c=data[name]['capture']
        lines.append(f"| {labels[name]} | {c['wall_ms']:.2f} | {c['kernel_sum_ms']:.2f} | {c['busy_pct']:.1f}% | {c['gpu_idle_ms']:.2f} | {c['kernel_calls']} |")
    lines+=['','## Decode 热点','',
            '下表为每次 Decode forward 的 kernel 时间（总和 / 93），不含 CPU 空闲。'
            'LM head、sampling 通过 CUDA API correlation ID 关联到 NVTX 调用，'
            '其余按真实 kernel 名称分类。Graph 内部没有逐层 CPU NVTX；使用节点和 kernel 名称定位。','',
            '| 场景 | CUDA GEMV ms | CUDA BF16 GEMM ms | FP8 GEMM ms | FP8 量化 ms | 调库线性 ms | LM head ms | Attention ms | Norm ms | Sampling ms |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    cats=['CUDA BF16 GEMV','CUDA BF16 GEMM','CUDA native FP8 GEMM','FP8 activation quantization',
          'Library GEMM / LM head','Library LM head','FlashAttention','RMSNorm','Sampling softmax / argmax']
    for name in cases:
        n=data[name]['phases']['decode']['calls']
        lines.append('| '+labels[name]+' | '+' | '.join(f'{ms(name,"decode",cat)/n:.4f}' for cat in cats)+' |')
    lines+=['','## Prefill 热点','',
            '| 场景 | Prefill kernel 总 ms / forward | 最耗时分类及 ms / forward |','|---|---:|---|']
    for name in cases:
        p=data[name]['phases']['prefill'];n=p['calls']
        top='；'.join(f"{x['name']}: {x['ms']/n:.3f}" for x in p['categories'][:3])
        lines.append(f"| {labels[name]} | {p['kernel_sum_ms']/n:.3f} | {top} |")
    lines+=['','## 主要瓶颈与证据','']
    for name in ('cuda_graph_b1','cuda_graph_b8','cuda_graph_long','fp8_graph_b8'):
        p=data[name]['phases']['decode'];total=p['kernel_sum_ms']
        proj=sum(ms(name,'decode',cat) for cat in ('CUDA BF16 GEMV','CUDA BF16 GEMM','CUDA native FP8 GEMM','Library GEMM / LM head','Library GEMV / LM head'))
        lm=ms(name,'decode','Library LM head');attn=ms(name,'decode','FlashAttention')
        lines.append(f"- **{labels[name]}**：Decode 投影占 kernel 时间 {proj/total*100:.1f}%，"
                     f"LM head {lm/total*100:.1f}%，Attention {attn/total*100:.1f}%。")
    lines+=['',
            '优先优化 Decode 的线性计算，再看仍走调库的词表投影（LM head）。'
            'Qwen3 的大词表会使 LM head 成为独立热点；仅替换 decoder 的 112 个投影不能消除这部分耗时。'
            '当前 BF16 B=8 的窄 M 路径，应优先验证转置映射 MMA、批量 GEMV 的权重复用；'
            'Nsight Systems 只能证明它们耗时，不能单凭 kernel 时长断定是显存带宽、低 occupancy 或 Tensor Core stall。','',
            f"B=8 对照中，自研 BF16 Decode 投影约 {cuda_proj/cuda_decode['calls']:.3f} ms/步，"
            f"调库约 {lib_proj/lib_decode['calls']:.3f} ms/步；"
            f"{cuda_decode['calls']} 步累计差约 {cuda_proj-lib_proj:.1f} ms，"
            f"两份轨迹的全部 kernel 总时间差 {data['cuda_graph_b8']['capture']['kernel_sum_ms']-data['torch_graph_b8']['capture']['kernel_sum_ms']:.1f} ms。"
            '因此相对调库的额外 GPU 耗时主要集中在 Decode 投影。不同轨迹的时钟/插桩波动仍需保留。'
            f"本次 FP8 轨迹的量化 kernel 只占全部 kernel 时间约 {quant/data['fp8_graph_b8']['capture']['kernel_sum_ms']*100:.1f}%，不是最大热点。",'',
            '## CPU 等待与真实拷贝','',
            '| 场景 | cudaMemcpyAsync CPU 累计 ms | 实际 GPU memcpy/memset 累计 ms | bytes（含 memset） |',
            '|---|---:|---:|---:|']
    for name in cases:
        c=data[name]['capture'];api=sum(x['ms'] for x in c['top_apis'] if 'MemcpyAsync' in x['name'])
        lines.append(f"| {labels[name]} | {api:.2f} | {c['copy_ms']:.3f} | {c['copy_bytes']} |")
    lines+=['',
            '`ModelRunner.run()` 每步在采样后执行 `.tolist()`，取回 token 会等待 GPU。'
            '因此 CUDA API 的长持续时间不能直接解释为 PCIe 传输慢：必须对照 GPU copy 轨道及前面的计算。'
            'CPU NVTX 也包含异步提交或等待，不能直接当作纯 CPU 计算成本。','',
            '## CPU NVTX 与 CUDA API（CUDA B=8 Graph）','',
            '| CPU 范围/API | 次数 | 累计 ms |','|---|---:|---:|']
    for x in data['cuda_graph_b8']['cpu_nvtx'][:8]+data['cuda_graph_b8']['capture']['top_apis'][:6]:
        lines.append(f"| `{x['name']}` | {x['calls']} | {x['ms']:.3f} |")
    lines+=['','## 在 Windows 查看轨道图','',
            '报告由 Nsight Systems 2026.5.1 生成，建议使用同版本或更新的 Windows Nsight Systems。'
            'WSL `/mnt/d/...` 对应 Windows `D:\\...`，直接打开下列 `.nsys-rep`：','']
    for name in cases:
        target=f'../{root.as_posix()}/{name}.nsys-rep'
        lines.append(f'- [{name}.nsys-rep]({target}) — {labels[name]}')
    lines+=['',
            '1. 在 NVTX 轨道找到 `nsys_capture`，放大其中一个 `request_batch/rep1/...`。启动和预热事件不计入结论。',
            '2. 找到 `ModelRunner.run [prefill]` 和后续 `ModelRunner.run [decode]`，对照 CPU CUDA API 与 GPU kernel 轨道。',
            '3. Graph 版本展开 CUDA Graph 节点，优先看 `linear_gemv_kernel`、`gemm_kernel`、`fp8_kernel`、`pack_rows_kernel` 及 cuBLAS/CUTLASS kernel。',
            '4. 在 eager 版本展开 `Qwen3.model.layers.*.self_attn.qkv_proj`、`o_proj`、`mlp.gate_up_proj`、`down_proj`。CPU 范围只代表调用，GPU kernel 通过关联的 launch 确认。',
            '5. 查看 `Qwen3.compute_logits` 对应的 LM head kernel，以及 `Sampler.forward` 后的 DtoH copy/API 等待。区分 Graph 内 kernel 间隙与相邻 Decode step 之间的空闲。','',
            '每份 `.analysis.json` 保存分类、逐阶段、热点 kernel、CUDA API、copy 大小和相邻 Decode 的 GPU 间隙；'
            '`.sqlite` 可继续查询，`.json` 保存参数与输出，`commands.json` 保存完整命令。','',
            '## 复现','',
            '```bash','bash benchmarks/run_qwen_cuda.sh --nsys --output profiles/qwen_cuda_nsys_NEW_RUN',
            'python -m benchmarks.qwen_cuda_nsys_report --root profiles/qwen_cuda_nsys_NEW_RUN','```','',
            '首轮因旧 profiling 脚本的模块导入歧义而失败，保留在 profiles/qwen_cuda_nsys_20261116；'
            '有效轨迹位于带 `_v2` 的目录。原推理性能实验和本次插桩分析分别保留。']
    out=Path('benchmarks/_qwen_cuda_nsys_20261116.md')
    out.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'README.md').write_text(('\n'.join(lines)+'\n').replace(f'../{root.as_posix()}/',''),encoding='utf-8')
    print(out)
    for name in cases:
        c=data[name]['capture']
        print(name,'busy%',round(c['busy_pct'],1),'links',data[name]['direct_kernel_nvtx_links'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('profiles/qwen_cuda_nsys_20261116_v2'))
    main(p.parse_args().root)
