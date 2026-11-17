"""Render the decode reuse experiment, separate from historical measurements."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics


def counters(path: Path) -> dict:
    lines=[line for line in path.read_text().splitlines() if line.startswith('"')]
    return {r['Metric Name']:r['Metric Value'] for r in csv.DictReader(lines)}


def main(root: Path, profiles: Path) -> None:
    deployment=json.loads((root/'deployment.json').read_text())
    models=[[json.loads((root/f'model_v{v}_rep{r}.json').read_text()) for r in range(2)] for v in range(3)]
    times=[[statistics.mean(r['cases'][c]['median_s'] for r in models[v])*1000 for c in range(3)] for v in range(3)]
    lines=['# Qwen3 CUDA Decode：权重复用与共享内存优化（2026-11-16）','',
           '## 结果','',
           'RTX 5060 Ti / SM120，Qwen3-0.6B BF16，TP=1，FP32 累加，无新增量化。'
           '固定 2048-token Prefill 预算、CUDA Graph Decode、每请求 32 个输出 token。'
           '两个新进程轮次按相反后端顺序运行，各点五次不同但后端间一致的合成输入；'
           '表格是两个轮次中位耗时的均值，包含引擎开销，排除初始化/编译/打包。','',
           '| batch / 输入长度 | 调库 ms | 旧调优 CUDA ms | 新 Decode 路径 ms | 相对旧 CUDA 加速 |',
           '|---|---:|---:|---:|---:|']
    for c,case in enumerate(models[0][0]['cases']):
        lines.append(f"| {case['batch']} / {case['input_length']} | {times[0][c]:.3f} | {times[1][c]:.3f} | {times[2][c]:.3f} | {times[1][c]/times[2][c]:.3f}× |")
    gain=(1-times[2][1]/times[1][1])*100
    lines+=['',f'B=8 / L=128 耗时下降 {gain:.1f}%，同期比调库快 {times[0][1]/times[2][1]:.3f}×。'
            '这是本机两个轮次的结果，不推广到其他 GPU/模型。M=1 保留旧 GEMV，'
            '没有针对单请求引入新 kernel；本轮单请求端到端略慢，原因尚未分离，不能宣称单请求加速。','',
            '## 实现','',
            '- **转置映射 MMA**：直接读 W[N,K] 与 X[M,K]，计算 Yᵀ=W·Xᵀ。输出通道放到 m16 轴，8 个请求放到 n8 轴；输出时调整索引，不启动完整矩阵转置。',
            '- **共享内存复用**：权重/激活 tile 由 CTA 协作 16-byte cp.async 搬入 shared；2/3-stage 环形缓冲在当前 tile 计算时预取下一 tile。ldmatrix 地址与 XOR swizzle 对应，覆盖 load 和 store 两侧。',
            '- **并行划分与归约**：输出通道按 warp 划分；窄 N、长 K 的 down 投影选择 Split-K=8（M=16 为 4），FP32 partial 加一个连续访问归约 kernel。临时结果/归约开销全部计入性能。',
            '- **批量 SIMT 候选**：1/4/8 请求共用一次寄存器中的权重向量；激活采用 padded uint4 shared 布局；8/16/32-lane 子组用 shuffle 归约。已验证，但最终配置中没有赢过 MMA，因此没有强行部署。',
            '- **选择规则**：M=1 固定保留旧路径；其他点只在训练轮换权重测试较旧 CUDA 至少快 2% 时启用新路径。验证 seed 不参与配置选择。QKV/gate-up 没有明确收益，保留旧路径。',
            '- M>32 继续使用原 GEMM。原 FP16/BF16/FP8 独立基准源文件和历史 config ID 不变；新配置属于独立 cuda_decode.cu。cfg23 是关闭 swizzle 的诊断对照，不参与调优/部署。','',
            '## 投影级独立验证','',
            'M=8，依次轮换 28 份权重，避免同一份权重反复驻留缓存。训练 seed17，'
            '冻结配置后用独立 seed99 的权重验证，九轮随机后端顺序的 CUDA Graph timing。'
            '单位 μs/投影，含 Split-K 归约和分配对应的 GPU 工作；轮换权重测试仍不是整个模型的缓存状态。','',
            '| 投影 | N,K | 旧 CUDA | 新候选 | 部署策略 |', '|---|---|---:|---:|---|']
    labels={(4096,1024):'QKV',(1024,2048):'O',(6144,1024):'gate/up',(1024,3072):'down'}
    for row in deployment['rows']:
        m,n,k=row['shape']
        if m!=8:continue
        key=f'{m},{n},{k}';h=row['heldout'];cfg=str(row['config'])
        policy=f"新 cfg{cfg}" if deployment['selectors'][key]['mode']=='decode' else '保留旧路径'
        lines.append(f"| {labels[n,k]} | {n},{k} | {h['old']['us']:.3f} | {h[cfg]['us']:.3f} | {policy} |")
    lines+=['',
            'O 投影原先是逐请求 GEMV，新 MMA 将权重 tile 供 8 个请求复用。'
            '这里减少的是逻辑加载/片上搬运；旧加载可能命中缓存，不能把逻辑次数下降等同于 DRAM 流量按同倍数下降。'
            'QKV/gate-up 的原 MMA 本来已复用权重，改映射后没有明显收益，所以保留。','',
            '## Bank-conflict 实测及布局消融','',
            'Nsight Compute 2026.3.1，单次代表性 capture，关闭工具强制锁频。'
            '下表是原始计数，不是时间占比；计数器采集耗时不用于替代无 profiler timing。','',
            '| 配置 | shared load bank conflicts | shared store bank conflicts | shared load wavefronts |',
            '|---|---:|---:|---:|']
    for cfg,label in ((5,'O MMA / swizzle'),(8,'down MMA / swizzle'),(23,'down MMA / 无 swizzle'),(21,'批量 SIMT 候选')):
        metrics=counters(root/f'counters_swizzle/cfg{cfg}.csv')
        values=[metrics.get(metric,'n/a') for metric in ('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
                 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum')]
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    bank=json.loads((root/'swizzle_timing.json').read_text())
    lines+=['',
            f"同一 down tile 的无 swizzle 轮换权重耗时 {bank['23']['us']:.3f} μs，"
            f"swizzle 为 {bank['8']['us']:.3f} μs（下降 {(1-bank['8']['us']/bank['23']['us'])*100:.1f}%）。"
            '该比较保持 tile、Split-K 和归约相同。残余 bank conflicts 并非全部为零；'
            '不能宣称所有形状完全消除了冲突。早期 requests 指标在 SM120 返回 n/a，保留在 counters/，未将其当作零。','',
            '## 端到端结果一致性','',
            '| workload | 新旧 CUDA 完整请求 token 一致 | 两轮旧 CUDA 中位 ms | 两轮新 CUDA 中位 ms |',
            '|---|---:|---|---|']
    for c,case in enumerate(models[0][0]['cases']):
        same=total=0
        for r in range(2):
            for old,new in zip(models[1][r]['cases'][c]['tokens'],models[2][r]['cases'][c]['tokens']):
                for a,b in zip(old,new):same+=a==b;total+=1
        old=[r['cases'][c]['median_s']*1000 for r in models[1]]
        new=[r['cases'][c]['median_s']*1000 for r in models[2]]
        lines.append(f"| {case['batch']} / {case['input_length']} | {same}/{total} | {old[0]:.3f}, {old[1]:.3f} | {new[0]:.3f}, {new[1]:.3f} |")
    lines+=['',
            'GPU 测试覆盖 FP16/BF16、全部 23 个正常配置与无 swizzle 对照、非整 tile 的 M/N/K、K=0，以及改变输入后的 CUDA Graph 重放。'
            '另验证无 tuning 参数时能够加载打包选择器并实际调用新的 O 投影内核。'
            'B=8 的完整请求 token 一致性为 64/80；累加顺序变化后不保证生成序列相同，不能据算术测试宣称端模型质量无损。'
            'Compute Sanitizer 因 WDDM debugger interface 未启用而初始化失败；未获得有效 memcheck 结果，不能把工具内显示的 pytest 通过当作 memcheck 通过。','',
            '## Nsight Systems 新旧对照','']
    lines+=['| B=8 Decode | 旧 CUDA ms/forward | 新 CUDA ms/forward |','|---|---:|---:|']
    traced=[json.loads((profiles/f'{name}.analysis.json').read_text()) for name in ('old_b8','reuse_b8')]
    groups=[('投影',('CUDA BF16 GEMM','CUDA BF16 GEMV','CUDA decode MMA','CUDA split-K reduction','CUDA batched SIMT')),
            ('LM head',('Library LM head',)),('Attention',('FlashAttention',))]
    for label,cats in groups:
        values=[]
        for d in traced:
            p=d['phases']['decode'];values.append(sum(r['ms'] for r in p['categories'] if r['name'] in cats)/p['calls'])
        lines.append(f'| {label} | {values[0]:.4f} | {values[1]:.4f} |')
    lines+=['','只分析预热后的 nsys_capture，包含三批请求、3 次 Prefill、93 次 Decode；'
            '此为插桩归因，不与无 profiler 耗时合并。Windows Nsight Systems 2026.5.1 可直接打开：','',
            f'- [旧 CUDA 时间线](../{profiles.as_posix()}/old_b8.nsys-rep)',
            f'- [新 CUDA 时间线](../{profiles.as_posix()}/reuse_b8.nsys-rep)','',
            '## 使用与复现','',
            '`LLM(model, linear_backend="cuda_decode")` 在 RTX 5060 Ti / BF16 上自动加载打包的冻结选择器；'
            '其他设备/dtype/未测形状保留旧 dispatch。也可显式传 cuda_linear_tuning=deployment.json。'
            '仅支持 TP=1、mixed 执行、无其他权重量化，默认仍为 torch。','',
            '```bash',
            'bash benchmarks/run_qwen_cuda.sh --decode-pytest',
            'bash benchmarks/run_qwen_cuda.sh --decode-study --mode tune --output results/cuda_decode_NEW',
            'bash benchmarks/run_qwen_cuda.sh --decode-study --mode freeze --output results/cuda_decode_NEW',
            'bash benchmarks/run_qwen_cuda.sh --decode-study --mode replay --output results/cuda_decode_NEW',
            'bash benchmarks/run_qwen_cuda.sh --decode-study --mode counters --output results/cuda_decode_NEW',
            'bash benchmarks/run_qwen_cuda.sh --decode-study --mode bank-timing --output results/cuda_decode_NEW',
            'bash benchmarks/run_qwen_cuda.sh --decode-study --mode nsys --output results/cuda_decode_NEW --profile-output profiles/cuda_decode_NEW',
            'python -m benchmarks.cuda_decode_report --root results/cuda_decode_NEW --profiles profiles/cuda_decode_NEW','```','',
            '原始 JSON、冻结配置、counter CSV、sanitize 日志保留在 results/cuda_decode_reuse_20261116/。'
            'Nsight 目录和 counter 目录已有数据时拒绝覆盖，复测请使用新目录。']
    Path('benchmarks/_cuda_decode_reuse_20261116.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('B8 model ms:',[round(times[v][1],3) for v in range(3)])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('results/cuda_decode_reuse_20261116'))
    parser.add_argument('--profiles',type=Path,default=Path('profiles/cuda_decode_reuse_20261116'))
    args=parser.parse_args();main(args.root,args.profiles)
