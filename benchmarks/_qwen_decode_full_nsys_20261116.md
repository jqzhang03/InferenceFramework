# Qwen3-0.6B + cuda_decode：完整推理 Nsight Systems 分解（2026-11-16）

## 本次配置与范围

当前工作树、RTX 5060 Ti / SM120、BF16、TP=1、linear_backend=cuda_decode。自动加载打包的冻结选择器，112 个 decoder 投影 adapter；M=4/8/16 的 O/down 使用新 MMA，其他形状保留旧 CUDA 路径。LM head 调库，Attention 使用原 FlashAttention。没有额外 FP8 权重量化。每批固定 32 输出 token，合成 token 输入，三批不同输入；两批同形状预热。

主结果恢复当前默认调度开关：multi_step_decode=True（最多4步）、mixed_cudagraph=True、latency/SLO/TPOT-aware scheduling=True、KV swap=True。为控制采集规模，资源上限是 max_num_seqs=16、max_num_batched_tokens=2048、max_model_len=2048、GPU memory utilization=0.55。固定预算对照关闭上述动态配额/multi-step/mixed Graph；eager 轨迹用于模块归因，额外带 CPU NVTX hooks。

Nsight Systems 2026.5.1 全进程采集 CUDA/NVTX/OSRT/cuBLAS，Graph node tracing 开启。启动和预热另有 NVTX 标签，正式推理只统计 nsys_capture。以下 wall time 含插桩开销，不替代无 profiler 的性能结果；CPU 调用时间与 GPU 活动可能重叠。

## 启动、预热和正式推理

| 场景 | 初始化 ms | 两批预热 ms | 三批正式推理 ms | 每批平均 ms | Graph replay（decode/mixed） |
|---|---:|---:|---:|---:|---|
| 默认调度 B=1/L=128 | 34675.44 | 1134.13 | 572.30 | 190.77 | 93/0 |
| 默认调度 B=8/L=128 | 10774.46 | 5190.65 | 657.04 | 219.01 | 93/3 |
| 默认调度 B=1/L=1024 | 8331.79 | 979.36 | 565.96 | 188.65 | 93/0 |
| 固定预算对照 B=8/L=128 | 7481.75 | 1003.49 | 577.31 | 192.44 | 93/0 |
| eager 逐层诊断 B=8/L=128 | 5119.17 | 3019.24 | 2444.05 | 814.68 | 0/0 |

初始化包含加载、CUDA/JIT 初始化和初始图捕获；预热还可能捕获 mixed 图。JIT/磁盘缓存状态会随采集顺序变化，不用初始化列比较不同 batch 的效率。这些成本已单列，正式推理没有重复提示带来的前缀命中。输入直接提供 token IDs，不包含输入文本分词；输出整理包含在正式 wall time 中。

## 整体 GPU 活动与间隙

| 场景 | GPU kernel 总 ms/批 | GPU 活动并集 ms/批 | 无 GPU 活动 ms/批 | GPU 活动比例 | kernel 数 |
|---|---:|---:|---:|---:|---:|
| 默认调度 B=1/L=128 | 114.623 | 114.928 | 75.839 | 60.2% | 38703 |
| 默认调度 B=8/L=128 | 154.059 | 154.422 | 64.592 | 70.5% | 39999 |
| 默认调度 B=1/L=1024 | 149.527 | 149.859 | 38.794 | 79.4% | 39080 |
| 固定预算对照 B=8/L=128 | 147.593 | 147.861 | 44.575 | 76.8% | 38700 |
| eager 逐层诊断 B=8/L=128 | 154.315 | 154.587 | 660.098 | 19.0% | 39723 |

GPU 活动是 kernel/memcpy/memset 时间区间的并集；空白区间包括主机提交、等待、调度等，不能全部叫作 CPU 计算。活动比例不是 achieved occupancy。kernel 时间求和不与 CPU NVTX 时间叠加。

## Prefill / Mixed / Decode 的 wall time 分区

| 场景 | Prefill 次数、ms/批 | Mixed 次数、ms/批 | Decode forward 次数、ms/批 | Runner 外 ms/批 |
|---|---|---|---|---:|
| 默认调度 B=1/L=128 | 3 / 40.488 | 0 / 0.000 | 93 / 147.229 | 3.050 |
| 默认调度 B=8/L=128 | 3 / 39.357 | 3 / 10.676 | 93 / 163.207 | 5.774 |
| 默认调度 B=1/L=1024 | 4 / 42.293 | 0 / 0.000 | 93 / 143.119 | 3.241 |
| 固定预算对照 B=8/L=128 | 3 / 32.267 | 0 / 0.000 | 93 / 154.428 | 5.740 |
| eager 逐层诊断 B=8/L=128 | 3 / 33.053 | 3 / 36.434 | 93 / 739.797 | 5.401 |

次数统计整份轨迹（三批），时间为每批平均。阶段 wall time 来自互不嵌套的 ModelRunner.run NVTX；Runner 外包括请求提交、调度/postprocess、生成结果整理等。Decode forward 次数不同于 engine.step 次数，multi-step 会在一个 step 内执行多次 forward。Mixed 同时处理两类 token，不用人为比例把它拆成纯 Prefill/Decode。

## 全过程各 GPU 部分耗时（ms/批）

| 场景 | 线性投影 | Split-K归约 | Attention | LM head | RMSNorm | RoPE | SiLU | KV写入 | 采样 | 其他 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 默认调度 B=1/L=128 | 72.458 | 0.000 | 10.741 | 23.521 | 2.853 | 1.384 | 0.496 | 0.695 | 2.402 | 0.073 |
| 默认调度 B=8/L=128 | 98.167 | 0.539 | 19.177 | 25.024 | 3.768 | 2.194 | 0.829 | 1.001 | 3.108 | 0.250 |
| 默认调度 B=1/L=1024 | 94.073 | 0.000 | 21.538 | 23.790 | 3.697 | 2.196 | 0.830 | 0.884 | 2.436 | 0.082 |
| 固定预算对照 B=8/L=128 | 94.002 | 0.557 | 18.103 | 24.052 | 3.764 | 2.162 | 0.835 | 0.895 | 3.025 | 0.198 |
| eager 逐层诊断 B=8/L=128 | 98.740 | 0.620 | 18.607 | 24.792 | 4.175 | 2.286 | 0.923 | 0.957 | 3.009 | 0.206 |

线性投影不含 LM head 和 Split-K 归约，避免重复计数。LM head/采样按 launch correlation ID 关联。其他类别按 kernel 名称；RoPE/Embedding 等仅在 eager 的 launch 归属唯一时，将核名映射用于 Graph 分类。其他包括未归类索引、embedding、逐元素等；0 表示该分类未找到匹配 kernel，不意味着某逻辑完全没有成本。

## 各阶段 GPU kernel 组成

| 场景/阶段 | forward 数 | kernel 总 ms/forward | 最耗时分类（占该阶段 kernel 时间） |
|---|---:|---:|---|
| 默认调度 B=1/L=128 / prefill | 3 | 5.8529 | CUDA BF16 GEMM: 74.1%；Library LM head: 12.7%；FlashAttention: 5.6%；RMSNorm: 2.6% |
| 默认调度 B=1/L=128 / decode | 93 | 3.5087 | CUDA BF16 GEMV: 62.6%；Library LM head: 20.9%；FlashAttention: 9.6%；RMSNorm: 2.5% |
| 默认调度 B=8/L=128 / prefill | 3 | 24.9090 | CUDA BF16 GEMM: 84.8%；FlashAttention: 4.6%；Library LM head: 3.1%；RMSNorm: 2.9% |
| 默认调度 B=8/L=128 / decode | 93 | 3.8811 | CUDA BF16 GEMM: 37.1%；CUDA decode MMA: 19.6%；Library LM head: 19.5%；FlashAttention: 14.2% |
| 默认调度 B=8/L=128 / mixed | 3 | 8.8350 | CUDA BF16 GEMM: 74.4%；FlashAttention: 10.4%；Library LM head: 8.6%；RoPE: 2.2% |
| 默认调度 B=1/L=1024 / prefill | 4 | 24.5232 | CUDA BF16 GEMM: 75.9%；FlashAttention: 13.8%；Library LM head: 3.1%；RMSNorm: 2.6% |
| 默认调度 B=1/L=1024 / decode | 93 | 3.7687 | CUDA BF16 GEMV: 59.3%；Library LM head: 19.5%；FlashAttention: 14.6%；RMSNorm: 2.4% |
| 固定预算对照 B=8/L=128 / prefill | 3 | 27.1172 | CUDA BF16 GEMM: 85.2%；FlashAttention: 4.2%；RMSNorm: 3.0%；Library LM head: 2.8% |
| 固定预算对照 B=8/L=128 / decode | 93 | 3.8863 | CUDA BF16 GEMM: 38.8%；CUDA decode MMA: 20.1%；Library LM head: 19.3%；FlashAttention: 14.1% |
| eager 逐层诊断 B=8/L=128 / prefill | 3 | 24.9517 | CUDA BF16 GEMM: 85.0%；FlashAttention: 4.4%；Library LM head: 3.1%；RMSNorm: 2.9% |
| eager 逐层诊断 B=8/L=128 / decode | 93 | 3.8960 | CUDA BF16 GEMM: 37.4%；CUDA decode MMA: 19.6%；Library LM head: 19.3%；FlashAttention: 13.8% |
| eager 逐层诊断 B=8/L=128 / mixed | 3 | 8.5875 | CUDA BF16 GEMM: 74.9%；FlashAttention: 9.9%；Library LM head: 8.8%；RMSNorm: 1.9% |

## 默认 B=8 的主要热点

- Decoder 线性投影：98.706 ms/批，GPU kernel 时间占比 64.1%。
- LM head：25.024 ms/批，GPU kernel 时间占比 16.2%。
- Attention：19.177 ms/批，GPU kernel 时间占比 12.4%。
- RMSNorm：3.768 ms/批，GPU kernel 时间占比 2.4%。
- 采样：3.108 ms/批，GPU kernel 时间占比 2.0%。

GPU 活动比例 70.5%，每批还有 64.592 ms 未记录到 GPU 活动。这些空白需对照主机轨道和 CUDA API 判断，不等于 scheduler 本身耗时。优化后仍应结合投影、词表层以及提交间隙分析；本次未采集硬件 stall/带宽计数器，不作 compute/memory-bound 定论。

## CPU 范围与真实拷贝

| 默认 B=8 CPU 范围/API | 次数 | 累计 ms |
|---|---:|---:|
| `LLMEngine.step` | 30 | 651.044 |
| `ModelRunner.prepare_decode` | 93 | 38.405 |
| `Sampler.forward` | 99 | 22.394 |
| `ModelRunner.prepare_sample` | 99 | 10.024 |
| `Qwen3.compute_logits` | 99 | 7.724 |
| `Scheduler.postprocess` | 99 | 6.586 |
| `ModelRunner.prepare_prefill` | 3 | 1.765 |
| `Scheduler.schedule` | 30 | 0.744 |
| `LLMEngine.add_request` | 24 | 0.113 |
| `cudaMemcpyAsync_v3020` | 1095 | 401.630 |
| `cudaLaunchKernel_v7000` | 819 | 18.817 |
| `cuLaunchKernel` | 1188 | 17.774 |
| `cudaGraphLaunch_v10000` | 96 | 6.400 |
| `cudaEventRecordWithFlags_v11010` | 600 | 2.202 |
| `cudaStreamIsCapturing_v10000` | 898 | 2.123 |
| `cuLaunchKernelEx` | 84 | 1.895 |
| `cudaEventQuery_v3020` | 817 | 1.746 |

默认 B=8 实际 GPU copy/memset 总时长 1.089 ms、1598112 bytes。cudaMemcpyAsync 的 CPU 长调用可能包含采样后 .tolist() 等待先前 GPU 计算；不能按 API 总时长认定 PCIe 瓶颈。LLMEngine.step 是包含模型运行和等待的外层范围，不是纯 CPU 时间。以上 CPU 范围有嵌套，不能相加。

## eager 逐模块 GPU 归因（仅诊断）

| 模块 | 三批 GPU kernel ms | ms/批 |
|---|---:|---:|
| Gate/up projection | 113.462 | 37.821 |
| QKV projection | 76.216 | 25.405 |
| Library LM head | 74.376 | 24.792 |
| Down projection | 65.836 | 21.945 |
| Attention + KV store | 58.879 | 19.626 |
| O projection | 42.567 | 14.189 |
| RMSNorm | 12.524 | 4.175 |
| Sampling softmax / argmax | 9.026 | 3.009 |
| RoPE | 6.857 | 2.286 |
| SiLU | 2.769 | 0.923 |
| Embedding | 0.432 | 0.144 |

eager 的模块 GPU 时间由 CPU NVTX 范围中的 launch correlation 归因，不是 NVTX 范围的 wall time。采用叶子模块归属，避免把 Attention 父范围与 QKV/O 再重复相加。RoPE 是共享模块，轨道名为 shared_rotary_emb。逐层 hooks 和 eager 提交会改变执行节奏，其绝对时间不能直接代替 Graph 版本。

## Windows 时间线与数据

- [decode_default_b1.nsys-rep](../profiles/qwen_decode_full_nsys_20261116/decode_default_b1.nsys-rep)：默认调度 B=1/L=128。
- [decode_default_b8.nsys-rep](../profiles/qwen_decode_full_nsys_20261116/decode_default_b8.nsys-rep)：默认调度 B=8/L=128。
- [decode_default_long.nsys-rep](../profiles/qwen_decode_full_nsys_20261116/decode_default_long.nsys-rep)：默认调度 B=1/L=1024。
- [decode_controlled_b8.nsys-rep](../profiles/qwen_decode_full_nsys_20261116/decode_controlled_b8.nsys-rep)：固定预算对照 B=8/L=128。
- [decode_eager_b8.nsys-rep](../profiles/qwen_decode_full_nsys_20261116/decode_eager_b8.nsys-rep)：eager 逐层诊断 B=8/L=128。

- [分项 CSV](../profiles/qwen_decode_full_nsys_20261116/components.csv)；[汇总 JSON](../profiles/qwen_decode_full_nsys_20261116/summary.json)。

使用 Nsight Systems 2026.5.1 或更新版本。先定位 startup/model_initialization、startup/inference_warmup，再放大 nsys_capture → request_batch/rep1。查看 LLMEngine.step 和 ModelRunner.run [prefill/mixed/decode]，展开 CUDA Graph 节点，对照 decode_mma_kernel、finish_kernel、gemm_kernel、LM head 和 FlashAttention。eager 版本可以展开 Qwen3.model.layers.* 各叶子模块。

原始 SQLite、命令/日志、配置/输出 JSON 和逐阶段 analysis.json 同目录保留。本次为合成 token 的离线分析，无线上到达/SLO 结论；未修改引擎或算子实现。

## 复现

```bash
bash benchmarks/run_qwen_cuda.sh --nsys --suite decode-current --output profiles/qwen_decode_full_NEW
bash benchmarks/run_qwen_cuda.sh --full-nsys-report --root profiles/qwen_decode_full_NEW
```

请使用新目录保留历史。旧 CUDA/FP8 和旧优化测量仍保留各自报告，不与本次数据合并。
