# Qwen3-0.6B 新 CUDA 算子：Nsight Systems 分析（2026-11-16）

## 测量范围

RTX 5060 Ti / SM120，Nsight Systems 2026.5.1，TP=1、BF16 模型。CUDA 使用冻结的 BF16 调优配置；FP8 使用冻结的混合选择配置。每份轨迹包含三批不同但各后端一致的合成 token 请求；每请求固定输出 32 token，先完成两批同形状预热。每份主采集有 3 次 Prefill、93 次 Decode。Prefill 预算固定为 2048，关闭动态配额、多步 Decode 和 mixed Graph；Graph 版本保留 Decode CUDA Graph，eager 版本供逐层分析。

采用全进程采集（含初始化），**下表只统计 `nsys_capture` NVTX 范围**。开启 `--cuda-graph-trace=node`，可在 Windows 展开 Graph 内核。未添加逐模块 CUDA Event；eager 版本添加逐层/投影 CPU NVTX 标记。Nsight 插桩和 eager hooks 有开销，这些数字用于归因，不替代无 profiler 的吞吐数据。

## 时间分布

GPU 活动时间为 kernel/memcpy/memset 区间的并集；GPU 空闲指窗口内未记录到这些活动的时间，不等于硬件 occupancy。kernel 总时间单独求和，不能与嵌套 NVTX 模块时间相加。

| 场景 | 窗口 ms | kernel 总 ms | GPU 活动占比 | GPU 空闲 ms | kernel 数量 |
|---|---:|---:|---:|---:|---:|
| 调库，B=8 / L=128，Graph | 552.99 | 481.54 | 87.3% | 70.30 | 36096 |
| CUDA BF16，B=1 / L=128，Graph | 494.38 | 367.95 | 74.6% | 125.37 | 38703 |
| CUDA BF16，B=8 / L=128，Graph | 637.80 | 528.44 | 83.0% | 108.13 | 36096 |
| CUDA BF16，B=1 / L=1024，Graph | 534.42 | 456.74 | 85.7% | 76.59 | 38703 |
| FP8 混合，B=8 / L=128，Graph | 456.88 | 371.59 | 81.5% | 84.44 | 41640 |
| CUDA BF16，B=8 / L=128，eager | 2541.37 | 556.29 | 21.9% | 1983.92 | 35817 |

## Decode 热点

下表为每次 Decode forward 的 kernel 时间（总和 / 93），不含 CPU 空闲。LM head、sampling 通过 CUDA API correlation ID 关联到 NVTX 调用，其余按真实 kernel 名称分类。Graph 内部没有逐层 CPU NVTX；使用节点和 kernel 名称定位。

| 场景 | CUDA GEMV ms | CUDA BF16 GEMM ms | FP8 GEMM ms | FP8 量化 ms | 调库线性 ms | LM head ms | Attention ms | Norm ms | Sampling ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 调库，B=8 / L=128，Graph | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.6516 | 0.8066 | 0.5780 | 0.0996 | 0.0960 |
| CUDA BF16，B=1 / L=128，Graph | 2.3630 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7786 | 0.3555 | 0.0988 | 0.0815 |
| CUDA BF16，B=8 / L=128，Graph | 0.6538 | 2.4944 | 0.0000 | 0.0000 | 0.0000 | 0.7992 | 0.5633 | 0.1042 | 0.0983 |
| CUDA BF16，B=1 / L=1024，Graph | 2.2652 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7755 | 0.5558 | 0.0930 | 0.0953 |
| FP8 混合，B=8 / L=128，Graph | 0.0000 | 0.0000 | 0.9110 | 0.0842 | 0.9172 | 0.7484 | 0.5333 | 0.0843 | 0.0907 |
| CUDA BF16，B=8 / L=128，eager | 0.7048 | 2.5372 | 0.0000 | 0.0000 | 0.0000 | 0.7514 | 0.6341 | 0.1133 | 0.0917 |

## Prefill 热点

| 场景 | Prefill kernel 总 ms / forward | 最耗时分类及 ms / forward |
|---|---:|---|
| 调库，B=8 / L=128，Graph | 26.338 | Library GEMM / LM head: 22.368；FlashAttention: 1.139；RMSNorm: 0.826 |
| CUDA BF16，B=1 / L=128，Graph | 5.881 | CUDA BF16 GEMM: 4.370；Library LM head: 0.748；FlashAttention: 0.325 |
| CUDA BF16，B=8 / L=128，Graph | 27.165 | CUDA BF16 GEMM: 23.185；FlashAttention: 1.135；RMSNorm: 0.812 |
| CUDA BF16，B=1 / L=1024，Graph | 32.353 | CUDA BF16 GEMM: 24.858；FlashAttention: 4.488；Library LM head: 0.867 |
| FP8 混合，B=8 / L=128，Graph | 16.713 | CUDA native FP8 GEMM: 11.274；FP8 activation quantization: 1.442；FlashAttention: 1.136 |
| CUDA BF16，B=8 / L=128，eager | 32.451 | CUDA BF16 GEMM: 26.991；FlashAttention: 1.261；Library LM head: 1.004 |

## 主要瓶颈与证据

- **CUDA BF16，B=1 / L=128，Graph**：Decode 投影占 kernel 时间 62.7%，LM head 20.7%，Attention 9.4%。
- **CUDA BF16，B=8 / L=128，Graph**：Decode 投影占 kernel 时间 65.5%，LM head 16.6%，Attention 11.7%。
- **CUDA BF16，B=1 / L=1024，Graph**：Decode 投影占 kernel 时间 58.6%，LM head 20.1%，Attention 14.4%。
- **FP8 混合，B=8 / L=128，Graph**：Decode 投影占 kernel 时间 52.9%，LM head 21.7%，Attention 15.4%。

优先优化 Decode 的线性计算，再看仍走调库的词表投影（LM head）。Qwen3 的大词表会使 LM head 成为独立热点；仅替换 decoder 的 112 个投影不能消除这部分耗时。当前 BF16 B=8 的窄 M 路径，应优先验证转置映射 MMA、批量 GEMV 的权重复用；Nsight Systems 只能证明它们耗时，不能单凭 kernel 时长断定是显存带宽、低 occupancy 或 Tensor Core stall。

B=8 对照中，自研 BF16 Decode 投影约 3.148 ms/步，调库约 2.652 ms/步；93 步累计差约 46.2 ms，两份轨迹的全部 kernel 总时间差 46.9 ms。因此相对调库的额外 GPU 耗时主要集中在 Decode 投影。不同轨迹的时钟/插桩波动仍需保留。本次 FP8 轨迹的量化 kernel 只占全部 kernel 时间约 3.3%，不是最大热点。

## CPU 等待与真实拷贝

| 场景 | cudaMemcpyAsync CPU 累计 ms | 实际 GPU memcpy/memset 累计 ms | bytes（含 memset） |
|---|---:|---:|---:|
| 调库，B=8 / L=128，Graph | 431.49 | 1.141 | 1633272 |
| CUDA BF16，B=1 / L=128，Graph | 349.09 | 1.062 | 204552 |
| CUDA BF16，B=8 / L=128，Graph | 473.41 | 1.230 | 1633272 |
| CUDA BF16，B=1 / L=1024，Graph | 401.64 | 1.091 | 261288 |
| FP8 混合，B=8 / L=128，Graph | 323.09 | 0.854 | 1633272 |
| CUDA BF16，B=8 / L=128，eager | 93.21 | 1.158 | 91704 |

`ModelRunner.run()` 每步在采样后执行 `.tolist()`，取回 token 会等待 GPU。因此 CUDA API 的长持续时间不能直接解释为 PCIe 传输慢：必须对照 GPU copy 轨道及前面的计算。CPU NVTX 也包含异步提交或等待，不能直接当作纯 CPU 计算成本。

## CPU NVTX 与 CUDA API（CUDA B=8 Graph）

| CPU 范围/API | 次数 | 累计 ms |
|---|---:|---:|
| `ModelRunner.prepare_decode` | 93 | 24.412 |
| `Sampler.forward` | 96 | 21.038 |
| `Qwen3.compute_logits` | 96 | 6.731 |
| `Scheduler.postprocess` | 96 | 6.432 |
| `ModelRunner.prepare_sample` | 96 | 4.963 |
| `Scheduler.schedule` | 96 | 1.289 |
| `ModelRunner.prepare_prefill` | 3 | 1.052 |
| `cudaMemcpyAsync_v3020` | 1044 | 473.408 |
| `cuLaunchKernel` | 1167 | 11.285 |
| `cudaLaunchKernel_v7000` | 807 | 10.307 |
| `cudaGraphLaunch_v10000` | 93 | 5.360 |
| `cudaEventRecordWithFlags_v11010` | 576 | 1.137 |
| `cuLaunchKernelEx` | 84 | 1.131 |

## 在 Windows 查看轨道图

报告由 Nsight Systems 2026.5.1 生成，建议使用同版本或更新的 Windows Nsight Systems。WSL `/mnt/d/...` 对应 Windows `D:\...`，直接打开下列 `.nsys-rep`：

- [torch_graph_b8.nsys-rep](../profiles/qwen_cuda_nsys_20261116_v2/torch_graph_b8.nsys-rep) — 调库，B=8 / L=128，Graph
- [cuda_graph_b1.nsys-rep](../profiles/qwen_cuda_nsys_20261116_v2/cuda_graph_b1.nsys-rep) — CUDA BF16，B=1 / L=128，Graph
- [cuda_graph_b8.nsys-rep](../profiles/qwen_cuda_nsys_20261116_v2/cuda_graph_b8.nsys-rep) — CUDA BF16，B=8 / L=128，Graph
- [cuda_graph_long.nsys-rep](../profiles/qwen_cuda_nsys_20261116_v2/cuda_graph_long.nsys-rep) — CUDA BF16，B=1 / L=1024，Graph
- [fp8_graph_b8.nsys-rep](../profiles/qwen_cuda_nsys_20261116_v2/fp8_graph_b8.nsys-rep) — FP8 混合，B=8 / L=128，Graph
- [cuda_eager_b8.nsys-rep](../profiles/qwen_cuda_nsys_20261116_v2/cuda_eager_b8.nsys-rep) — CUDA BF16，B=8 / L=128，eager

1. 在 NVTX 轨道找到 `nsys_capture`，放大其中一个 `request_batch/rep1/...`。启动和预热事件不计入结论。
2. 找到 `ModelRunner.run [prefill]` 和后续 `ModelRunner.run [decode]`，对照 CPU CUDA API 与 GPU kernel 轨道。
3. Graph 版本展开 CUDA Graph 节点，优先看 `linear_gemv_kernel`、`gemm_kernel`、`fp8_kernel`、`pack_rows_kernel` 及 cuBLAS/CUTLASS kernel。
4. 在 eager 版本展开 `Qwen3.model.layers.*.self_attn.qkv_proj`、`o_proj`、`mlp.gate_up_proj`、`down_proj`。CPU 范围只代表调用，GPU kernel 通过关联的 launch 确认。
5. 查看 `Qwen3.compute_logits` 对应的 LM head kernel，以及 `Sampler.forward` 后的 DtoH copy/API 等待。区分 Graph 内 kernel 间隙与相邻 Decode step 之间的空闲。

每份 `.analysis.json` 保存分类、逐阶段、热点 kernel、CUDA API、copy 大小和相邻 Decode 的 GPU 间隙；`.sqlite` 可继续查询，`.json` 保存参数与输出，`commands.json` 保存完整命令。

## 复现

```bash
bash benchmarks/run_qwen_cuda.sh --nsys --output profiles/qwen_cuda_nsys_NEW_RUN
python -m benchmarks.qwen_cuda_nsys_report --root profiles/qwen_cuda_nsys_NEW_RUN
```

首轮因旧 profiling 脚本的模块导入歧义而失败，保留在 profiles/qwen_cuda_nsys_20261116；有效轨迹位于带 `_v2` 的目录。原推理性能实验和本次插桩分析分别保留。
