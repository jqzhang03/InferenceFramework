# DeepSeek-V2-Lite CUDA MLA decode 瓶颈诊断

在 RTX 5060 Ti（SM120）、WSL2、PyTorch 2.8 CUDA 12.8 环境中，使用
16 头、512 维潜变量、64 维 RoPE 和 BF16 分页 KV。每个 CUDA/Triton
对照使用相同 query、页表、页内容和上下文长度。分析对象是单层 MLA decode
算子，包含 query 布局转换、注意力主核和 CUDA 侧的分段合并。

## 独立计时

每个形状使用 3 个种子。以下是 profiler 外 CUDA Event 图重放计时的中位数。

| Decode 行 | 上下文 | Triton µs | CUDA µs |
|---:|---:|---:|---:|
| 8 | 4,096 | 403.47 | 150.14 |
| 32 | 128 | 30.53 | 28.59 |
| 32 | 1,024 | 109.92 | 155.59 |
| 32 | 4,096 | 414.78 | 553.90 |

[独立计时](../profiles/cuda_mla_profile_20261128/timing.json)

## Nsight Systems

按 NVTX 窗口内的 CUDA API correlation ID 归属 GPU 节点。每个图窗口包含
8 次 `cudaGraphLaunch`、64 次完整 MLA 调用、192 个 GPU kernel 节点。
各表值是时间线上每次 MLA 调用的均值，单位为 µs。

| 形状 | 用时 | Triton | CUDA |
|---|---|---:|---:|
| 32 × 128 | 注意力主核 | 26.36 | 19.91 |
| 32 × 1,024 | 注意力主核 | 104.21 | 144.89 |
| 32 × 4,096 | Query 布局转换 | 4.72 | 5.05 |
| 32 × 4,096 | 注意力主核 | 411.89 | 542.24 |
| 32 × 4,096 | Split-KV 合并 | — | 7.67 |
| 32 × 4,096 | GPU 核间空档 | 0.88 | 0.93 |
| 32 × 4,096 | GPU 总跨度 | 417.49 | 555.89 |

32 × 4,096 的 GPU 时间差约 138.40 µs，其中主核多用 130.35 µs，
约占差距的 94%。图提交次数一致，核间空档差约 0.05 µs，合并本身约
7.67 µs。大批量长上下文的优化目标在 `mla_decode_cuda_kernel` 内。

[原生 Nsight Systems 报告](../profiles/cuda_mla_profile_20261128/mla_timeline.nsys-rep) ·
[按窗口拆解的 JSON](../profiles/cuda_mla_profile_20261128/timeline.json)

## Nsight Compute

以下为 32 × 4,096 的两个主核，在输入预热后保持缓存状态的 kernel replay
计数器。Stall 指标使用 NCU 导出的每次发射归一化比值。性能比较使用上面的独立计时。

| 主核指标 | Triton | CUDA |
|---|---:|---:|
| Grid CTA 数 | 32 | 128 |
| 每块已分配共享内存（kB） | 74.75 | 47.36 |
| 实测占用率 | 16.63% | 31.60% |
| 可发射 warp / scheduler / 活跃周期 | 0.295 | 0.249 |
| Tensor 管线活跃率 | 57.96% | 16.84% |
| DRAM 读取（MB） | 151.66 | 152.98 |
| DRAM 吞吐 / 持续峰值 | 89.09% | 69.47% |
| L2 命中率 | 0.06% | 4.42% |
| Long Scoreboard | 0.26 | 8.33 |
| Short Scoreboard | 0.96 | 3.74 |
| Barrier stall | 0.61 | 2.39 |
| Shared load bank conflict 计数 | 0 | 10,760,733 |

**主核主要受 KV 供数和同步限制。** 两边读取约 152 MB，CUDA 的 DRAM
吞吐占比明显低于 Triton，且更多 warp 在等待 L1TEX 依赖（Long Scoreboard）。
源码中每个 16-token tile 都先做全局 KV 加载、BF16 到 FP16 转换和共享内存写入，
然后进行 CTA 同步。Triton 捕获到了全局到共享的 `LDGSTS` 相关 wavefront，
CUDA 这一路径为零。结合更少的可发射 warp 和更低的 Tensor 管线活跃率，
证据指向 CUDA 主核的数据供给和延迟隐藏不足。Long Scoreboard 本身统计
L1TEX 依赖等待。SourceCounters 中，17,401 个 Long Scoreboard 未发射样本有
16,696 个落在 BF16 KV 全局加载的首条依赖指令 `PRMT`，约占 96%。
这是 stall 样本比例。 [CUDA KV 取数](../nanovllm/layers/csrc/cuda_mla_decode.cu#L44) ·
[NVIDIA stall 指标定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

CUDA 的 Short Scoreboard 和共享加载冲突也明显偏高，构成另一处已观测到的
片上访存开销。两套主核的 local load/store sector 均为零。CUDA 的占用率更高，
但每个 scheduler 的可发射 warp 更少。清空缓存的 32 × 4,096 对照复现了
相同趋势。CUDA/Triton 的 DRAM 吞吐分别为 66.87%/89.64%，Long Scoreboard
分别为 8.16/0.28。

短上下文对照中，两边 L2 命中率都超过 99%，CUDA 仍更快。8 × 4,096
时 CUDA 的 128 个 CTA 覆盖更多 SM，Triton 的 8 个 CTA 使得总体 DRAM
吞吐仅约持续峰值的 25.80%。因此，小批量 Split-KV 的设备级并行收益与
大批量长上下文时的单核供数瓶颈可以同时存在。

[CUDA 主核原生报告](../profiles/cuda_mla_profile_20261128/ncu/b32_l4096_cuda_none.ncu-repz) ·
[Triton 主核原生报告](../profiles/cuda_mla_profile_20261128/ncu/b32_l4096_triton_none.ncu-repz)

## 共享内存布局对照

原始 CUDA 主核的 Q 和 KV 共享布局行跨度分别为 576 和 592 个 FP16 元素。
SourceCounters 将约 826 万次额外 shared wavefront 归到 WMMA 的 Q 读取，
约 118 万次归到 KV 读取。诊断副本只把这两个行跨度都改为 584。
每块分配的共享内存、每线程寄存器数量和 grid 保持相同。三种随机输入均通过
位级输出比较。以下计时为诊断会话内的同输入配对结果。

| 32 × 4,096 | 原始 CUDA | 行填充诊断 |
|---|---:|---:|
| 独立图计时中位数（µs） | 578.62 | 510.94 |
| Shared load bank conflict | 10,760,733 | 271,190 |
| Short Scoreboard | 3.74 | 1.13 |
| Long Scoreboard | 8.33 | 8.63 |
| DRAM 吞吐 / 持续峰值 | 69.47% | 78.66% |

共享加载冲突下降 97.5%，图重放用时下降 11.7%，三个种子都改善。
布局是已用对照证实的开销来源。消除大部分共享冲突后，KV 读取依赖
仍是最大的 warp stall。优化方向分别是 Q/KV 共享布局和 KV tile 加载、
类型转换与 MMA 的重叠。

[独立配对计时](../profiles/cuda_mla_profile_20261128/diagnostic/padding_timing.json) ·
[基线 SourceCounters](../profiles/cuda_mla_profile_20261128/diagnostic/cuda_source.csv) ·
[行填充原生报告](../profiles/cuda_mla_profile_20261128/ncu/b32_l4096_cuda_padding_none.ncu-repz)

## 复现

```bash
python -m benchmarks.cuda_mla_profile --output profiles/cuda_mla_profile_run
python -m benchmarks.cuda_mla_profile --output profiles/cuda_mla_profile_run --mode layout
```

原生报告、SQLite、原始单位的 NCU CSV 和精确工具命令都保存在输出目录。
