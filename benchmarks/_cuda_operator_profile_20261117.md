# 自研 CUDA 算子独立 profiling（2026-11-17）

本轮只测现有原生 CUDA 线性算子，不运行模型、调度器或 Attention。
**下一步优先验证小 M 的分派和 tile：FP16 的默认 GEMM、B=8 的 O 投影，以及 DeepSeek 尺寸的小批浮点 GEMM。**
Qwen QKV/gate-up 在轮换权重下主要受显存带宽约束，不能把 warm-cache 的 bank-conflict 改善直接当成实际推理收益。

## 代码与产物

- 测试入口：[cuda_operator_profile.py](cuda_operator_profile.py)。
- 解析与完整表格生成：[cuda_operator_profile_report.py](cuda_operator_profile_report.py)。
- [全部算例、计时和计数器表格](../profiles/cuda_operator_20261117/REPORT.md)。
- [Eager Systems 报告](../profiles/cuda_operator_20261117/nsys_eager.nsys-rep)。
- [CUDA Graph Systems 报告](../profiles/cuda_operator_20261117/nsys_graph.nsys-rep)。
- [Compute 原生报告目录](../profiles/cuda_operator_20261117/ncu/)。
- [汇总 JSON](../profiles/cuda_operator_20261117/summary.json)。

环境：RTX 5060 Ti，36 SM，WSL2；Systems CLI 2026.5.1，Compute CLI 2026.3.1。
原生 Compute 输出是压缩的 `.ncu-repz`，同时保留原始单位 CSV、准确度检查、命令和日志。
运行时 CUDA 源码和冻结 selector 均未修改。

## 测量口径

- 44 个算例：Qwen 的 M=1/8/512 × 四种投影 × FP16/BF16，DeepSeek 的 M=1/8/128 × 三种矩阵 × FP16/BF16，以及两个 M=128 的 FP8 算例。
- DeepSeek 测试采用真实矩阵尺寸，但调用独立浮点算子；不代表当前部署的 INT4 MoE 路径，也不模拟路由或真实专家负载。
- Qwen BF16 匹配当前 RTX 5060 Ti 的冻结 selector。FP16 沿用无匹配 selector 时的默认分派。因此两者的差值包含配置差异，不能直接解释成 dtype 差异。
- resident 连续执行同一权重 32 次；rotating28 轮换 28 份独立权重，每份一次。它们是缓存工作集对照，不是完整模型模拟。
- 无 profiler 的正式计时采用三种 seed，每种九轮 CUDA Graph replay，先取各 seed 中位数再汇总。共 264 组种子/工作集记录；跨 seed 范围保留在 JSON。
- 输入、权重和输出预分配，权重转换在计时外。FP8 包含动态激活量化和 GEMM。Decode Split-K 保留实际 API 内的 partial 分配，Graph 捕获后复用。
- FP64 参考计算和初始化仅在测量范围外。首尾权重的首尾最多八行进行参考检查，并检查这些权重的整个输出是否有限。
- Systems 各有 88 个 `op_capture/` 范围；按同线程 CUDA API 的 correlation ID 关联全部 GPU kernel，核对每个范围的完整调用数。不能只按 GPU 时间是否完全落在 CPU NVTX 区间内筛选。
- Compute 对九个代表算例分别执行同权重预热和逐 pass 清缓存，每组重复三次：54 份报告、66 个 kernel 实例；包括 Split-K finish 和 FP8 pack。
- NCU 使用 kernel replay、clock-control=none。预热条件的缓存状态可能随 pass 变化；NCU 的 kernel 时间不作为独立 benchmark，更不能累加它来代替连续执行耗时。

准确度检查全部通过：FP16、BF16 的最大 relative L2 分别为 0.000216、0.001733；FP8 对解码后量化输入的运算误差最大为 0.001672。FP8 相对原 BF16 输入的量化误差最大约 3.81%，仅是本次随机矩阵结果，不代表模型质量。

## 独立计时的主要结果

单位为每次线性操作的 μs，包含该操作全部自研 kernel。

| 算例 | M,N,K | 模式/cfg | resident | rotating28 |
|---|---|---|---:|---:|
| Qwen BF16 QKV | 1,4096,1024 | GEMV/0 | 6.200 | 22.056 |
| Qwen BF16 QKV | 8,4096,1024 | GEMM/22 | 13.427 | 21.665 |
| Qwen BF16 gate-up | 8,6144,1024 | GEMM/22 | 14.570 | 32.271 |
| Qwen BF16 O | 8,1024,2048 | Decode/5 | 7.979 | 11.937 |
| Qwen BF16 down | 8,1024,3072 | Decode/8，含 finish | 5.754 | 17.216 |
| Qwen FP16 down | 8,1024,3072 | GEMM/10 | 63.132 | 63.143 |
| DeepSeek BF16 expert gate-up | 8,2816,2048 | GEMM/10 | 43.299 | 43.942 |
| Qwen BF16 gate-up | 512,6144,1024 | GEMM/10 | 166.484 | 167.434 |
| Qwen FP8 gate-up，含量化 | 128,6144,1024 | FP8/3 | 27.648 | 28.926 |

## Compute 证据与下一步实验

以下为清缓存条件下三次计数器中位数。DRAM% 和 Tensor% 均为 NCU 对应 sustained-peak 指标，不是端到端利用率。

| 算例/主 kernel | DRAM% | L2 hit% | Tensor% | grid CTA | load bank conflicts |
|---|---:|---:|---:|---:|---:|
| BF16 M=1 QKV GEMV | 86.72 | 2.58 | 0 | 1024 | 0 |
| BF16 M=8 QKV cfg22 | 85.61 | 8.04 | 11.91 | 32 | 32768 |
| BF16 M=8 gate-up cfg22 | 88.11 | 8.30 | 12.14 | 48 | 49160 |
| BF16 M=8 O Decode cfg5 | 71.03 | 11.01 | 4.95 | 16 | 0 |
| BF16 M=8 down Decode cfg8 | 83.72 | 10.63 | 5.91 | 128 | 267 |
| DeepSeek BF16 M=8 expert gate-up | 59.34 | 7.29 | 32.54 | 22 | 90112 |
| BF16 M=512 gate-up cfg10 | 17.66 | 91.31 | 78.35 | 384 | 799005 |
| FP8 M=128 gate-up cfg3 主 GEMM | 54.63 | 69.22 | 59.88 | 96 | 1809 |

1. **先补小 M 的配置与分派。** FP16 B=8 down 当前仍使用 cfg10；其 BM=64，只有八行有效，且 N=1024 只产生八个 CTA。先验证现有 FP16 Decode/GEMV 候选，再冻结独立 FP16 selector，避免先重写计算核心。当前 BF16/FP16 差距只是动机，收益必须通过同 dtype 候选对照确认。
2. **O 投影增加可用并行度。** Decode cfg5 只有 16 个 CTA，对 36 SM 覆盖不足；eligible warp 约 0.141。尝试更小 BN 或适度 Split-K，但必须把 finish、额外 partial 读写和 warm/cold 两种条件一起计时。
3. **DeepSeek 尺寸的小批浮点 GEMM 缩小 BM。** 当前 cfg10 是 BM=64、BN=128，M=8 大量行计算无有效输出，grid 也只有 22。先验证 BM=8/16 和 NT Decode 候选。这只是未来浮点接入的诊断，不是 INT4 专家性能优化结论。
4. **cfg22 和 GEMV 以权重流量为约束。** 冷条件 DRAM 已达到约 86%–88%；进一步优化 shared 布局应同时跑 rotating28。不能因为 QKV 有 32768 次 bank conflict，就把它当成当前 streaming 路径的首要瓶颈。若要明显减少权重字节，需要另行评估量化或提高批内复用；本轮没有验证这些改动。
5. **Prefill 检查 MMA/流水和布局，不优先追求更高 occupancy。** BF16 M=512 的 Tensor 活动约 78.35%，DRAM 仅 17.66%；occupancy 约 29.5%，128 registers/thread。可对照 warp tile、寄存器用量、流水 stage 和 shared 布局。799005 次冲突不等于同等数量的浪费周期，应保留 tile 只改布局进行验证。
6. **FP8 先看主 GEMM。** 本次 Graph resident 轨道中，pack 占两个 kernel 活动时间约 7.75%，主 GEMM 约 92.25%；主 GEMM 清缓存 Tensor 活动约 59.88%，119 registers/thread、96 CTA。优先检查 FP8 tile、寄存器和数据供给，再评估量化融合。本轮 FP8 为 M=128，不能与表中 M=512 BF16 直接比较精度路径快慢。

## Systems 轨道应该看什么

打开 `nsys_eager.nsys-rep` 与 `nsys_graph.nsys-rep`，搜索 `op_capture/`，选择同一 case 和 resident/rotating28 条件。

| resident 算例 | Eager GPU span 内 kernel 活动% | Graph GPU span 内 kernel 活动% |
|---|---:|---:|
| BF16 M=1 QKV | 13.19 | 95.64 |
| BF16 M=8 down | 29.15 | 92.42 |
| BF16 M=512 gate-up | 98.24 | 99.86 |
| FP8 M=128 gate-up | 80.85 | 98.42 |

这些比例只描述首个 kernel 开始到末个 kernel 结束之间的忙闲，并非 SM occupancy；也不是无 instrumentation 的延迟对比。小算子在 eager 的大间隙优先排查 CPU 提交与 launch，而 Prefill 已基本连续执行，应继续查看 kernel 内部。

正式范围内未发现 CUDA memcpy/memset。权重从显存读取属于 kernel 内访存，不会出现在 Memcpy 轨道。CPU 的 `cudaDeviceSynchronize` 等待与 GPU 执行重叠，不应相加。

## 复现与检查

```bash
# WSL 环境，一次生成独立计时与两类原生报告；使用新目录保留旧数据
bash benchmarks/run_qwen_cuda.sh --operator-profile --output profiles/cuda_operator_NEW

# 只测试一个算例；其全部 kernel 都会被 NCU 采集
bash benchmarks/run_qwen_cuda.sh --operator-profile --case qwen_bf16_m8_down --output profiles/cuda_operator_DOWN_NEW

# 从现有 SQLite/CSV 重新汇总，无需 GPU 采集
bash benchmarks/run_qwen_cuda.sh --operator-profile --mode report --output profiles/cuda_operator_20261117

bash benchmarks/run_qwen_cuda.sh --operator-pytest
```

报告解析四项 CPU 测试通过，覆盖多 kernel 的单位转换、排除范围外参考运算、拒绝范围内调库 kernel，以及 GPU 事件越过 CPU NVTX 边界时的 correlation 归属。
本轮没有调库基线，不声称相对 cuBLAS 的效率，也没有对运行时部署配置作出修改。
