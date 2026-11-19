# CUDA 优化学习索引：Qwen3 down 投影（2026-11-17）

主线固定 **FP16、M=8、N=1024、K=3072，Y=XWᵀ**。
旧运行时为 GEMM cfg10，当前运行时为 Decode cfg7。这条主线具备旧版 Systems、
新旧原生 Compute、独立三种子计时，以及本次补采的新旧同条件 Systems。

这些文件不是“每个提交都各有一套报告”：旧代码保留在当前源码中，部分新旧比较是后来
同条件重放；以下逐项注明。最新 M1～4 warp GEMV 没有改变这个 M8 的分派，
因此 Decode7 仍是这个固定算例的当前版本。

## 1. 先打开旧版 Systems，判断该优化什么

- [当时采集的 Graph 时间线](../profiles/cuda_operator_20261117/nsys_graph.nsys-rep)
- [当时采集的 eager 时间线](../profiles/cuda_operator_20261117/nsys_eager.nsys-rep)
- [解析结果](../profiles/cuda_operator_20261117/nsys_graph.analysis.json)
- [原始独立计时](../profiles/cuda_operator_20261117/timing.json)

在 NVTX 中定位：

```text
op_capture/qwen_fp16_m8_down/rotating28/graph
op_capture/qwen_fp16_m8_down/resident/graph
```

看 GPU 轨道中的 `gemm_kernel`，不是将整个进程的 kernel 百分比当作该操作的耗时。
resident 有 96 次操作，rotating28 有 84 次。旧 graph 的 rotating28 kernel 平均约
62.874 μs，间隙约 0.022 μs；当时的独立三种子计时约 63.143 μs/操作。
在这个 Graph 条件下，优先优化 kernel 本体，有比优化 Python 发射间隙更直接的依据。
eager 时间线用来区分发射间隙影响，不能与 Graph 数字混成一组性能结果。

## 2. 打开原生 Compute，确认旧 kernel 的资源与工作划分

- [旧 GEMM10：r0](../profiles/cuda_pipeline_20261117/ncu/qwen_fp16_m8_down_baseline_all_r0.ncu-repz)
- [新 Decode7：r0](../profiles/cuda_pipeline_20261117/ncu/qwen_fp16_m8_down_winner_all_r0.ncu-repz)
- 同目录还有 r1/r2，分别为同一条件的另外两次采集。
- [旧原始数值与单位](../profiles/cuda_pipeline_20261117/ncu/qwen_fp16_m8_down_baseline_all_r0.analysis.json)
- [新原始数值与单位](../profiles/cuda_pipeline_20261117/ncu/qwen_fp16_m8_down_winner_all_r0.analysis.json)

这是后续 pipeline 实验中对**保留的旧 kernel 和新候选进行的同条件采集**，
不是旧版最初部署时就已有的 NCU 记录。`all` 表示 NCU 每个 replay pass 清缓存。
不要直接拿它与 `none` 的 warm-cache capture 比较。

按 Launch Statistics → Occupancy → Memory Workload → Compute/Warps 的顺序看：

| 观察项，主 kernel | 旧 GEMM10 | 新 Decode7 | 能支持的判断 |
|---|---:|---:|---|
| grid CTA 数 | 8 | 64 | 旧 grid 在 36 SM 设备上过小；新映射/Split-K 增加可分配工作 |
| threads/CTA | 128 | 128 | 这次不是单纯增加每块线程数 |
| registers/thread | 128 | 54 | 新主 kernel 的寄存器压力下降 |
| 动态 shared，字节/CTA | 24576 | 18432 | 每块暂存资源下降 |
| DRAM sustained-peak 比例 | 22.52% | 84.54% | 新主 kernel 的实测权重服务吞吐更高 |
| shared-load conflicts | 49152 | 0 | 本组新映射的 shared 读取没有观测到冲突 |

在新报告中还必须打开 `finish_kernel`：其 grid 为 32，256 threads/CTA，
负责累加四份 FP32 部分和并写回 FP16。不能只看新主 kernel 而漏掉这一步。

## 3. 对照源码，理解改变了什么

- [旧源码](../nanovllm/layers/csrc/cuda_gemm.cu)：普通 `dispatch` 中
  `CASE(10,64,128,32,32,64,2)`；不要误读后面诊断入口里同为 10 的编号。
- [新源码](../nanovllm/layers/csrc/cuda_decode.cu)：`MMA_CONFIGS` 中
  `X(7,64,8,64,4,2,4)`，以及 `decode_mma_kernel`、`stage`、`finish_kernel`。

| 旧问题 | 新做法 | 预期减少的开销 |
|---|---|---|
| 请求方向 BM=64，实际只有 8 行 | 将 W 的输出通道放在 MMA m16 轴，请求放在 n8 轴，计算 Yᵀ 后映射写回 | 填充行的无效计算和相应资源占用 |
| 只有 8 个输出 tile | BN=64，外加 K 方向分成 4 份 | 增加 CTA 数，改善工作分配；代价是部分和存取和额外 finish |
| 较高的每线程寄存器与 shared 开销 | 适合小 M 的 tile 和 fragment 划分 | 减轻主 kernel 资源压力 |
| shared 读写映射存在冲突 | 加载与 ldmatrix 地址采用一致的 XOR 布局 | 减少 shared 服务的额外波次 |

这里包含映射、tile、Split-K、布局等**综合改变**。现有这组前后报告不能分别算出
“转置映射单独贡献多少、Split-K 单独贡献多少”；不要把相关计数器变化当成各项独立收益。
cp.async 和双缓冲在旧 GEMM10 中已经存在，不能把这次加速全归因于新引入异步复制。

## 4. 看当前新旧同条件 Systems，检查收益有没有被额外 kernel 抵消

- [本次补采：旧与当前版本在同一文件](../profiles/cuda_down_history_20261117/down_fp16_m8_old_current_graph.nsys-rep)
- [解析结果](../profiles/cuda_down_history_20261117/analysis.json)
- [输入、精度与代码哈希](../profiles/cuda_down_history_20261117/trace.meta.json)
- [补采代码](cuda_down_history.py)

该文件是 **2026-11-17 当前源码重放保留旧版本和当前版本**，不是伪装成历史现场的 capture。
相同 seed42、输入/权重值与操作次数；初始化、FP64 参考和预热在正式范围之外。

```text
op_capture/qwen_fp16_m8_down_baseline/rotating28/graph
op_capture/qwen_fp16_m8_down_current/rotating28/graph
```

两组另有对应的 resident 范围。旧 rotating28 为 84 次单 kernel 操作；
新 rotating28 为 84 次操作、168 个 kernel，即主 kernel/finish 各 84 次。
resident 则是旧 96 个、新 192 个 kernel。四个范围合计 540 个 kernel，已核对。
先观察 `gemm_kernel` 如何变成 `decode_mma_kernel + finish_kernel`，再看两者总时间和间隙。

正式性能仍读独立验证：

- [三种子独立计时](../profiles/cuda_pipeline_20261117/validation.json)，搜索 `qwen_fp16_m8_down`。
- [完整表](../profiles/cuda_pipeline_20261117/REPORT.md)
- [实验说明](_cuda_pipeline_20261117.md)

resident **63.506→6.068 μs**，rotating28 **63.488→17.210 μs**，
rotating 的配对延迟降幅约 **72.9%**；完整操作包含 finish。
不能将这些独立计时与 NCU replay 或不同日期 Systems 的时间相减来构造收益。

复现补采：

```bash
bash benchmarks/run_qwen_cuda.sh --down-history --output profiles/cuda_down_history_NEW
```

## 5. 再看两个控制变量例子，学习为什么有些改动不部署

以下仍是 down 投影，但 **dtype/M 不同，不是上面 FP16 M8 的连续性能版本**。

### 同几何 swizzle 对照：BF16 M8 down

- [无 swizzle 原始 NCU CSV](../results/cuda_decode_reuse_20261116/counters_swizzle/cfg23.csv)
- [有 swizzle 原始 NCU CSV](../results/cuda_decode_reuse_20261116/counters_swizzle/cfg8.csv)
- [独立 swizzle 计时](../results/cuda_decode_reuse_20261116/swizzle_timing.json)
- [当时的实验说明](_cuda_decode_reuse_20261116.md)
- [当时模型旧 Systems](../profiles/cuda_decode_reuse_20261116/old_b8.nsys-rep)
- [当时模型新 Systems](../profiles/cuda_decode_reuse_20261116/reuse_b8.nsys-rep)

这次保留相同 tile、Split-K 和归约，只改 swizzle；冲突 516330→245，
独立旋转权重计时 18.963→17.382 μs。它比综合改变更适合学习单因素归因。
**当时 NCU 只导出了 CSV/日志，没有保存原生 .ncu-rep/.ncu-repz，不能在 NCU GUI 中当原生报告打开。**
模型 Systems 是整模型、BF16、B8 的轨道，不能当作 FP16 固定算子的配对计时。

### 去 shared/块级同步：BF16 M4 down

- [旧 shared SIMT 原生 NCU](../profiles/cuda_warp_gemv_20261117/ncu/qwen_bf16_m4_down_shared_r0.ncu-repz)
- [新 warp 原生 NCU](../profiles/cuda_warp_gemv_20261117/ncu/qwen_bf16_m4_down_warp_r0.ncu-repz)
- [计时与计数器表](../profiles/cuda_warp_gemv_20261117/REPORT.md)
- [实现说明](_cuda_warp_gemv_20261117.md)

shared SIMT cfg17 → warp cfg40：静态 shared 2112→0 字节，barrier 等待指标归零，
resident 13.816→7.027 μs；但实际已部署的 MMA cfg8 为 5.556 μs，因此新 warp 不替换它。
这是“相对实验基线改善，不代表优于当前最佳路径”的例子。
本轮 warp 实验只有 NCU 和独立计时，没有相应的历史 Systems 文件。

## 更早的工程起点

如果想从项目最早的通用 GEMM 开始看：

- [2026-11-15 通用 GEMM Systems](../results/cuda_gemm_profile/timeline.nsys-rep)
- [恢复计数器后的 FP16 1024³ NCU](../results/cuda_gemm_profile/ncu_2026_3_1/ncu_cuda_fp16_1024_cfg10.ncu-repz)
- [当时的归因报告](_cuda_gemm_profile_20261115.md)

它们的矩阵是 1024³/4096³，适合学习最初的 GEMM 与 cuBLAS 差距和固定开销诊断；
**不是 Qwen down 的相同尺寸，不能把它们放进上述 63.5→17.2 μs 的同形状性能曲线。**
