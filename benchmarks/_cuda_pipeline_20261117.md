# CUDA 并行度、每线程工作量与流水调优（2026-11-17）

**本轮启用 Qwen FP16 的四个 B=8 投影配置，以及实验 FP8 的 M=128 QKV 配置。**
FP16 O/down 的配对延迟约降低 71%/73%，新 Decode cfg26 的 FP16 gate-up 约降低 8%，FP8 QKV 约降低 12%。
当前默认 BF16 路径没有通过本轮替换门槛，维持原冻结选择；Prefill 的更深流水和备用 swizzle 未部署。

## 代码与数据

- [Decode CUDA](../nanovllm/layers/csrc/cuda_decode.cu)：新增 cfg24–37，改变 BN/BK、1/2 个 warp、2/3/4 阶段和 Split-K；保留原 cfg0–23。提供相同几何的禁用 swizzle 诊断入口。
- [GEMM CUDA](../nanovllm/layers/csrc/cuda_gemm.cu)：新增 cfg24–35，覆盖小 M 流水和备用 A swizzle，保留原配置编号。
- [FP8 CUDA](../nanovllm/layers/csrc/cuda_fp8_gemm.cu)：新增 cfg28–33，调整 BM、warp tile 和流水阶段。
- [运行时分派](../nanovllm/layers/cuda_linear.py)按后端、dtype 和设备加载冻结配置；[FP16 selector](../nanovllm/layers/cuda_decode_sm120_fp16.json)、[FP8 selector](../nanovllm/layers/cuda_fp8_sm120.json)已打包。未匹配尺寸继续使用原默认分派。
- [配对测试代码](cuda_pipeline_study.py)、[GPU 回归](../tests/test_cuda_pipeline.py)。
- [完整计时与计数器表](../profiles/cuda_pipeline_20261117/REPORT.md)、[汇总 JSON](../profiles/cuda_pipeline_20261117/summary.json)、[原生 NCU 目录](../profiles/cuda_pipeline_20261117/ncu/)。

环境为 RTX 5060 Ti / 36 SM / WSL2。所有 CUDA kernel 使用当前 stream。
本轮只测试算子，没有模型吞吐或 cuBLAS 对比结论。DeepSeek 使用真实矩阵尺寸的独立浮点算子，不代表当前 INT4 MoE 实际部署路径。

## 方法与独立验证

18 个目标算例：Qwen B=8 四投影 × FP16/BF16、两种精度的 M=512 gate-up、DeepSeek B=8 三矩阵 × FP16/BF16，以及 FP8 M=128 QKV/gate-up。
共筛选 1176 个 resident 配置；seed17 用于筛选，42/99/123 用于独立验证。
相同 seed 使用相同随机输入和权重值，随机交错新旧 Graph 的执行顺序；输出预分配，FP8 包含 pack，Split-K 包含 finish。
resident 是同权重 32 次操作；rotating28 是 28 份独立权重。

验证共 192 组配置/种子/工作集记录，固定几何的附加对照为 102 组。正式时延来自无 profiler 的 Graph replay；NCU 使用 kernel replay，只用于计数器诊断。
每个 NCU 条件三次重复，包含该操作全部 kernel；保留原始单位 CSV、命令、检查结果和压缩 `.ncu-repz`。
正常对照和附加对照合计 87 份原生报告、111 个 kernel 实例。

替换门槛在独立验证前固定：rotating28 的逐 seed 新/旧比值中位数 ≤0.97，各 seed 的 rotating 回退不超过 2%，resident 回退不超过 5%。
表中 μs 为各方案的三 seed 中位数；百分比为逐 seed 配对比值的中位数，因此不一定等于表中两个 μs 数字直接相除。

| 算例 | 原配置 → 候选 | resident μs | rotating28 μs | 部署状态 |
|---|---|---:|---:|---|
| FP16 B8 QKV | GEMM10 → Decode2 | 22.118 → 6.455 | 24.185 → 22.225 | 启用 |
| FP16 B8 O | GEMM10 → Decode7 | 43.192 → 4.704 | 43.431 → 12.712 | 启用 |
| FP16 B8 gate-up | GEMM10 → Decode26 | 30.136 → 9.271 | 34.680 → 32.045 | 启用 |
| FP16 B8 down | GEMM10 → Decode7 | 63.506 → 6.068 | 63.488 → 17.210 | 启用 |
| FP8 M128 QKV，含 pack | FP8 cfg3 → cfg0 | 21.804 → 18.921 | 22.330 → 19.555 | 仅 cuda_fp8 后端启用 |
| BF16 B8 QKV | GEMM22 → Decode2 | 12.862 → 6.655 | 21.967 → 21.479 | 不替换，配对中位数改善不足 3% |
| BF16 B8 gate-up | GEMM22 → Decode11 | 15.450 → 9.949 | 32.614 → 31.322 | 不替换，配对中位数改善约 2.3% |
| BF16 B8 down | Decode8 → Decode7 | 6.582 → 6.978 | 18.249 → 18.113 | 不替换，resident 回退 |

两种精度的 M=512 gate-up 保留 cfg10；BF16 B8 O 保留 Decode5；FP8 M128 gate-up 保留 cfg3。
DeepSeek 浮点尺寸候选的 rotating28 延迟改善约 28%–49%，但未修改 DeepSeek 推理接入或 INT4 分派。

## 一个 thread 处理更多数据：收益有条件

Decode 的 cp.async 仍以 16 字节为单位。warp 数减少时，每线程循环承担更多向量加载，并处理更多 MMA fragment；读写两侧保持一致的 XOR swizzle。
固定 BF16 B8 O 的 BN=64/BM=8/BK=64、两阶段、无 Split-K，比较 4 warp 与 2 warp：

| 配置 | warp/块 | 每 warp 的权重行数 | registers/thread | NCU occupancy | shared-load conflicts | resident / rotating28 μs |
|---|---:|---:|---:|---:|---:|---:|
| Decode1 | 4 | 16 | 54 | 8.28% | 0 | 8.371 / 12.153 |
| Decode26 | 2 | 32 | 72 | 4.17% | 0 | 13.412 / 14.177 |

每线程承担更多工作在该形状上增加了寄存器和串行工作量，同时减少可用 warp，结果变慢；这组对照没有 bank conflict，不能再把差距归因于 shared 冲突。
FP16 gate-up 的 cfg26 相对旧 GEMM10 有综合收益，包含 tile、NT MMA 映射、有效行利用和寄存器等变化，不能把其全部收益归因于减少 warp 数。

## 流水深度与 shared bank conflict

固定 Decode26 的几何，只将两阶段改为三/四阶段，resident 为 13.412 / 13.421 / 13.809 μs，rotating28 为 14.177 / 13.553 / 13.891 μs。
它们均未超过原 O 投影 Decode5 的 7.344 / 11.701 μs；增加缓存槽数量不能替代合适的线程划分和 K 分块。

固定 Decode26 几何和线程数，关闭 swizzle 后：

- shared-load conflicts：0 → 286720（同权重预热 NCU 三次中位数）。
- resident：13.412 → 17.733 μs；rotating28：14.177 → 18.197 μs。
- 该对照支持保留 XOR 布局；正常布局仍未优于原 O 配置。

Prefill BF16 M512 gate-up 保留 cfg10 几何，仅改 A swizzle 为 cfg28：

| 指标 | 原 cfg10 | 备用 A swizzle cfg28 |
|---|---:|---:|
| NCU shared-load conflicts，清缓存 | 约 80 万 | 10141 |
| registers/thread | 128 | 130 |
| NCU 寄存器限制下最大 blocks/SM | 4 | 3 |
| resident μs | 167.663 | 171.886 |
| rotating28 μs | 168.624 | 173.211 |

冲突大幅减少，但寄存器跨过驻留阈值，最终略慢。计数器显示存在资源权衡，尚不能把全部时延变化归因于单一指标；该备用布局不部署。
同几何增加到三/四阶段的 Prefill resident 约为 194.304 / 193.331 μs，也不采用。

FP8 gate-up 的小 BM 候选 cfg30 在 resident 为 25.813 μs，优于原 cfg3 的 26.839 μs；但 rotating28 达 35.170 μs，原配置为 28.033 μs。
缩小 BM 增加 M 方向 tile 数，也增加权重服务多个 tile 的工作；本轮证据不足以将该回退唯一归因于 DRAM，保留原 cfg3。

## 硬件计数器验证

FP16 down 的主 kernel：grid 从 8 增至 64 CTA，registers/thread 从 128 降至 54，清缓存 DRAM sustained-peak 比例从 22.52% 升至 84.54%，shared-load conflicts 从 49152 降至 0。
完整独立时延包含新增 finish，仍从约 63.5 μs 降至 17.2 μs；不能通过相加 NCU replay 时间来验证这个收益。

DeepSeek BF16 expert gate-up 浮点诊断：grid 22 → 88 CTA，registers/thread 128 → 54，DRAM 59.14% → 89.67%，shared-load conflicts 90112 → 0。
其 Tensor 活动从 32.48% 降至 6.39%，但算子更快：原大 BM 含无效行运算，因此 Tensor 活动百分比不能单独代表有效计算效率。

FP16 gate-up 从 GEMM10 改为 Decode26：grid 48→96 CTA，registers/thread 128→72，
清缓存 DRAM sustained-peak 比例 80.42%→87.79%，shared-load conflicts 98317→3。
该变化包含 tile、MMA 映射和线程工作量调整，不能单独归因于减少 warp。

FP8 QKV 从 cfg3 改为 cfg0：主 kernel 的 grid 64→128 CTA，registers/thread 119→70，
NCU occupancy 14.69%→27.84%，Tensor 活动 50.37%→60.13%，DRAM 48.93%→55.29%。
shared-load conflicts 则从 0 增至 1986，仍取得完整操作延迟收益；不能以冲突计数单独判优。
pack 保持原实现并计入正式时延。逐 kernel 原始单位与计数器见完整报告与原生文件。

## 验证与复现

`--pipeline-pytest` 最终回归 **15 passed in 24.98s**。

新增和原有 GPU 回归包含：FP16/BF16 边界尺寸、短 K、多阶段预取、非默认 stream、Graph 重放改变输入、swizzle/无 swizzle 一致性、FP8 动态量化，以及冻结 selector 的真实调用。
保留现有 Qwen3-0.6B 默认 BF16 的八请求生成回归；本轮没有模型质量或模型吞吐结论。
Compute Sanitizer 在本机的历史 WDDM 初始化问题尚未解决，本轮不声称通过 memcheck。

```bash
# 一次运行筛选、独立验证、计数器与固定几何对照；使用新目录
bash benchmarks/run_qwen_cuda.sh --pipeline-study --output profiles/cuda_pipeline_NEW
# 只重建已有数据报告
bash benchmarks/run_qwen_cuda.sh --pipeline-study --mode report --output profiles/cuda_pipeline_20261117
# GPU 回归
bash benchmarks/run_qwen_cuda.sh --pipeline-pytest
```

运行时基线在首次执行时保存到 snapshot.json；本次旧分派和原始计时留在该目录，不覆盖早前的 cuda_operator/cfg22 报告。
