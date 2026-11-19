# M=1～4 的 warp 独立 CUDA GEMV（2026-11-17）

实现位于 [cuda_decode.cu](../nanovllm/layers/csrc/cuda_decode.cu) 的
`decode_warp_gemv_kernel`，通过 `decode_out(..., config)` 调用，新增 cfg38–49，
旧 cfg0–37 的编号和行为保持不变。支持连续、16 字节对齐的 FP16/BF16，
X[M,K]、W[N,K]，M=1～4、K 为 8 的倍数，FP32 累加，使用当前 CUDA stream。

## 线程与数据划分

- 每个 CTA 128 个线程；每个 8/16/32 lane 组负责 1/2 个输出列，并计算全部请求行。
- 每线程以 `uint4` 一次读取 8 个 FP16/BF16 元素；加载的权重向量用于全部 M 行，
  每行激活向量用于该组的 1/2 个输出。
- 使用 1/2 条独立 FP32 累加链，最后通过 `__shfl_down_sync(..., width=LANES)` 归约。
- 不使用 shared memory、`__syncthreads()`、Split-K 部分和缓冲或 finish kernel。
  边界输出采用加载/写回谓词，所有线程仍执行 shuffle；不能提前退出部分 lane。
- M=1～4 分别实例化，避免补齐到 4 行产生无效计算；M>4 明确拒绝。

| cfg | lane 组宽度 | 每组输出数 | 累加链数 |
|---|---:|---:|---:|
| 38 / 39 | 32 | 1 | 1 / 2 |
| 40 / 41 | 32 | 2 | 1 / 2 |
| 42 / 43 | 16 | 1 | 1 / 2 |
| 44 / 45 | 16 | 2 | 1 / 2 |
| 46 / 47 | 8 | 1 | 1 / 2 |
| 48 / 49 | 8 | 2 | 1 / 2 |

普通 `linear_gemv_kernel` 本来就没有 shared memory 和块级屏障；新实现相对它的
主要变化是跨 M 行复用权重、复用 X 计算多个输出，以及 lane/累加链划分。
原 `decode_simt_kernel` 则将 X 放入 shared，每个 K 块有两次块级同步，作为另一条对照。

## 验证方法

`--warp-gemv-pytest` 最终 **29 passed in 67.42s**：24 项 GPU 测试与 5 项 profiling 解析测试。
新路径覆盖 M1～4、FP16/BF16、短 K/非整 tile、非默认 stream、输入及权重改变后的 Graph 重放、
零 K 和 M>4 拒绝；真实 selector 安装测试包含 BF16 M2/M3、FP16 M2/M4，
并保留原默认 BF16 Qwen3-0.6B 八请求生成回归。

[性能代码](cuda_warp_gemv_study.py) 覆盖 Qwen3-0.6B 四投影 × M=1/2/3/4 × FP16/BF16，
共 32 个算例。seed17 筛选 12 个新配置，以当前冻结运行时、普通 GEMV 和 shared SIMT 为对照。
候选在 seed42/99/123 上进行独立验证，每条件 13 轮随机交错的 Graph 计时；
resident 重复 32 次同一权重，rotating28 使用 28 份独立权重，输出预分配。
相同 seed 的各候选接收相同输入和权重值，FP64 精度检查在计时之外。

替换门槛沿用预先固定的规则：逐 seed 的 rotating28 新/旧比值中位数 ≤0.97，
任一 rotating28 回退 ≤2%，任一 resident 回退 ≤5%。门槛比较的是当前运行时配置，
不能仅因快于 shared SIMT 就替换已更快的 MMA 路径。

NCU 对 BF16 M1/M4 down 的 shared 与新 warp 版本各采集三次，保留原生报告、
原始 CSV 单位、精度结果、命令和日志。warm-cache NCU 仅验证资源/等待，
不以 replay 时间作为性能收益。静态/动态 shared 分配与驱动保留的 shared 分开统计。

## 复现与数据

32 个算例完成 452 组 resident 配置筛选（其中 384 组为新 kernel），
以及 600 组独立种子/方案/工作集验证记录。18 个算例通过门槛，
其中 BF16 6 个（M2 QKV/gate-up、M3 四投影），FP16 12 个（M2～4 四投影）。
仅这些尺寸加入 `cuda_decode` 的设备/dtype 冻结 selector；其他已有配置保留。

| 算例 | 新 cfg | resident 原→新 μs | rotating28 原→新 μs | 配对延迟降幅 |
|---|---:|---:|---:|---:|
| BF16 M2 gate-up | 49 | 13.795→8.387 | 35.034→31.226 | 10.89% |
| BF16 M3 O | 38 | 7.679→4.610 | 13.472→11.408 | 15.32% |
| BF16 M3 gate-up | 40 | 19.550→10.043 | 41.778→31.834 | 24.62% |
| FP16 M3 QKV | 48 | 13.438→6.515 | 28.206→21.975 | 22.09% |
| FP16 M4 gate-up | 48 | 25.820→10.600 | 48.328→32.040 | 33.70% |

μs 为各方案三 seed 的中位数；降幅来自逐 seed 配对比值的中位数，不能直接用表中两个 μs 重算。
M1 两种精度均未达到替换门槛，保持普通 GEMV；BF16 M4 保持已有小 M GEMM/MMA。
FP16 M4 的原路径仍为普通 GEMV，因此这组收益不代表超过 BF16 已优化的 MMA 配置，
也不代表同配置的 FP16/BF16 精度差异。

新 kernel 对多请求复用寄存器中的权重，减少重复的 global 加载指令与计算组织开销；
实际 DRAM 流量仍受 L1/L2 命中影响，不能将请求数直接当作 DRAM 流量缩减倍数。
完整报告同时给出 shared SIMT 对照，防止将相比旧默认 GEMV 的全部收益归因于去除屏障。

12 份 NCU 报告全部成功，两组条件各三次重复的中位数如下：

| BF16 down 对照 | 原 shared SIMT → 新 warp | M1 | M4 |
|---|---|---:|---:|
| 静态 shared，字节/CTA | 原→新 | 528→0 | 2112→0 |
| 动态 shared，字节/CTA | 原→新 | 0→0 | 0→0 |
| registers/thread | 原→新 | 40→37 | 38→52 |
| shared-load conflicts | 原→新 | 38→0 | 0→0 |
| barrier stall / issue-active ratio | 原→新 | 7.041377→0 | 5.597539→0 |
| 独立 resident 计时，μs | 原→新 | 6.828→4.663 | 13.816→7.027 |

两组新 kernel 均未观测到 local spilling requests；shared 数字不含驱动保留部分。
barrier ratio 是 issue-active 周期对应的等待 warp 指标，**不是耗时百分比**。
这两组用于验证无 shared/屏障的实现，M1/M4 BF16 候选均未部署：
M1 原普通 GEMV resident 为 4.540 μs，M4 原 MMA 为 5.556 μs，仍优于相应新候选。

```bash
bash benchmarks/run_qwen_cuda.sh --warp-gemv-pytest
bash benchmarks/run_qwen_cuda.sh --warp-gemv-study --output profiles/cuda_warp_gemv_NEW
# 只重建已有数据的报告
bash benchmarks/run_qwen_cuda.sh --warp-gemv-study --mode report --output profiles/cuda_warp_gemv_20261117
```

[完整计时与计数器表](../profiles/cuda_warp_gemv_20261117/REPORT.md)、
[独立验证原始结果](../profiles/cuda_warp_gemv_20261117/validation.json)、
[原生 NCU 文件](../profiles/cuda_warp_gemv_20261117/ncu/)。
历史 `cuda_pipeline_20261117/` 和 `cuda_operator_20261117/` 数据保留，均对应当时源代码和分派。

本轮是算子实验，没有 cuBLAS 或模型吞吐加速结论。现有默认 BF16 模型生成回归用于兼容性，
不代替新 kernel 的模型质量评估。
