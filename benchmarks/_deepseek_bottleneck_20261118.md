# DeepSeek-V2-Lite 推理性能与瓶颈

RTX 5060 Ti，WSL，PyTorch 2.8.0、CUDA 12.8，单卡 TP=1，BF16 激活和 MLA KV，INT4 group 64，流式载入。通过本地 LLM API 测量 eager 固定批次。每个请求输出 4 token，禁用前缀复用、KV swap 和多步 decode。引擎统一使用 365 块 MLA KV 池。

每组配置在同一个模型实例上重复 6 轮并交错执行，固定 prompt、输出 token 和专家路由。每次仍执行真实 router 和 sampler。CUDA 逐专家使用已实现的 W4A16，M1–4 启用完整 expert 融合。两个 dispatch 消融沿用这些 CUDA 算子和专家累加顺序。独立计时与 Nsight Systems 采集分开进行。

## 端到端性能

TTFT 和 TPOT 先取单批请求中位数，再取六轮中位数。输出吞吐包含 prefill。输入为固定种子的合成 token。

| 请求数 × 输入长度 | 路径 | TTFT ms | TPOT ms | 输出 tok/s | 批次总耗时 ms |
|---|---|---:|---:|---:|---:|
| 1 × 64 | Triton 逐专家 | 1,067.38 | 834.01 | 1.09 | 3,674.52 |
| 1 × 64 | CUDA 逐专家 | 909.56 | 656.58 | 1.50 | 2,664.28 |
| 1 × 64 | CUDA 按层打包诊断 | 242.87 | 44.66 | 11.17 | 360.28 |
| 4 × 64 | Triton 逐专家 | 1,246.15 | 811.71 | 4.38 | 3,657.41 |
| 4 × 64 | CUDA 逐专家 | 1,153.92 | 620.51 | 5.31 | 3,015.52 |
| 4 × 64 | CUDA 每个 mask 生成一次索引 | 596.47 | 364.11 | 9.00 | 1,778.31 |
| 4 × 64 | CUDA 按层打包诊断 | 379.81 | 63.09 | 25.75 | 628.26 |
| 8 × 64 | Triton 逐专家 | 1,336.31 | 873.01 | 8.28 | 3,955.42 |
| 8 × 64 | CUDA 逐专家 | 1,040.82 | 595.94 | 10.91 | 2,986.97 |
| 8 × 64 | CUDA 按层打包诊断 | 431.31 | 85.75 | 46.27 | 691.85 |
| 4 × 512 | Triton 逐专家 | 2,033.56 | 903.71 | 3.42 | 4,697.82 |
| 4 × 512 | CUDA 逐专家 | 1,655.88 | 623.41 | 4.54 | 3,526.22 |
| 4 × 512 | CUDA 按层打包诊断 | 768.00 | 95.97 | 15.06 | 1,062.52 |

按层打包的 24 个 TPOT 配对全部下降，四组工作负载的配对降幅中位数为 87.5%–91.8%。复用 mask 索引在 4 × 64 中改善了 5/6 轮，配对降幅中位数为 50.8%。两个 CUDA dispatch 消融在各自覆盖的工作负载上与 CUDA 逐专家路径的全部 logits 逐位一致。各路径的 PyTorch 峰值已分配显存按输入形状落在 11,840.53–11,990.84 MiB。

## Decode 瓶颈在逐专家 dispatch

26 个 MoE 层在每次前向各循环 64 个 expert。循环中 `tgt[m]` 执行两次，`w_flat[m]` 执行一次。布尔索引根据 mask 生成动态长度的结果，原路径每次前向记录到 4,992 次 4 字节的设备到主机拷贝和 4,992 次 stream 同步。4 × 64 的 decode 中，这些拷贝的 GPU 执行时间每步约 5.14 ms，stream 同步的 CPU API 时长累计约 103.14 ms。整个 routed expert 区间平均 812.71 ms，占这份时间线单步墙钟的 95.3%，包含索引生成、Python 循环、启动和等待。

| 4 × 64 的单个 decode 前向，三步平均 | CUDA 逐专家 | 每个 mask 生成一次索引 | 按层打包诊断 |
|---|---:|---:|---:|
| Routed expert D2H 次数 | 4,992 | 1,664 | 26 |
| Routed expert stream 同步次数 | 4,992 | 1,664 | 52 |
| 整模 GPU kernel 启动数 | 21,783 | 11,381 | 2,251 |
| Routed expert 的 CPU NVTX 区间，含等待 | 812.71 ms | 341.43 ms | 29.52 ms |
| 其中 CUDA 全融合 expert kernel 累计 GPU 时间 | 13.89 ms | 13.76 ms | 13.96 ms |
| MLA decode 核心 kernel 累计 GPU 时间 | 0.80 ms | 0.71 ms | 0.71 ms |
| 整模 GPU kernel 活动时间并集 | 36.78 ms | 28.52 ms | 22.17 ms |

复用索引仍执行每一层的 64 次循环，每个 expert 只做一次动态索引。按层打包每层把 top-k expert ID 读回一次，统一打包 token，再按原先的 expert 顺序执行非空 expert。两者沿用同一套投影 kernel。时间线按 CUDA API correlation ID 归属 GPU kernel 与拷贝。CPU NVTX 包含异步提交和同步等待，独立计时用于评估端到端收益。

## 长输入的 prefill

4 × 512 按层打包后，prefill 的 GPU kernel 活动时间为 707.90 ms。按提交模块归属，routed expert kernel 累计 519.10 ms，其中 W4A16 MMA 为 482.67 ms。Dense 与共享 FFN 累计 106.42 ms，含投影的 MLA 模块累计 78.12 ms，其中 FlashAttention 核心为 6.28 ms。

## 下一步优化顺序

1. 把路由索引打包和 expert offset 生成放在 GPU 上，使用同一份索引做 gather、权重选择及 scatter。保留原累加顺序，用四组工作负载验证 logits、TPOT 和 routed expert 区间的 D2H 次数。
2. 有了 GPU 上的 expert 分段后，测多 expert 合并提交。Decode 重点降低小 M 的启动开销，长输入 prefill 重点测较大 M 的 W4A16 grouped GEMM。用相同路由和权重检查短输入与长输入的收益。

## 复现

```bash
out=profiles/deepseek_bottleneck_repro
bash benchmarks/run_deepseek_w4a16.sh bottleneck "$out"
bash benchmarks/run_deepseek_w4a16.sh bottleneck_nsys "$out"
python3 -m benchmarks.deepseek_bottleneck_trace "$out/bottleneck.sqlite"
```

[独立计时与数值核对原始数据](../profiles/deepseek_bottleneck_20261118/timing.json) · [Nsight Systems 时间线](../profiles/deepseek_bottleneck_20261118/bottleneck.nsys-rep) · [Nsight SQLite](../profiles/deepseek_bottleneck_20261118/bottleneck.sqlite)
