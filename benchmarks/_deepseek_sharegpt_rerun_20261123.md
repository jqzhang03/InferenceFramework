# DeepSeek-V2-Lite ShareGPT 复测：端到端、nsys 瓶颈与 NCU 算子结论

2026-11-23，RTX 5060 Ti 16GB / WSL2。承接
[最终 ShareGPT 性能报告](_deepseek_sharegpt_final_perf_20261123.md)、
[dense tile 调优](_deepseek_dense_tiles_20261123.md) 和
[cache-hit prefill 修正](_deepseek_cache_hit_prefill_20261123.md) 之后，对当前
worktree 重新做端到端计时、Nsight Systems 抓取和 Nsight Compute 算子采集。

## 0. 结论（TL;DR）

- **端到端（预热后稳态，3 个 batch）**：wall **3.43 s**，output **298.8 tok/s**，
  TTFT **519.7 ms**，TPOT **93.6 ms**，E2E **3,421.6 ms**；三个 batch 的 wall 为
  3.427 / 3.442 / 3.374 s，稳定。
- **nsys**：capture 3,459.5 ms，GPU active 3,244.2 ms（93.8%）。prefill
  478.5 ms wall / 392.1 ms GPU（81.9%）；decode 31 步共 2,875.7 ms wall /
  2,774.5 ms GPU（96.5%），平均 **92.76 ms wall / 89.50 ms GPU 每步**。
- **瓶颈是 routed expert 的 W4A16 投影，不是 attention、host 打包或 kernel launch**：
  prefill GPU 的 70.1% 在 compact expert 的 gate/up + down（274.9/392.1 ms）；
  decode GPU 的 82.4% 在 expert 投影（73.8/89.5 ms/step），其中 GEMV
  gate/up + down 就有 59.7 ms/step。dense 投影（MLA + dense/shared MLP）另占
  prefill 17.5% / decode 10.6%。
- **prefill 的额外问题**：478.5 ms wall 里有 **86.4 ms GPU 空转**，来自 82 次
  小 D2H（合计 707 B）和 82 次 `cudaStreamSynchronize`（CPU 侧 300.9 ms），根因是
  `MLAAttention._assemble` 每层读一次输出行数 `.item()`；这不是 GPU 算力问题。
- **NCU 判定：算子仍然需要加速**。最明确的两处：
  1. **decode compact GEMV** 只有 30.4% / 37.2% DRAM 吞吐、1.78–2.37 waves，
     1.289 ms（gate/up）+ 0.570 ms（down）每层，属于 memory-latency/ILP 受限；
  2. **prefill gate/up MMA32** 已经到 66.8% DRAM，但一次 launch 读 **1.338 GB**，
     是 64 个 expert 唯一 gate/up 权重（184 MB）的 **7.3×**；这是 4 个 row chunk
     × channel fan-out 造成的重复流量，降流量比提算力更有效。
- **不需要优先加速**：dense 的 `w4a16_mma_kernel` 已到 71–80% tensor pipe；
  attention 核心 1.58 ms/step、LM head 约 1.3 ms/step、sampling/radix 约 0.75 ms/step，
  合计不到 decode 的 5%。

## 1. 测试口径

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti，16,311 MiB，sm_120 |
| 软件 | PyTorch 2.8.0+cu128，CUDA 12.8，flash-attn 2.8.3.post1，Triton 3.4 |
| 模型 | `~/huggingface/DeepSeek-V2-Lite`，INT4 group 64，streaming load |
| MoE | `moe_int4_backend="compact"`，`cta_fanout=0`（自动），gate/up + SiLU 融合开启 |
| 序列 | `max_model_len=1024`，`max_num_seqs=32`（nsys）；bench 由 compact graph 上限覆盖 b≤32 |
| KV | 220 blocks × 256 token = 56,320 token；BF16 `auto`；无抢占/swap |
| 执行 | decode CUDA Graph，31 replay / 0 fallback；bench 为 multi-step decode（8 engine steps / 31 forwards） |
| workload | ShareGPT trace，32 请求，总 prompt 1,769 token，512 token 命中 prefix cache（2 请求），prefill query 1,257 token，每请求 32 输出 token（总 1,024 completion，decode forward 992） |
| 预热 | 真实 prompts × 2 遍（第一遍建 prefix cache，第二遍编译命中态 prefill） |
| 计时 | 3 个相同 workload batch；batch 0 作为主结果 |

说明：

- `benchmarks/bench.py` 默认多步 decode（每个 engine step 最多 4 个 forward），
  所以 bench 报 8 个 decode steps / 31 forwards；`benchmarks/nsys_profile.py`
  固定 `multi_step_decode=False`，用于逐 step 归因。两者 TPOT 口径一致
  （bench 93.6 ms vs nsys 92.76 ms/step）。
- nsys 使用 `--cuda-graph-trace=node`，把 decode CUDA Graph 展开成节点，
  才能看到 graph 内部 kernel；node trace 有额外开销，因此**吞吐以 bench 为准，
  nsys 只用于阶段/kernel 占比**。
- NCU 用 `--cache-control all` 隔离每次 replay 的 cache，并 flush L2；单 kernel
  绝对时间不能和 nsys 的 in-model 时间直接相加，**NCU 只用于利用率、stall、
  DRAM 字节数**。

## 2. 端到端性能

### 2.1 主结果

| 指标 | 本轮（batch 0） | 上一轮 dense-tile 报告 | 更早的 final perf 报告 |
|---|---:|---:|---:|
| wall | **3.427 s** | — | 4.363 s |
| output throughput | **298.8 tok/s** | — | 234.6 tok/s |
| TTFT avg / p50 | **519.7 / 519.7 ms** | 496.8 ms | 915.1 / 915.1 ms |
| TPOT avg / p50 | **93.6 / 93.6 ms** | 100.6 ms | 111.2 / 111.2 ms |
| E2E avg / p50 | **3,421.6 / 3,421.6 ms** | 3,614 ms | 4,363.2 ms |
| prefill | 1,257 tok / 1 step（2,418 tok/s） | 0.497 / 0.472 / 0.477 s | 1,257 tok / 1 step |
| decode | 992 tok / 8 engine steps（342 tok/s） | — | 992 tok / 8 engine steps |
| decode graph | 31 replay，0 fallback | 31 replay，0 fallback | 31 replay，0 fallback |
| preemptions / swaps | 0 / 0 | 0 / 0 | 0 / 0 |
| prefix cache | 512 hit token，2/32 请求 | 同 | 同 |

throughput 用 completion token 1,024 / wall 计算；decode 行的 992 是 prefill
之后的 forward token（每请求首 token 在 prefill 产生）。

相对更早的 final perf 报告：TTFT **−43.2%**、TPOT **−15.8%**、wall **−21.4%**、
output throughput **+27.4%**。相对 dense-tile 报告：TPOT **−7.0%**、E2E **−5.3%**，
TTFT 略高 4.6%（496.8 → 519.7 ms，属于不同次运行/prefix-cache 状态的波动，见 2.2）。

### 2.2 三个 batch 的稳定性

| batch | wall | prefill step | decode | TTFT |
|---|---:|---:|---:|---:|
| 0 | 3.427 s | 0.520 s | 2.902 s | 519.7 ms |
| 1 | 3.442 s | 0.522 s | 2.917 s | 522.0 ms |
| 2 | 3.374 s | 0.439 s | 2.933 s | 439.0 ms |

batch 2 的 prefill 更快，是重复同一批 prompts 后 prefix cache 命中更多 block；
decode 三步基本不变。报告主表使用 batch 0，避免把 cache 增长带来的额外命中算进 TTFT。

SLO：`TTFT≤500 ms` 命中 0%（p50 519.7 ms，刚好超过），`TPOT≤10 ms` 命中 0%；
这是 16 GB 单卡 + INT4 + 15.7B 模型的硬件限制，不作为本轮优化目标。

## 3. Nsight Systems：阶段与 kernel

产物：[profile.nsys-rep](../profiles/deepseek_rerun_20261123/nsys/profile.nsys-rep)、
[analysis.json](../profiles/deepseek_rerun_20261123/nsys/analysis.json)、
[events.json](../profiles/deepseek_rerun_20261123/nsys/events.json)。

### 3.1 阶段表

| 阶段 | steps | wall | GPU active | GPU 占比 | kernel 数 | D2H |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 478.5 ms | 392.1 ms | 81.9% | 3,273 | 82 |
| mixed | 1 | 101.8 ms | 77.6 ms | 76.2% | 2,176 | 1 |
| decode | 31 | 2,875.7 ms | 2,774.5 ms | 96.5% | 63,031 | 31 |
| **capture** | 33 | **3,459.5 ms** | **3,244.2 ms** | **93.8%** | — | 33 |

decode 每步平均 92.76 ms wall / 89.50 ms GPU / **2,033 个 kernel**（node trace 展开）。
kernel 数很多，但 launch 已被 CUDA Graph 消掉；时间集中在少数 W4A16 投影上。

### 3.2 prefill top kernel（GPU 392.1 ms）

| kernel | 时间 | 次数 | 占 prefill GPU |
|---|---:|---:|---:|
| `w4a16_compact_mma_gate_up_silu_kernel`（MMA32 / 17+ 行 band） | 135.4 ms | 78 | 34.5% |
| `w4a16_compact_mma_kernel`（routed down） | 75.4 ms | 78 | 19.2% |
| `w4a16_mma_kernel`（dense/shared/MLA 投影） | 68.7 ms | 188 | 17.5% |
| `w4a16_compact_gate_up_kernel`（≤4 行 GEMV） | 41.8 ms | 26 | 10.7% |
| `w4a16_compact_down_kernel`（≤4 行 GEMV） | 22.3 ms | 26 | 5.7% |
| `Kernel2`（cuBLAS GEMM） | 7.5 ms | 55 | 1.9% |
| `_scatter_gather_elementwise_kernel` | 7.3 ms | 79 | 1.9% |
| `CatArrayBatchedCopy` | 6.1 ms | 108 | 1.6% |
| `vectorized_elementwise_kernel` | 6.0 ms | 715 | 1.5% |
| `flash_fwd_kernel` | 4.9 ms | 27 | 1.3% |

routed expert 四项合计 **274.9 ms（70.1%）**；dense 投影 68.7 ms（17.5%）。
prefill 的 `w4a16_compact_mma_gate_up_silu_kernel` 里 `<8>`/`<16>` 两个 band 在
prefill 基本是空的（NCU 单次 3.9/6.7 µs），真正干活的是 `<32>` band。

### 3.3 decode top kernel（31 步聚合，每步平均 89.50 ms GPU）

| kernel | 31 步合计 | 每步 | 占 decode GPU |
|---|---:|---:|---:|
| `w4a16_compact_gate_up_kernel`（GEMV） | 1,309.4 ms | 42.24 ms | 47.2% |
| `w4a16_compact_down_kernel`（GEMV） | 540.5 ms | 17.43 ms | 19.5% |
| `w4a16_mma_kernel`（dense/shared/MLA） | 292.9 ms | 9.45 ms | 10.6% |
| `w4a16_compact_mma_gate_up_silu_kernel`（5–16 行 band） | 289.1 ms | 9.33 ms | 10.4% |
| `w4a16_compact_mma_kernel`（5–16 行 band） | 148.6 ms | 4.79 ms | 5.4% |
| `Kernel2`（cuBLAS GEMM） | 50.9 ms | 1.64 ms | 1.8% |
| `mla_decode_kernel` | 49.0 ms | 1.58 ms | 1.8% |
| `radixSortKVInPlace`（sampling） | 23.3 ms | 0.75 ms | 0.8% |
| elementwise 合计 | 37.5 ms | 1.21 ms | 1.4% |

decode 的 5 个 W4A16 kernel 合计 **83.2 ms/step（93.0%）**；其中 GEMV 两条
（gate/up + down）**59.7 ms/step（66.7%）**，5–16 行 MMA band 两条
**14.1 ms/step（15.8%）**，dense 9.45 ms（10.6%）。attention 核心只有
1.58 ms/step（1.8%）。

## 4. 瓶颈归因（从数据到结论）

### 4.1 TTFT：prefill GPU + 86 ms host-sync 空转

TTFT 519.7 ms 由 prefill step 的 0.520 s 直接决定（queue wait p50 0.4 ms）。
prefill wall 478.5 ms 中：

- GPU 392.1 ms（81.9%），其中 routed expert 274.9 ms、dense 68.7 ms、其余约 48.5 ms；
- GPU 空转 86.4 ms（18.1%），对应 82 次小 D2H（合计 707 B）、82 次
  `cudaStreamSynchronize`（CPU 侧 300.9 ms）。`_assemble` 每层构造 cache-shaped
  K/V 时读一次输出行数（`.item()`），每层 1 次（26 次）+ 其它同步点，共 82 次。
- host launch 本身只有 39.6 ms CPU（3,050 次 `cudaLaunchKernel`），不是 launch-bound；
  也没有 graph fallback 或抢占。

所以 TTFT 中约 **75% 是 prefill GPU，17% 是 MLA 组装的行数 host 同步**，
其余约 8% 是调度/launch。host 同步这部分在 [cache-hit prefill 报告](_deepseek_cache_hit_prefill_20261123.md) 中已从
4,213 次 D2H 降到 82 次，但每层还剩 1 次；要把剩余 86 ms 拿掉，需要把
`_assemble` 的输出行数通过 `Context` 传给 kernel，而不是 `.item()`。

### 4.2 TPOT：decode GPU 的 routed expert 投影

decode 92.76 ms wall 中 GPU 89.50 ms（96.5%），只剩 3.26 ms 空转；每步
2,033 个 kernel 已被 graph 压住。时间结构：

- **GEMV gate/up + down：59.7 ms/step（66.7%）**。每层每个 expert 只处理
  1–4 行；NCU 单次 1.289 ms（gate/up）+ 0.570 ms（down），单层 1.859 ms；
  nsys 全模型 59.7 ms/step（各层 expert 分布不同，单层外推只作量级参考）。
- **5–16 行 MMA band：14.1 ms/step（15.8%）**。T=32 时每个 expert 平均 3 行，
  但仍有一批 expert 落在 5–8 / 9–16 行；两条 MMA kernel 每层各 0.243/0.105 ms
  和 0.125/0.111 ms，tensor pipe 只有 7.6–19.8%。
- dense 投影 9.45 ms（10.6%）、attention 1.58 ms（1.8%）、LM head 1.3 ms、
  sampling/radix 0.75 ms。

按 expert 权重粗算：T=32、top_k=6 → 192 个 route，通常覆盖 55–64 个 expert；
每层 gate/up 约 0.16 GB、down 约 0.08 GB DRAM 读，26 层约 6.2 GB/step。
59.7 ms 对应约 **104 GB/s 聚合读带宽**（按 448 GB/s 规格约 23%），
远低于带宽上限；结合 NCU 的 30–37% 单 kernel DRAM 和 1.78–2.37 waves，
可以确定是 **memory-latency / 指令级并行受限，而不是 DRAM 带宽打满**。

### 4.3 为什么不是这些

- **attention**：prefill `flash_fwd_kernel` 4.9 ms（1.3%），decode
  `mla_decode_kernel` 1.58 ms/step（1.8%）；MLA 的 q/kv/o 投影算在 dense 的
  68.7 ms 里，但 attention 核心不是瓶颈。
- **LM head / sampling**：prefill 约 7.5 ms（1.9%，`Kernel2`/cuBLAS），decode
  约 1.3 + 0.75 ms/step。
- **D2H / host 打包**：prefill 82 次小 D2H（见 4.1），decode 每步只有 1 次
  sampler D2H；没有 per-layer route D2H。
- **kernel launch**：decode 是 CUDA Graph（31 replay / 0 fallback），prefill
  launch CPU 39.6 ms；graph 已把 launch 税消掉。
- **KV/抢占**：0 preemption、0 swap、容量 56,320 token，本轮 workload 不触发。

## 5. Nsight Compute：算子速度与加速判定

### 5.1 采集方法

- NCU 2026.3.1（`/usr/local/NVIDIA-Nsight-Compute-2026.3/ncu`），sm_120。
- `--set detailed --cache-control all`，kernel replay；每个 phase 只抓前 20 个
  匹配的 W4A16 launch（layer 0 dense MLP + layer 1 MLA + layer 1 全部 MoE band）。
- harness：[benchmarks/dsv2_operator_ncu.py](../benchmarks/dsv2_operator_ncu.py)，
  强制 eager（graph 会隐藏 kernel），用 NVTX start/end range 标出
  `ncu_prefill` / `ncu_decode`；prefill 形状为 1,250 query token（另有 mixed 7 token），
  decode 为 batch 32。
- 只使用利用率/字节数/occupancy/stall；绝对时间不与 nsys 混算。

### 5.2 prefill（单次 launch，NCU）

| kernel（template） | 时间 | grid × block | DRAM | SM | tensor | occupancy | waves | DRAM 读 | 主要 stall |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `compact_mma_gate_up_silu<1,32>` | 4.656 ms | (64,4,44) × 128 | **66.8%** | 46.3% | 46.3% | 41.5% | 62.6 | **1.338 GB** | long scoreboard |
| `compact_mma<1,32>`（down） | 3.308 ms | (64,4,64) × 64 | 18.7% | 42.4% | 42.4% | 39.8% | 45.5 | 0.161 GB | long scoreboard / wait / math pipe |
| `compact_gate_up<1,4>`（GEMV） | 1.616 ms | (64,1,8) × 128 | 29.8% | 33.2% | 0% | 46.1% | **2.37** | 0.199 GB | long scoreboard |
| `compact_down<1,4>`（GEMV） | 0.674 ms | (64,1,8) × 128 | 40.3% | 50.5% | 0% | 63.2% | 1.78 | 0.106 GB | long scoreboard |
| `w4a16_mma<1,128,256>`（q_b 样例） | 1.606 ms | (3420) × 256 | 7.5% | **80.1%** | **80.1%** | 65.6% | 23.8 | 0.026 GB | math pipe throttle / long scoreboard |

### 5.3 decode（单次 launch，NCU）

| kernel（template） | 时间 | grid × block | DRAM | SM | tensor | occupancy | waves | DRAM 读 | 主要 stall |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `compact_gate_up<1,4>`（GEMV） | 1.289 ms | (64,1,8) × 128 | 30.4% | 33.7% | 0% | 49.6% | 2.37 | 0.160 GB | long scoreboard |
| `compact_down<1,4>`（GEMV） | 0.570 ms | (64,1,8) × 128 | 37.2% | 48.6% | 0% | 56.6% | 1.78 | 0.080 GB | long scoreboard |
| `compact_mma_gate_up_silu<1,8>` | 0.243 ms | (64,1,44) × 128 | 34.6% | 18.7% | 7.6% | 62.4% | 9.8 | 0.028 GB | long scoreboard |
| `compact_mma_gate_up_silu<1,16>` | 0.105 ms | (64,1,44) × 128 | 38.5% | 25.6% | 19.8% | 55.1% | 9.8 | 0.016 GB | long scoreboard |
| `compact_mma<1,8>`（down） | 0.125 ms | (64,1,64) × 64 | 29.3% | 21.9% | 7.7% | 47.4% | 7.1 | 0.014 GB | long scoreboard |
| `compact_mma<1,16>`（down） | 0.111 ms | (64,1,64) × 64 | 21.0% | 15.2% | 9.1% | 23.2% | 11.4 | 0.009 GB | long scoreboard |
| `w4a16_mma<1,16,128>`（q_b 样例） | 0.116 ms | (684) × 128 | 27.0% | 55.3% | 30.2% | 82.5% | 1.58 | 0.012 GB | long scoreboard |

（完整 20+20 条记录见 [summary.json](../profiles/deepseek_rerun_20261123/ncu/summary.json)；
原始 CSV：[prefill](../profiles/deepseek_rerun_20261123/ncu/prefill/report.csv) /
[decode](../profiles/deepseek_rerun_20261123/ncu/decode/report.csv)。）

### 5.4 逐算子判断

**prefill gate/up MMA32：需要降流量，优先级最高（prefill 侧）。**
4.656 ms 对应 66.8% DRAM、287 GB/s 读；一次 launch 读 1.338 GB，而 64 个
expert 的唯一 gate/up 权重只有约 184 MB（1.338/0.184 ≈ **7.3×**）。
32 行 M tile 下的 4 个 row chunk（`grid.y=4`）和 44 个 channel CTA 会重复读
权重/激活。算术强度约 87 GFLOP / 1.338 GB ≈ **65 FLOP/B**，低于
~90 TFLOPS / 448 GB/s 屋顶线的拐点（约 200 FLOP/B），所以这是
**memory-traffic bound**。把 chunk 改成“一个 CTA 顺序走完所有 chunk、权重 tile
复用”或减少 channel CTA 的重复激活读，理论上有 1.5–2× 空间；26 层合计
135.4 ms，可争取 −30~−50 ms。

**prefill down MMA32：需要 dequant/ALU 加速。**
3.308 ms，DRAM 仅 18.7%，SM/tensor 42.4%，算术强度约 270 FLOP/B，已经是
compute-bound；瓶颈是每 K 组的 int4→bf16 反量化和 MMA 之间的依赖。
把 8 次标量 `cvt` 改成 4 次 `cvt.rn.bf16x2.f32`、K 循环双缓冲，目标 1.3–1.6×；
26 层 75.4 ms，可争取 −15~−25 ms。

**prefill 小 expert GEMV 尾巴：需要并行度/调度。**
`compact_gate_up<1,4>` 单次 1.616 ms、只有 2.37 waves、DRAM 29.8%、SM 33.2%；
它只服务 ≤4 行的少数 expert，却占 prefill GPU 10.7%（41.8 ms），down 另占
5.7%（22.3 ms）。这是典型的“expert 太小、CTA 太少、权重照读”问题；
合并小 expert 或按实际 expert 数缩 grid 可以拿回一半左右（−20~−30 ms）。

**dense `w4a16_mma_kernel`：暂不需要优先加速。**
prefill 样例 tensor pipe 80.1%、occupancy 65.6%、DRAM 7.5%；decode 样例
tensor 30.2%、occupancy 82.5%。它已经是“算力/流水线受限”而不是“流量受限”，
继续调 tile 的收益远小于 expert 路径；且它承担 27×7 个 MLA/dense 投影，
是模型本身的 FLOPs，不能靠算子融合消掉。

**decode compact GEMV：需要加速，优先级最高（decode 侧）。**
gate/up 1.289 ms + down 0.570 ms，单层 1.859 ms；nsys 全模型 59.7 ms/step
（各层 expert 分布不同，单层外推只作量级参考）。
DRAM 30.4% / 37.2%，waves 2.37 / 1.78，top stall 都是 long scoreboard；
算术强度约 14 FLOP/B，远低于屋顶线拐点，但带宽只用了三成——
说明是 **load 粒度/ILP/occupancy 受限**，不是 DRAM 打满。
把权重 load 从 4/8-byte 提升到 16-byte、增加每 CTA 的 K 并行和 CTA 数，
目标 1.5–2×；若提升 1.5×，可省约 20 ms，TPOT 约 73 ms（−22%）。

**decode 5–16 行 MMA band：可以合并/收敛。**
`<8>` 0.243 ms + `<16>` 0.105 ms（gate/up），`<8>` 0.125 ms + `<16>` 0.111 ms
（down），每层 0.584 ms，26 层 15.2 ms/step；tensor 只有 7.6–19.8%，
DRAM 21–38%。这些 band 是真实存在的 5–16 行 expert，但用三个独立模板
（8/16/32）分别 launch，效率低。把 band 合并成一个按 count 分支的 kernel，
或对 `max_count≤16` 只发 8/16 两个模板，可争取 −5~−8 ms/step。

**attention / LM head / sampling：不优先。**
decode attention 1.58 ms/step、LM head 1.3 ms、sampling 0.75 ms；
prefill attention 4.9 ms。除非 expert 路径已经压到极限，否则不值得为它们改 kernel。

### 5.5 总体判定

**算子还需要加速，且加速重点不是“更大的 MMA tile”，而是：**

1. **decode GEMV 的访存并行度**（收益最大，直接降 TPOT）；
2. **prefill gate/up 的 chunk/权重复用**（降 7.3× 重复 DRAM 流量）；
3. **prefill down 的 int4→bf16 反量化指令**；
4. **prefill 小 expert GEMV 的 grid 收缩**；
5. **decode 5–16 行 band 的模板合并**。

dense 投影、attention、LM head 已经接近各自的局部上限，不是本轮瓶颈。
按上述 1–3 的保守目标（不改变模型/精度），预计 TTFT 可降 15–25%，
TPOT 可降 20–30%；这些是 kernel 级判断，仍需端到端 A/B 验证，不能把
NCU 单 kernel 时间直接外推成 serving 收益。

## 6. 正确性与回归

- `pytest tests/test_mla.py tests/test_cuda_int4.py -q`：**27 passed**（28.0 s），
  日志 [pytest.log](../profiles/deepseek_rerun_20261123/pytest.log)。
- 本轮 bench：3 个 batch 的 decode token 数一致（992；completion 1,024），
  prefill token 1,257，decode 31 forwards，graph 31 replay / 0 fallback，
  0 preemption / 0 swap。
- nsys 与 bench 的 step 结构一致（prefill 1 + mixed 1 + decode 31）；
  NCU harness 的 prefill/mixed/decode 形状为 1,250+7 / 31 / 32 rows，符合
  prefix-cache 命中后的真实调度。

## 7. 产物

保留的最终产物（共 22 MB）：

- [bench.json](../profiles/deepseek_rerun_20261123/bench/bench.json) ·
  [bench.log](../profiles/deepseek_rerun_20261123/bench/bench.log)
- [profile.nsys-rep](../profiles/deepseek_rerun_20261123/nsys/profile.nsys-rep) ·
  [analysis.json](../profiles/deepseek_rerun_20261123/nsys/analysis.json) ·
  [events.json](../profiles/deepseek_rerun_20261123/nsys/events.json)
- NCU：
  [prefill rep](../profiles/deepseek_rerun_20261123/ncu/prefill/profile.ncu-repz) ·
  [prefill CSV](../profiles/deepseek_rerun_20261123/ncu/prefill/report.csv) ·
  [decode rep](../profiles/deepseek_rerun_20261123/ncu/decode/profile.ncu-repz) ·
  [decode CSV](../profiles/deepseek_rerun_20261123/ncu/decode/report.csv) ·
  [summary.json](../profiles/deepseek_rerun_20261123/ncu/summary.json)
- harness：[dsv2_operator_ncu.py](../benchmarks/dsv2_operator_ncu.py)
- 回归：[pytest.log](../profiles/deepseek_rerun_20261123/pytest.log)

## 8. 复现

```bash
cd /mnt/d/Project/nano-vllm
export PATH="/usr/local/cuda-12.8/bin:$HOME/miniconda3/envs/nano-vllm/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export PYTHONPATH=/mnt/d/Project/nano-vllm

# 端到端（3 个 batch，预热 2 遍真实 prompts）
python benchmarks/bench.py --model ~/huggingface/DeepSeek-V2-Lite \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --quantization int4 --int4-group-size 64 --streaming-load \
  --moe-int4-backend compact --max-model-len 1024 \
  --gpu-memory-utilization .85 --warmup-seqs 32 --warmup-passes 2 \
  --repeat-batches 3 \
  --output profiles/deepseek_rerun_20261123/bench/bench.json

# nsys（graph node trace 才能在 decode 里逐 kernel 归因）
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --force-overwrite=true --stats=false \
  -o profiles/deepseek_rerun_20261123/nsys/profile \
  python -m benchmarks.nsys_profile \
  --model ~/huggingface/DeepSeek-V2-Lite \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --quantization int4 --int4-group-size 64 --streaming-load \
  --moe-int4-backend compact --moe-int4-cta-fanout 0 \
  --max-num-seqs 32 --max-model-len 1024 --gpu-memory-utilization .85 \
  --warmup-seqs 32 --warmup-prefix-hits --cuda-graph \
  --gpu-event-json profiles/deepseek_rerun_20261123/nsys/events.json
nsys export --type sqlite --force-overwrite=true \
  --output profiles/deepseek_rerun_20261123/nsys/profile.sqlite \
  profiles/deepseek_rerun_20261123/nsys/profile.nsys-rep
python -m benchmarks.sharegpt_deepseek_nsys \
  profiles/deepseek_rerun_20261123/nsys/profile.sqlite \
  --events profiles/deepseek_rerun_20261123/nsys/events.json \
  --output profiles/deepseek_rerun_20261123/nsys/analysis.json

# NCU（每个 phase 抓前 20 个 W4A16 launch）
NCU=/usr/local/NVIDIA-Nsight-Compute-2026.3/ncu
"$NCU" --nvtx --nvtx-include ncu_prefill --kernel-name-base function \
  -k 'regex:.*w4a16_(compact_mma_gate_up_silu_kernel|compact_mma_kernel|compact_gate_up_kernel|compact_down_kernel|mma_kernel).*' \
  -c 20 --set detailed --cache-control all --force-overwrite \
  --export profiles/deepseek_rerun_20261123/ncu/prefill/profile \
  python -m benchmarks.dsv2_operator_ncu --phase prefill \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --output profiles/deepseek_rerun_20261123/ncu/prefill/harness.json
# decode 把 --phase 换成 decode、--nvtx-include 换成 ncu_decode；CSV 由
#   ncu --import <rep> --csv --page raw > report.csv
# 导出。
```

## 9. 清理

本轮测试后删除了可重建的中间产物和缓存，保留上面的 `.nsys-rep` / `.ncu-repz` /
CSV / JSON：

- `nsys export` 的 sqlite：本轮 63.4 MB，已由 `analysis.json` 取代；
  另外删除 23 份与同名 `.nsys-rep` 并存的旧 sqlite（约 0.91 GB，均可重新导出）。
  旧报告 `_deepseek_bottleneck_20261118.md` 直接引用的
  `profiles/deepseek_bottleneck_20261118/bottleneck.sqlite` **已保留**。
- WSL `~/.nv/ComputeCache` 1.2 GB、`~/pip_tmp` 274 MB、
  `profiles/.torch_extensions_w4a16` 4.1 MB。
- 本轮临时脚本、失败/冒烟 NCU 报告、空 import 日志、`__pycache__`、`.pytest_cache`。
- `profiles/` 从 1.8 GB 降到约 0.9 GB；磁盘剩余 777 GB。
- 保留 `~/.cache/torch_extensions`（69 MB，重跑时免重新编译 INT4 扩展）和
  `~/.triton`（11 MB，Triton JIT cache）。如果后续不再跑，可以一并删除，
  代价是下一次启动重新编译。
