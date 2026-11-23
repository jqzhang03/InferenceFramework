# DeepSeek-V2-Lite compact：GPU prefill 分段/MMA32 follow-up

延续 [ShareGPT baseline 报告](_deepseek_sharegpt_nsys_20261123.md)。baseline 的 prefill/mixed 走 `_forward_compact_host`：每层一次路由 `.tolist()`、逐 expert 调用参考 MMA，产生 **26 次 routed D2H/层** 和 **5,180 次 per-expert kernel**。

本次把 prefill 也切到 compact GPU 分段：

- ≤4 行/expert：GEMV；
- 5–8 / 9–16 行：MMA8 / MMA16；
- ≥17 行：MMA32，按 32 行 chunk，CTA 用 `start += gridDim.y * BLOCK_M` 走剩余 chunk；
- 只有 **>4096 行** 的 chunk 才回退 host；
- MMA32 grid 的 chunk 维上限为 4，避免“每 chunk 一个 CTA”导致大量空 CTA；
- GEMV 仍用 8 CTA/expert 处理少量 ≤4 行 expert。

## 正确性

- `tests/test_moe_int4_compact.py`：**40 passed**，T=1/2/3/4/5/8/9/16/17/32/64/256/1024，全部与 `_forward_loop` 位级一致。
- DeepSeek 完整模型 b4_ctx512（prefill **T=2048**）的 `cuda_gpu_compact` 与 `cuda_loop` 位级一致（`deepseek_bottleneck` 严格 parity gate 通过）。
- 26 层 MoE 的 GPU prefill 与 decode 共用同一 `order/counts/offsets`，gather、权重选择、scatter 不再跨 host。

## ShareGPT 前后对比

同一 32 条 ShareGPT burst（1,769 prompt token，32 token/请求），eager compact，`max_num_seqs=32`：

| 阶段 | 指标 | baseline（host prefill） | **GPU prefill** | 变化 |
|---|---|---:|---:|---:|
| prefill | wall | 893.3 ms | **889.4 ms** | −0.4% |
| prefill | GPU active | 734.5 ms | 804.8 ms | +9.6% |
| prefill | routed D2H | 27（26 层 + sampler） | **1（sampler）** | −26 |
| prefill | kernel 数 | 12,851 | **2,269** | −82% |
| prefill | `w4a16_mma_kernel`（逐 expert） | 675.9 ms / 5,180 次 | **0** | 消除 |
| mixed | wall | 1,799.3 ms | **1,643.1 ms** | −8.7% |
| mixed | GPU active | 139.1 ms | 95.5 ms | −31% |
| mixed | routed D2H | 27 | **1（sampler）** | −26 |
| mixed | kernel 数 | 10,172 | **2,306** | −77% |
| decode | wall / GPU | 10,061.5 / 3,014.6 ms | 10,240.7 / 3,139.5 ms | 尾编译噪声，稳态不变 |
| decode | routed D2H | 0 | 0 | 不变 |

GPU prefill 的顶层 kernel：

| kernel | 时间 | 次数 | 说明 |
|---|---:|---:|---|
| `w4a16_compact_mma_kernel<32>` | 489.5 ms | 234 | 17+ 行 expert，MMA32 chunk |
| `w4a16_mma_kernel` | 172.0 ms | 188 | dense/shared，不是逐 expert |
| `w4a16_compact_gate_up_kernel<4>` | 51.5 ms | 26 | 少量 ≤4 行 expert |
| `vectorized_elementwise` | 20.2 ms | 497 | router/SiLU/scatter |
| `w4a16_compact_down_kernel<4>` | 19.9 ms | 26 | ≤4 行 down |

## mixed 第一个 MLA 的 1.52 s 是什么

`ModelRunner.step[mixed]` 里 `DeepSeekV2.Layer[0].MLA [mixed]` 为 **1,522 ms**，而 Layer[1..26] 的 MLA 只有 0.9–3.2 ms。这个差异不是 layer 0 的 MLA 计算更重，而是**首次按 mixed/32-row shape 触发的 lazy JIT 编译**：

- 该 NVTX 范围内 GPU kernel 总时长只有约 0.2 ms；`cuModuleLoadData` 仅 1.642 ms，其余 1.5 s 是 CPU 侧编译，没有对应的大 GPU kernel。
- 同范围里出现 `mla_decode_kernel` 以及 `triton_per_fused__to_copy_add_*` / `triton_poi_fused__to_copy_*`，这些是 Triton / `torch.compile` 的变体。
- `nsys_profile` 的 warmup 只有 8 条请求；正式 mixed step 是 32 行。首次遇到新 batch/shape 时，layer 0 触发编译；Layer[1..26] 复用同一份编译结果。
- graph 模式下 `capture_cudagraph` 在 capture 前已经用 batch 32 走过这些 shape，所以 mixed wall 约 290 ms；eager 模式没有这一步，才在 mixed 里付 1.5 s。

处理方式是 **在 capture 前用最终 batch size 做 warmup**（例如 `--warmup-seqs 32`，并覆盖 mixed 的 prefill+decode 形状）；这段应算 startup/首次编译，不应算 MLA 稳态延迟。

## 结论：结构目标达成，wall 时间中性

**已完成用户要求的两个消除：**

1. **26 次 routed D2H/层 → 0**；prefill/mixed 只剩 sampler 的 1 次 D2H。
2. **5,180 次逐 expert `w4a16_mma_kernel` → 0**；prefill 现在 2,269 个 kernel（234 个 compact MMA + 188 个 dense/shared reference + 少量 elementwise），不再有 per-expert Python/launch 循环。

**但 prefill wall 没有明显变快（893→889 ms），GPU 时间反而 +9.6%。** 原因是：

- compact MMA32（489 ms）与 baseline 的逐 expert MMA 部分（约 504 ms）基本持平，GPU 计算量没有本质下降；
- 少量 ≤4 行 expert 的 GEMV tail 约 71 ms；
- 省掉的是 host 同步、Python 循环和 launch 次数，但 baseline 的 prefill 是 GPU-bound，CPU 开销原本被 GPU 掩盖。

因此这次改动的价值主要在**结构**而不是墙钟：

- 去掉了每层 D2H 同步，prefill 可以与后续传输/调度 overlap；
- launch 数从万级降到千级，CPU 侧更稳定；
- 为后续 CUDA Graph/固定容量 prefill、以及融合 gate/up/down 打基础。

## 下一步

1. **融合 compact gate/up**，复用一次 x 读取，目标是把 489 ms 的 MMA32 压到 baseline 以下；
2. 评估把 MMA32 的 chunk cap 从 4 调到 2/8 的 trade-off（当前 4 比 8 的 502 ms 好 13 ms）；
3. 仍要消除尾步 6.2 s 的 late compile：capture 前用最终 batch/采样形状 warmup；
4. 如果 prefill 继续是主要瓶颈，考虑更大的 tile / 双缓冲权重复用，而不是单纯增加 CTA。

## 产物

- [最终 GPU-prefill Nsight 报告](../profiles/sharegpt_deepseek_20261123/eager_gpu_prefill3/profile.nsys-rep)
- [最终分析 JSON](../profiles/sharegpt_deepseek_20261123/eager_gpu_prefill3/analysis.json)
- [baseline Nsight 报告](../profiles/sharegpt_deepseek_20261123/eager/profile.nsys-rep) · [baseline 分析 JSON](../profiles/sharegpt_deepseek_20261123/eager/analysis.json)
- 代码：[moe.py](../nanovllm/layers/moe.py) · [cuda_int4.cu](../nanovllm/layers/csrc/cuda_int4.cu)
- 测试：[test_moe_int4_compact.py](../tests/test_moe_int4_compact.py)
