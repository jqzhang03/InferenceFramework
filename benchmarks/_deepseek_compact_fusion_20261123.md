# DeepSeek-V2-Lite compact decode：gate/up 融合 + fan-out 调优

在 [GPU prefill follow-up](_deepseek_sharegpt_gpu_prefill_20261123.md) 之后，decode 保持 CUDA Graph，继续处理 compact expert 的融合和 fan-out。

## 实现

### 融合 gate/up

新增 `w4a16_compact_mma_gate_up_kernel<BF16, BLOCK_M>` 和对应的 `mma_gate_up_tile`：

- 128 线程，4 warp：warp 0–1 算 gate，warp 2–3 算 up；
- 每个 CTA 只加载一次 activation tile，gate/up 各一份权重 tile 放在 shared；
- 每个 warp 只保留一组 accumulator，避免 64 线程版本“每线程两套 accum”带来的寄存器和 shared 压力；
- gate/up 一次 launch 完成，替代原来的两次 projection launch；
- 每个输出元素的 MMA K 顺序与 `mma_tile` 完全相同，因此保持位级一致。

**down 仍单独 launch**：down 依赖 `silu(gate)*up` 的完整 intermediate，一个 CTA 只拥有一个 32 列 intermediate tile；要在一个 kernel 内算完 down，需要跨 CTA 的 grid sync 或重算 gate/up，不划算。因此当前是 **gate/up 融合 + down 分离** 的两 kernel 结构，而不是把三个投影硬塞进一个 kernel。

Python 侧 `gemm_gate_up_out_compact` 写入 `gate_out/up_out`，仍由 `F.silu(gate_out)*up_out` 产生 hidden；没有把 SiLU 放进 kernel，避免 BF16 舍入顺序改变位级结果。

### fan-out 重调

融合后重新扫 fan-out（5 repeats，eager 单进程）：

| 负载 | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|
| b4_ctx64 | 65.59 | 73.02 | **58.96** | 73.29 | 74.56 |
| b8_ctx64 | 65.21 | 61.19 | 65.21 | **59.20** | 59.43 |
| b16_ctx64 | 88.46 | 87.34 | **77.63** | 80.11 | 81.05 |
| b32_ctx64 | 119.29 | 105.72 | 102.39 | 102.41 | **102.27** |

自动规则从 `8/64/128` 改为：

```text
T≤4     → 8
5≤T≤8   → 128
T>8     → 64
```

b32 在 64/128/256 基本平坦，取 64；b8 的 128 最明显；b16 的 64 最稳。

## CUDA Graph 结果

同一套 graph capture `[1,2,4,8,16,32]`，6 repeats，tokens 全部一致，`replays=3`、`fallbacks=0`：

| 负载 | baseline graph | **fused + tuned graph** | 变化 |
|---|---:|---:|---:|
| b1_ctx64 | 19.43 ms | **19.18 ms** | −1.3% |
| b4_ctx64 | 30.75 ms | **30.88 ms** | +0.4% |
| b8_ctx64 | 52.11 ms | **49.60 ms** | **−4.8%** |
| b16_ctx64 | 74.37 ms | **74.24 ms** | −0.2% |
| b32_ctx64 | 94.82 ms | **95.72 ms** | +0.9% |

结论要诚实：

- **b8 有约 4.8% 的稳定收益**，因为 5–8 行 expert 的 gate/up 融合 + 128 fan-out 最匹配；
- b1/b16 基本持平；b4 在噪声内；
- **b32 略慢 0.9%**，当前 fan-out 64 没有找到比 baseline 更好的组合；
- 融合的主要价值是减少 launch 和 activation 读取，不是所有 batch 都能转化为 wall 时间。

eager 单次对比噪声更大（b16 109 ms、b32 102 ms），不作为主结论；CUDA Graph 是 decode 的正式路径。

## 正确性

- `tests/test_moe_int4_compact.py`：**40 passed**，T=1…1024，包含 gate/up 融合后的所有 row range；
- 全量回归：**110 passed**；
- graph/eager tokens 在 b1/b4/b8/b16/b32 全部一致；
- graph replay 3 次、fallback 0。

## 产物

- [6-repeat graph JSON](../profiles/deepseek_compact_20261122/graph_fused_final/graph.json) · [eager JSON](../profiles/deepseek_compact_20261122/graph_fused_final/eager.json)
- [fan-out sweep JSON](../profiles/deepseek_compact_20261122/fanout_fused.json)
- 代码：[moe.py](../nanovllm/layers/moe.py) · [cuda_int4.cu](../nanovllm/layers/csrc/cuda_int4.cu)
- 测试：[test_moe_int4_compact.py](../tests/test_moe_int4_compact.py)
