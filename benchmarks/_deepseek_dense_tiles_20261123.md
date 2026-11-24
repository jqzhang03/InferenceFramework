# DeepSeek-V2-Lite：dense/shared 的 W4A16 MMA tile 调优

承接 [最终 ShareGPT 性能报告](_deepseek_sharegpt_final_perf_20261123.md) §5 的第 2 条：
**dense/shared 的 175.9 ms 仍是参考 `w4a16_mma_kernel`，需要单独的 tile/fan-out 调优**。

先澄清归属：那 188 个 launch 并不只是 MLP，而是**所有非 routed 的 INT4 线性层**——
27 层 × 7 个线性（MLA 的 q_a/q_b/kv_a/kv_b/o + dense/shared MLP 的 gate_up/down）= 189 个，
`moe_int4_backend="compact"` 只接管 routed experts。所以这一项调的是整条 dense 投影路径。

## 1. 结论

| 指标 | 调整前 | 调整后 | 变化 |
|---|---:|---:|---:|
| kernel（layered，M=1257，189 launches） | 108.9 ms | **65.4 ms** | **−39.9%** |
| kernel（layered，M=1769） | 147.3 ms | **86.9 ms** | **−41.0%** |
| kernel（layered，M=32，decode） | 20.1 ms | **8.5 ms** | **−57.6%** |
| nsys prefill `w4a16_mma_kernel`（M=1257） | 120.5 ms | **76.4 ms** | **−36.6%** |
| nsys prefill wall / GPU | 634.6 / 463.9 ms | **510.3 / 416.5 ms** | −19.6% / −10.2% |
| ShareGPT TTFT | 582.6 ms | **496.8 ms** | **−14.7%** |
| ShareGPT TPOT | 110.7 ms | **100.6 ms** | **−9.1%** |
| graph decode b8 / b16 / b32 TPOT | 47.54 / 72.60 / 90.08 ms | **43.38 / 66.78 / 83.35 ms** | −8.8% / −8.0% / −7.5% |

b1/b4 的 decode 走 1–4 行 GEMV（本次未改），TPOT 在 ±3% 噪声内。

## 2. 诊断

旧 kernel（`mma_tile<BLOCK_M=32>`）：64 线程 = 2 warp，CTA 覆盖 32 个输出通道 × 32 行；
**每个 CTA 都把自己那份权重 tile 重新反量化一次**。prefill M=1257 时同一份权重被 40 个
M-tile 重复反量化，M=1769 时 56 次——反量化（每 4 字节 int4 → 8 个 bf16，含 8 次
int→float、8 次乘 scale、8 次 cvt、一次 16B shared store）成为主成本，kernel 只有
~20 TFLOP/s。

为排除 L2 复用带来的乐观偏差，微基准用 **layered** 模式：27 层 × 7 个真实 shape、每层不同
权重（约 324 MB，超过 32 MB L2），一次 pass 就是一次真实 prefill 的 dense 投影。

## 3. 改动

`mma_tile` 增加 `THREADS` 模板参数：`col_groups = block_n/16 = 2`，
`row_groups = warps/col_groups`，每个 warp 负责 `BLOCK_M/row_groups` 行、16 个通道；
权重/激活 tile 的加载循环按 `THREADS` 均分，输出 tile 仍是 32 通道 × BLOCK_M 行。
每个输出元素的 MMA K 累加顺序不变，因此**与旧 kernel 逐位相同**（微基准对每个 shape/M
都做了 `torch.equal` 校验）。

按 m 的调度（阈值来自 layered 实测）：

```text
m ≤ 4          GEMV（不变）
5 ≤ m ≤ 64     BLOCK_M=16,  128 threads (4 warps)
65 ≤ m ≤ 160   BLOCK_M=32,  128 threads (4 warps)
m > 160        BLOCK_M=128, 256 threads (8 warps)
```

理由（layered，每层 7 个 linears 合计 ms）：M=32 时 16×4warp 0.332 ms 优于 128×8warp
0.341、32×4warp 0.402；M=64 时 0.365 vs 0.420/0.429；M=96–160 时 32×4warp 最好
（0.445/0.557）；M≥192 后 128×8warp 反超（0.633 vs 0.651，M=1257 时 2.424 vs 2.780）。
M=1257 下 7 个候选 tile 的完整对比：

| tile (BLOCK_M × threads) | M=1257 | M=1769 |
|---|---:|---:|
| 32 × 64（旧） | 107.8 ms | 146.9 ms |
| 64 × 128 | 79.7 | 106.4 |
| 128 × 128 | 100.5 | 131.2 |
| 64 × 64 | 138.9 | 184.2 |
| **128 × 256（选中）** | **66.8** | **87.1** |
| 32 × 128 | 76.5 | 104.4 |
| 256 × 256 | 78.2 | 103.1 |

## 4. 端到端验证

同一台机器、同一 ShareGPT trace（32 请求 / 1,257 query token，2 遍预热，冷 Triton cache）：

| 指标 | 调整前 | 调整后 |
|---|---:|---:|
| TTFT | 582.6 ms | **496.8 ms**（−14.7%） |
| TPOT | 110.7 ms | **100.6 ms**（−9.1%） |
| E2E | 4,132 ms | **3,614 ms**（−12.5%） |
| prefill step（3 个 batch） | 0.583 / 0.515 / 0.517 s | **0.497 / 0.472 / 0.477 s** |

nsys（`--warmup-prefix-hits`，cache 命中态 prefill）：

| prefill 阶段 | 调整前 | 调整后 |
|---|---:|---:|
| wall | 634.6 ms | 510.3 ms |
| GPU active | 463.9 ms | 416.5 ms |
| `w4a16_mma_kernel` | 120.47 ms × 188 | **76.39 ms × 188** |
| `w4a16_compact_mma_gate_up_silu_kernel` | 142.17 ms × 78 | 141.20 ms × 78 |
| `w4a16_compact_mma_kernel`（routed down） | 80.76 ms × 78 | 78.84 ms × 78 |
| decode 阶段 GPU | 3,345.0 ms | 3,016.8 ms |

CUDA Graph decode（6 repeats 中位数，graph replay 3 次、fallback 0，输出 token 与调整前逐
token 相同）：b8 47.54 → 43.38 ms，b16 72.60 → 66.78 ms，b32 90.08 → 83.35 ms。decode 每个
step 的 dense 投影从 ~20 ms 降到 ~8.5 ms（b32 实测 −6.7 ms，其余被 attention/MoE 等摊薄）。

## 5. 正确性

- `mma_tile` 的所有 tile 变体对每个 shape、每个 M 与旧 kernel **逐位相同**（微基准内置
  `torch.equal` 断言；K 顺序未变）。
- `tests/test_cuda_int4.py`、`test_moe_int4_compact.py`、`test_mla.py`、`test_moe.py`、
  `test_models.py`、`test_linear_backend_default.py`：**120 passed**（含 FP16/BF16、
  graph replay、m=7/13/33 等 tail 形状）。
- CUDA Graph capture 安全：新 kernel 无 host 侧同步，graph replay 3 次、fallback 0。

## 6. 复现

```bash
# 逐 shape / 逐 tile 扫描 + layered 27 层 pass（不需要加载模型）
python -m benchmarks.int4_dense_tuning --root profiles/int4_dense_20261123/prod \
  --variants=-1,0 --tokens 32,1257 --repeats 3 --layered 32,1257,1769

# 端到端
python benchmarks/bench.py --model ~/huggingface/DeepSeek-V2-Lite \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --quantization int4 --int4-group-size 64 --streaming-load \
  --moe-int4-backend compact --max-model-len 1024 --gpu-memory-utilization .85 \
  --warmup-seqs 32 --warmup-passes 2 --repeat-batches 3 \
  --output profiles/int4_dense_20261123/bench_prefill.json
python -m benchmarks.compact_graph --root profiles/int4_dense_20261123/graph_tuned \
  --mode graph --workloads b1_ctx64,b4_ctx64,b8_ctx64,b16_ctx64,b32_ctx64 \
  --repeats 6 --max-num-seqs 32
```

诊断入口 `gemm_out_variant(x, packed, scales, out, variant)` 保留在扩展里（variant 0 是旧
32 行 × 2 warp tile），便于后续继续比较 tile；生产路径由 `launch()` 按 m 选择。

## 7. 产物与清理

- kernel 扫描：[sweep.json](../profiles/int4_dense_20261123/sweep.json) ·
  [prod 对比](../profiles/int4_dense_20261123/prod/sweep.json) ·
  [mid-M](../profiles/int4_dense_20261123/midlayered/sweep.json) ·
  [160–512](../profiles/int4_dense_20261123/midlayered2/sweep.json)
- 端到端：[ShareGPT bench](../profiles/int4_dense_20261123/bench_prefill.json) ·
  [graph decode](../profiles/int4_dense_20261123/graph_tuned/graph.json)
- nsys：[trace](../profiles/int4_dense_20261123/nsys/profile.nsys-rep) ·
  [analysis](../profiles/int4_dense_20261123/nsys/analysis.json)（sqlite 导出已删，可由
  `.nsys-rep` 重新生成）
- 代码：[cuda_int4.cu](../nanovllm/layers/csrc/cuda_int4.cu) ·
  [int4_dense_tuning.py](../benchmarks/int4_dense_tuning.py)

## 8. 剩余空间

- 选中 tile 仍只有 ~25 TFLOP/s（BF16 峰值 ~90）：下一步可试 `block_n=64`（减少 x tile
  重复读取与 CTA 数）、k 循环双缓冲（反量化与 MMA 重叠）、以及把 `dequantize` 的 8 次标量
  `cvt` 换成 4 次 `cvt.rn.bf16x2.f32`。
- 65–160 行区间各 variant 差距只有 ~10%，阈值可按其它机型/模型再标定；`compact` 的
  routed expert kernel 走的是另一套 tile，未受影响。
