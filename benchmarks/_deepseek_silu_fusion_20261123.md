# DeepSeek-V2-Lite compact gate/up：把 SiLU 融进 kernel

在 [GPU prefill 性能报告](_deepseek_sharegpt_final_perf_20261123.md) 的"下一步优先级"里，第 1 条是
**prefill gate/up 的 SiLU 融合**：`w4a16_compact_mma_gate_up_kernel` 当时已经一次 launch 同时算 gate/up，
但仍写回两份 intermediate，再由 Python 的 `F.silu(gate_out) * up_out` 产生 hidden。本轮把激活搬进
kernel epilogue，省掉 gate/up 写回和 elementwise。1–4 行 expert 的 GEMV 路径本来就在 kernel 内
完成 `silu_product`，这次不改。

## 1. 实现

### 1.1 新的单输出入口

```text
gemm_gate_up_silu_compact(x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs,
                          offsets, counts, hidden, max_rows, logical_rows,
                          group_size, cta_fanout, block_m, min_count, max_count)
```

对应 `w4a16_compact_mma_gate_up_silu_kernel<BF16, BLOCK_M>`：route-major grid、per-expert chunk
walk、MMA K 顺序全部与 split kernel 相同，只把 epilogue 从"写 gate / 写 up"改成"写
`silu(gate)*up`"。

### 1.2 32 行模板：每个 warp 同时持有 gate+up（关键）

第一版实现沿用 split 的 warp 分工（warp 0–1 gate、warp 2–3 up），epilogue 让 gate 半边把 BF16
值暂存到 shared，barrier 后由 up warp 做 `silu*up`。它在 kernel 级是**回退**的（routed 批量、
mean 48 行/expert）：

| ms | split kernel | +elementwise | 首版 staged fused |
|---|---:|---:|---:|
| T=512（mean 48 行/expert） | 2.083 | 2.265 | 2.567（+23.2%） |
| T=2048（mean 192 行/expert） | 9.093 | 9.601 | 9.895（+8.8%） |

原因不是寄存器或 shared（两者 REG:96 / SHARED 13,312 → 15,360，寄存器限占用 5 CTA 不变），而是
**epilogue 串行化**：gate warp 暂存 → barrier → up warp 单独做 16 次 expf+store，而 split 是 4 个
warp 并行写。chunk 越多、每个 CTA 反复付这笔钱，回退越大（T=2048 时 +8.8%，48 行/expert +23%）。

最终 32 行模板改成 **4 个 warp 各领 8 行 × 32 通道，同时算 gate 和 up**：

- 每个 warp 仍是 4 组 accumulator（2 个 16 通道 tile × 2 个投影），**registers 与 split 完全相同**；
- 共享 tile 仍是 `(2*block_n + BLOCK_M) * block_k`，**shared 字节不变**（13,312 B）；
- 每 k-step 仍是 5 条 ldmatrix + 4 条 MMA（原来 1×A + 4×B，现在 4×A/2 投影 + 1×B），指令数相同；
- 激活直接在寄存器里做，**没有跨 warp 暂存、没有额外 barrier**；
- 每个输出元素的 MMA K 累加顺序不变，因此位级等价。

### 1.3 8/16 行模板：保留 staged epilogue

5–8 行与 9–16 行模板继续用"warp 0–1 gate / 2–3 up + gate 暂存 shared"的写法。这两档每 expert
只有一个 chunk，暂存开销小于省下的写回，实测已经胜过 split（见 §3）；若套用 §1.2 的
4-warp 重排，`BLOCK_M=16` 要让 CTA 覆盖 64 列、shared 涨到 18,432 B，`BLOCK_M=8` 要覆盖 128 列、
shared 涨到 33,792 B（占用 8→3 CTA），反而更差，所以不采用。

### 1.4 Python 侧

- `moe_int4_fuse_gate_up_silu`（默认 `True`，`Config.moe_int4_fuse_gate_up_silu`）：
  `_forward_compact_int4` 只用一块 `hidden` 缓冲，ranges 里每个 band 一次 gate/up launch + 一次
  down launch；不再 `new_empty` 两块 gate/up、不再发射 `F.silu` 与 `mul`。
- `--no-moe-int4-fuse-silu`（`bench.py` / `nsys_profile.py` / `compact_graph.py`）保留旧的
  两输出 + elementwise 路径做位级/性能消融。
- 只影响 `moe_int4_backend="compact"`。默认后端仍是 `loop`，`segment` 后端的 Triton gate/up
  也保持原样。

## 2. 位级一致性

- `tests/test_moe_int4_compact.py`：**51 passed**。其中
  - `test_compact_int4_matches_per_expert_loop`：compact ↔ loop 参考，T=1…1024、top_k=1/2/6，
    `rtol=0, atol=0`（融合后的 MMA 路径与原路径逐位相同）；
  - `test_fused_silu_epilogue_matches_split_gate_up`：fused ↔ split，bf16/fp16、T=8…128，
    覆盖 MMA8/16/32 三档模板，`rtol=0, atol=0`；
  - `test_compact_int4_fused_silu_graph_replay`：CUDA Graph capture 后改输入 replay，与 eager 逐位相同；
  - `test_compact_int4_has_no_host_sync_after_warmup`：融合后仍无 device→host 同步。
- 加 `test_cuda_int4.py` / `test_moe_int4_segment.py` / `test_moe.py`：**84 passed**。
- 微基准对每个模板都先 `torch.equal(fused, split)` 再计时，全部通过。
- CUDA Graph decode 的 b1/b4/b8/b16/b32 输出 token 与 split 版本完全一致（`compact_graph.py` 的
  `tokens` 字段逐 token 相同）。

## 3. kernel 级收益

合成满路由：64 experts 全部激活、H=2048、I=1408、BF16、INT4 group 64、fanout 64，
每档 5 repeats 取中位数。`split kernel` 是两输出 gate/up kernel；`+elementwise` 再加
`F.silu(gate)*up`；`fused` 是单输出融合 kernel。

| rows/expert | BLOCK_M | split kernel ms | +elementwise ms | fused ms | kernel Δ | 含 elementwise Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 8 | 1.847 | 1.839 | 1.823 | −1.28% | −0.86% |
| 6 | 8 | 1.856 | 1.852 | 1.791 | −3.50% | −3.25% |
| 8 | 8 | 1.886 | 1.899 | 1.806 | −4.25% | −4.91% |
| 10 | 16 | 1.850 | 1.903 | 1.823 | −1.48% | −4.23% |
| 12 | 16 | 1.871 | 1.909 | 1.836 | −1.88% | −3.82% |
| 16 | 16 | 1.854 | 1.899 | 1.822 | −1.72% | −4.09% |
| 20 | 32 | 1.846 | 1.860 | 1.860 | +0.75% | 0.00% |
| 24 | 32 | 1.845 | 1.924 | 1.821 | −1.30% | −5.37% |
| 32 | 32 | 1.779 | 1.857 | 1.731 | −2.70% | −6.78% |
| 40 | 32 | 2.039 | 2.126 | 2.007 | −1.55% | −5.61% |
| 48 | 32 | 2.118 | 2.245 | 2.044 | −3.50% | −8.96% |
| 64 | 32 | 2.149 | 2.322 | 2.089 | −2.80% | −10.04% |
| 96 | 32 | 3.690 | 3.924 | 3.487 | −5.49% | −11.15% |

读法：

- 单看 gate/up kernel，融合后稳定快 **1–5%**（20 行/expert 的 +0.75% 在 ±1% 噪声内）；
- 把被删掉的 elementwise 算进去，收益 **3–11%**，且行数越多收益越大——省下的是
  `2 × rows × I` 的写回加两次读改写；
- 这一档时间主要由 64 个 expert 的权重流（每 call 约 184 MB INT4）决定，所以绝对收益只有
  0.03–0.4 ms/call。

## 4. prefill：真实模型 nsys 对比

同一套 ShareGPT 32 请求、`warmup-seqs=32`、CUDA Graph、`nsys_capture` 窗口，两次 capture
（fused / `--no-moe-int4-fuse-silu`）间隔约 5 分钟：

| prefill 阶段 | split | fused | 变化 |
|---|---:|---:|---:|
| wall | 644.1 ms | 627.5 ms | −2.6% |
| GPU active | 622.3 ms | 599.9 ms | −3.6% |
| kernel 数 | 2,191 | 2,139 | −52 |
| `w4a16_compact_mma_gate_up_*`（5+ 行） | 199.76 ms × 78 | 196.22 ms × 78 | **−1.8%** |
| `vectorized_elementwise_kernel` | 19.94 ms × 497 | 7.86 ms × 445 | **−60.6%，−52 个** |
| `w4a16_mma_kernel`（dense/shared） | 165.07 ms × 188 | 163.37 ms × 188 | −1.0% |
| `w4a16_compact_mma_kernel`（down） | 122.62 ms × 78 | 117.34 ms × 78 | −4.3% |
| `w4a16_compact_gate_up_kernel`（≤4 行 GEMV） | 46.19 ms × 26 | 45.88 ms × 26 | −0.7% |

- 少掉的 **52 = 2 × 26** 个 elementwise kernel 正好是每层 `F.silu` + `mul`，时间从 19.94 ms
  降到 7.86 ms（剩下的 445 个是别的逐元素算子）。
- gate/up kernel 自身 −1.8%，与 §3 中 MMA32 档的 −2~−3.5% 同量级。
- down kernel 的 −4.3%、decode GPU 的 +2.4% 说明两次 capture 之间有 **~2% 的时钟漂移**，所以
  "prefill GPU −3.6%" 里约一半是融合、一半是漂移；把逐 kernel 数字与精确的 kernel 计数当作主证据。

## 5. decode（CUDA Graph，成对）

同一进程参数、`compact_graph.py --mode graph`，5 repeats 取中位数，两个版本各跑一次：

| workload | split TPOT | fused TPOT | 变化 |
|---|---:|---:|---:|
| b1_ctx64 | 18.54 ms | 18.22 ms | −1.8% |
| b4_ctx64 | 29.64 ms | 29.73 ms | +0.3% |
| b8_ctx64 | 48.13 ms | 47.54 ms | −1.2% |
| b16_ctx64 | 72.29 ms | 72.60 ms | +0.4% |
| b32_ctx64 | 91.03 ms | 90.08 ms | −1.0% |

全部落在 ±1.8% 内，且正负都有：**decode TPOT 无系统性变化**。原因很直接——DSV2-Lite 有 64 个
expert、top_k=6，b≤32 时每个 expert 平均不到 4 行，decode 走的是 1–4 行 GEMV，而 GEMV 的
`silu_product` 本来就在 kernel 里。融合只覆盖 5+ 行的 band，decode 基本碰不到。
graph replay 3 次、fallback 0，输出 token 与 split 版本逐 token 相同。

## 6. 端到端 prefill 成对 A/B

跨进程跑 ShareGPT（32 请求 × `--repeat-batches 4`，先 fused 再 split）得到：

| 指标（跨进程） | split | fused |
|---|---:|---:|
| TTFT（batch 0） | 967.5 ms | 791.5 ms |
| TPOT | 106.9 ms | 107.3 ms |
| wall（batch 0） | 4.282 s | 4.119 s |
| prefill step（batch 1–3 中位数） | 0.689 s | 0.693 s |

batch 0 的 TTFT 差 18%，但 batch 1–3 的 prefill step 两边都是 ~0.69 s：跨进程的 batch 0 里混着
首次 batch 的分配器/编译状态。更关键的是，同一配置早前的 4 次跨进程运行 TTFT 曾在 758–892 ms 之间
漂移（那 4 次因 `bench.py` 的 `--no-moe-int4-fuse-silu` dest 写错，实际都跑了融合路径；已修复，
现在 JSON 的 `benchmark_config.moe_int4_fuse_gate_up_silu` 会记录开关）。因此跨进程数据**不作为
主证据**。

改用**单进程内交替开关**：加载一次模型，先把 32 条请求各跑一遍预热，然后 20 对交替执行
fused / split，取 `step_stats.prefill_time`：

| 指标（单进程成对） | split | fused | 变化 |
|---|---:|---:|---:|
| prefill step 中位数（1,257 tok） | 705.4 ms | 677.9 ms | −3.9% |
| 逐对胜率 | — | 16 / 20 | — |

- 逐对中位数 −2.4%，20 对里 16 对 fused 更快；但逐对波动很大（−23.7% ~ +13.2%），有大块
  离群值，所以**只能表述为"约 2–4%、方差大"**，不是稳定可复现的定值。
- split 路径的 prefill 更容易出现 >800 ms 的离群（872.9 / 822.4 / 808.3 / 807.3 ms），fused 最大
  778.9 ms。一个未验证的解释是 split 每层额外 `new_empty` 两块 21 MB 缓冲 + 一次 elementwise
  输出分配，长时间迭代下分配器碎片/回收更差；与之相符的是跨进程 batch 1–3 里两边又回到同一水平
  （分配器已预热）。这里只记录现象，不当作结论。

## 7. 结论与限制

1. **融合本身是确定的**：gate/up 的两次写回和每层 2 个 elementwise kernel 已经被删掉
   （nsys prefill kernel 数 2,191 → 2,139，`vectorized_elementwise` 19.94 → 7.86 ms），
   gate/up kernel 在合成微基准里快 1–5%。
2. **端到端收益有限**：prefill step 约 −2~−4%（方差大），prefill GPU 约 −3.6%（含 ~2% 漂移）；
   decode TPOT 不变。原因是 prefill 的 gate/up 只占整步约 29%，而其中真正的 SiLU/写回开销
   只有 ~20 ms；真正的瓶颈仍是 64 个 expert × 26 层的 INT4 权重流。
3. **32 行模板必须做 warp 重排**：沿用 split 的 warp 分工、只把激活塞进 epilogue，会在
   多 chunk 的 prefill 场景回退 8–23%；这是本轮最重要的实现教训。
4. 默认后端仍是 `loop`，本优化只对 `moe_int4_backend="compact"` 生效；`segment` 后端的
   `moe_segment_int4_gup` 仍是 `[R, 2I]` + elementwise，未在本次范围内。
5. 位级等价依赖 `silu_product` 的舍入顺序（BF16 存回 gate/up → `x/(1+exp(-x))` → 再乘），
   与 ATen `F.silu` + `mul` 完全一致；换成 `__expf` 或先乘后舍入都会破坏位级一致。

## 8. 产物

- kernel 微基准：[kernel_ab.json](../profiles/deepseek_silu_fusion_20261123/kernel_ab.json) ·
  [script](../benchmarks/compact_silu_fusion.py)
- ShareGPT nsys：[fused analysis](../profiles/deepseek_silu_fusion_20261123/nsys_fused/analysis.json) ·
  [split analysis](../profiles/deepseek_silu_fusion_20261123/nsys_split/analysis.json) ·
  [fused trace](../profiles/deepseek_silu_fusion_20261123/nsys_fused/profile.nsys-rep) ·
  [split trace](../profiles/deepseek_silu_fusion_20261123/nsys_split/profile.nsys-rep)
- Graph decode：[fused](../profiles/deepseek_silu_fusion_20261123/graph_fused/graph.json) ·
  [split](../profiles/deepseek_silu_fusion_20261123/graph_split/graph.json)
- 单进程成对 prefill：[engine20](../profiles/deepseek_silu_fusion_20261123/engine20/kernel_ab.json) ·
  [engine（8 对）](../profiles/deepseek_silu_fusion_20261123/engine/kernel_ab.json)
- 跨进程 ShareGPT（只作漂移记录）：[fused](../profiles/deepseek_silu_fusion_20261123/bench_pair_fused.json) ·
  [split](../profiles/deepseek_silu_fusion_20261123/bench_pair_split.json)
- 代码：[cuda_int4.cu](../nanovllm/layers/csrc/cuda_int4.cu) · [moe.py](../nanovllm/layers/moe.py) ·
  [config.py](../nanovllm/config.py)
- 测试：[test_moe_int4_compact.py](../tests/test_moe_int4_compact.py)
