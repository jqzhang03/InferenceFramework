# DeepSeek-V2-Lite INT4 路由索引 GPU 打包、CTA fan-out 与 decode CUDA Graph

> 2026-11-23 follow-up：prefill 也已改走 compact GPU MMA32 chunk（T≤4096），prefill/mixed 的 routed D2H 从 27 次降到 1 次 sampler-only，5,180 次逐 expert MMA 调用消除；ShareGPT wall 约 893→889 ms（中性），GPU 时间 +9.6%。decode 继续把 5+ 行 gate/up 融合为一个 128-thread MMA kernel，fan-out 改为 8/128/64；b8 graph 约 −4.8%，其余基本持平。详见 [GPU prefill follow-up](_deepseek_sharegpt_gpu_prefill_20261123.md) 和 [fusion follow-up](_deepseek_compact_fusion_20261123.md)。

RTX 5060 Ti，WSL，PyTorch 2.8.0，CUDA 12.4/12.8 工具链，TP=1，BF16 激活与 MLA KV，INT4 group 64，流式载入。四组固定负载与 2026-11-18 瓶颈报告一致：`b1_ctx64`、`b4_ctx64`、`b8_ctx64`、`b4_ctx512`，每请求输出 4 token。计时每配置 6 轮，按 `cuda_loop`、`cuda_compact`（每层一次 host 打包）、`cuda_gpu_compact` 交错执行；每次仍执行真实 router 和 sampler。logits 校验先固定输入与路由再比较。

## 实现

`Config.moe_int4_backend="compact"` 新增一条 decode 专用路径：

- router 的 `[T, k]` 索引在 GPU 上稳定排序；`index_add_` 在固定大小直方图上生成 `counts`，`cumsum` 生成 `offsets`。同一份 `order` 用于行 gather、expert 权重指针选择和 `inverse` scatter；不再对每个 expert 做布尔掩码索引。
- [moe.py:270](../nanovllm/layers/moe.py#L270) 的 `_forward_compact_int4` 把 `offsets`/`counts` 与六张 expert 指针表传给 CUDA 扩展，由 [cuda_int4.cu:146](../nanovllm/layers/csrc/cuda_int4.cu#L146) 的 2D grid kernel 一次覆盖全部 expert；每个 expert 的 CTA 从设备内存读取 `counts/offsets`，空 expert 直接退出。
- 1–4 行的 expert 走 GEMV：gate/up 融合一次 launch，down 一次 launch。5–8、9–16、17–32 行的 expert 分别走 [cuda_int4.cu:309](../nanovllm/layers/csrc/cuda_int4.cu#L309) 的 compact MMA8/16/32 投影：gate、up、down 各一次 launch，中间 `F.silu(gate)*up`。各范围按 `counts[e]` 在设备端互斥选择，与参考实现的 `launch` 分派一致，因此 logits 位级一致。
- 最终按 `top_idx.argsort` 的 expert 升序逐 slot 累加，保持 `_forward_loop` 的累加顺序；`torch.bincount` 会因动态输出长度触发同步，已换成固定大小 `index_add_` 直方图。
- 大于 32 行的 prefill chunk 走 [moe.py:238](../nanovllm/layers/moe.py#L238) 的 `_forward_compact_host`：每层只读一次路由，调用原有 expert 线性层，保持 MMA 路径与位级输出；它仍然是每层一次 D2H，不是零同步路径。

`cuda_int4.cu` 的既有 `gemv_columns` 增加 `rows` 掩码；原 1–4 行 expert 融合和 `gemm_out`/MMA 路径保持行为不变。CUDA W4A16 的安装、TP=1、group size 和非 AWQ/非 `w_deq` 约束沿用原检查。

## CTA fan-out 调优

`Config.moe_int4_cta_fanout` 控制每个 expert 展开多少个 CTA；`0` 表示按 T 自动：

- T≤4：8 个 CTA/expert；
- 5≤T≤8：64 个 CTA/expert；
- T>8：128 个 CTA/expert。

10 轮中位数（TPOT，ms；同一进程内切换 fan-out）：

| 负载 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| b1_ctx64 | 68.75 | **56.25** | 57.44 | 56.06 | 59.64 | 56.63 |
| b4_ctx64 | 58.78 | **58.89** | 59.43 | 58.13 | 91.50* | 58.22 |
| b8_ctx64 | 121.19 | 86.99 | 71.47 | 65.03 | **61.35** | 63.06 |
| b16_ctx64 | 193.66 | 135.75 | 101.08 | 98.22 | 79.32 | **78.58** |
| b32_ctx64 | 279.20 | 189.31 | 136.90 | 115.18 | 101.21 | **100.94** |

\* b4 的 64 是本轮异常值；b4 在 8–128 之间基本平坦。b1/b4 的差异在噪声内；b8 的 64 最好，b16/b32 的 128 最好。自动规则按 T 取 8/64/128。

fan-out 只改变“每个 expert 的输出列拆给多少个 CTA”，不直接指定“同时几个 expert”。GPU 仍在 36 个 SM 上调度整个 `(experts, ctas)` grid。

## 正确性

- `tests/test_moe_int4_compact.py` 31 项通过：top_k 1/2/6，token 数 1/2/3/4/5/8/9/16/17/32，输出与 `_forward_loop` 位级一致；T=9–32 现在走 GPU MMA8/16/32，T>32 才回退 host 打包。同步调试模式下 decode forward 不再调用同步 API。
- 既有 `tests/test_cuda_int4.py` + `tests/test_moe_int4_segment.py` 28 项通过，说明 `gemv_columns`/`mma_tile` 的重构没有改变原融合/分段路径。
- b16/b32 的完整模型 logits 也与 `cuda_loop` 位级一致；1–4 行 GEMV 与 5–8/9–16/17–32 行 MMA 按 `counts[e]` 互斥，覆盖参考实现的分派。

| 负载 | 位级一致 | 相对 L2 | 最大绝对差 |
|---|---|---:|---:|
| b1_ctx64 | 是 | 0 | 0 |
| b4_ctx64 | 是 | 0 | 0 |
| b8_ctx64 | 是 | 0 | 0 |
| b4_ctx512 | 是 | 0 | 0 |

## 端到端计时（eager compact + 自动 fan-out）

单元格为 6 轮中位数；TTFT/TPOT 单位 ms，wall 为整批耗时。`cuda_compact` 是每层 host 打包对照。

| 负载 | 路径 | TTFT | TPOT | wall |
|---|---|---:|---:|---:|
| 1 × 64 | cuda_loop | 725.12 | 539.89 | 2,344.92 |
| 1 × 64 | cuda_compact | 213.46 | 40.94 | 344.01 |
| 1 × 64 | **cuda_gpu_compact** | **209.08** | **55.71** | **376.88** |
| 4 × 64 | cuda_loop | 1,051.74 | 570.64 | 2,785.80 |
| 4 × 64 | cuda_compact | 393.00 | 53.73 | 554.73 |
| 4 × 64 | **cuda_gpu_compact** | **376.83** | **57.40** | **547.71** |
| 8 × 64 | cuda_loop | 956.14 | 653.32 | 3,193.81 |
| 8 × 64 | cuda_compact | 473.94 | 73.62 | 697.66 |
| 8 × 64 | **cuda_gpu_compact** | **477.43** | **64.10** | **665.63** |
| 4 × 512 | cuda_loop | 1,499.86 | 607.48 | 3,304.39 |
| 4 × 512 | cuda_compact | 940.22 | 55.16 | 1,105.51 |
| 4 × 512 | **cuda_gpu_compact** | **929.17** | **59.39** | **1,106.29** |

相对 `cuda_loop`，自动 fan-out 的 TPOT 配对中位降幅为 **−89.6%、−89.9%、−88.9%、−89.2%**，四组均 6/6 胜。相对 host 打包：b1 +35.9%（0/6 胜）、b4 +6.3%（1/6 胜）、b8 −16.3%（4/6 胜）、b4_ctx512 +5.6%（1/6 胜）。b1 的 host 打包仍最快；b4/b4_ctx512 已接近；b8 上 compact 胜出，同时保持零 D2H。

## decode CUDA Graph

Qwen3-0.6B 的 dense decode 原本已有 `capture_cudagraph`；DSV2-Lite 之前因 MLA/动态 MoE 走 eager。现在 compact 的 decode 是固定形状、无 host 路由读取，因此可以直接捕获：

- `capture_cudagraph` 在 compact 模式下捕获 `[1,2,4,8,16,32]`（[model_runner.py:1741](../nanovllm/engine/model_runner.py#L1741)）；更大的 decode batch（T>32）会走 host 打包，无法进 graph，`run_model` 明确 eager 回退。
- `graph_vars` 继续保存静态输入/输出/block table/context lens；compact 的 `torch.argsort`、`index_add_`、compact kernel 都在图内，无 `.tolist()`/`.item()`。
- `compact_graph.py` 用同一进程分别跑 eager/graph，固定 seed 比较 4-token 输出；graph 的 `decode_cudagraph_replay_count` 每轮 3，fallback 0。

| 负载 | eager TPOT | graph TPOT | graph 重放/回退 | 输出 token |
|---|---:|---:|---|---|
| b1_ctx64 | 84.08* | **19.43** | 3 / 0 | 与 eager 一致 |
| b4_ctx64 | 60.20 | **30.75** | 3 / 0 | 与 eager 一致 |
| b8_ctx64 | 60.75 | **52.11** | 3 / 0 | 与 eager 一致 |
| b16_ctx64 | 80.73 | **74.37** | 3 / 0 | 与 eager 一致 |
| b32_ctx64 | 126.18 | **94.82** | 3 / 0 | 与 eager 一致 |

\* eager b1 本组有异常慢样本；同轮 6 轮 eager 中位为 55.71。graph 的收益在 b1/b4 最大（launch 税占比高）；b8 约 −14%，b16 约 −8%，b32 约 −25%（b32 的 MMA/engine 波动更大）。

边界验证：`graph_bs=[1,2,4,8,16,32]`；batch>32 会在 `run_model` 里走 eager 回退。prefill 仍是 host 回退 + eager，不在 graph 内。

## Routed expert 区间 D2H

Nsight Systems 采集 `b4_ctx64`、`b4_ctx512`，按 NVTX `component/routed_experts` 归属 CUDA memcpy 与 kernel。单元格为单个 step 的 D2H 次数 / routed CPU 区间 / GPU kernel 数。

| 采集 | step | cuda_loop | cuda_compact | cuda_gpu_compact |
|---|---|---:|---:|---:|
| b4_ctx64 | prefill | 4,992 / 1,255.89 ms / 31,820 | 26 / 567.51 ms / 9,069 | 26 / 452.34 ms / 9,173 |
| b4_ctx64 | decode ×3 | 4,992 / 712–783 ms / 20,950–21,112 | 26 / 35–94 ms / 1,466–1,547 | **0 / 65–253 ms / 1,144** |
| b4_ctx512 | prefill | 4,992 / 1,618.91 ms / 39,334 | — | 26 / 1,095.08 ms / 11,518 |
| b4_ctx512 | decode ×3 | 4,992 / 764–1,027 ms / 20,902–21,094 | — | **0 / 48–94 ms / 1,144** |

decode 的 compact 路径 D2H 为 0，routed expert 区间从 20k+ kernel 降到 1,144；prefill T>32 仍为 26 次/层（host 回退）。这份 trace 采集于固定 fan-out=16、T≤8 的 compact 版本；fan-out、T=16/32 的 MMA 路径和 CUDA Graph 都不改变 decode 的 D2H 次数（Graph 仍复用同一批设备端 kernel）。decode CUDA Graph 省的是 kernel launch/CPU 税，sampler 的 1 次 `.tolist()` 仍在图外。

## 结论与后续

- GPU 路由把 decode 的 4,992 次 D2H/同步降为 0，自动 fan-out 后 TPOT 相对逐 expert 路径降低约 89–90%。
- 5–8/9–16/17–32 行 MMA8/16/32 分支补齐后，b8/b16/b32 全部位级一致；b8 的 MMA 版比非位级 GEMV 版多 3 个 launch，但避免了 logits 偏差。
- decode CUDA Graph 已可用：b1/b4/b8 均 3 次重放、0 回退、输出 token 与 eager 一致；收益随 batch 增大而下降（launch 税占比变小）。
- 边界：prefill T>32 仍是 host 回退（26 D2H/层）；decode T>32 回落 eager；compact 只支持纯 INT4、TP=1、CUDA W4A16、非 AWQ/无 `w_deq`；b1 的 host 打包仍更快。
- 下一步：GPU prefill 分段（去掉 prefill 26 D2H 并让 prefill 也可 graph）、decode T>32 的 GPU 路径、以及更大 batch 的 graph 容量。

## 复现

```bash
out=profiles/deepseek_compact_20261122
bash benchmarks/run_deepseek_w4a16.sh bottleneck "$out"
# 或直接使用已归档结果：
python -m benchmarks.deepseek_bottleneck --root "$out" --repeats 6
python -m benchmarks.compact_fanout --root "$out" --fanouts 4,8,16,32,64,128 \
  --repeats 10 --output fanout_rep10.json
python -m benchmarks.compact_graph --root "$out/graph" --mode eager --repeats 3
python -m benchmarks.compact_graph --root "$out/graph" --mode graph --repeats 3
/usr/local/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o "$out/trace/bottleneck" python -m benchmarks.deepseek_bottleneck \
  --root "$out/trace" --trace --repeats 1
/usr/local/bin/nsys export --type=sqlite -o "$out/trace/bottleneck.sqlite" \
  "$out/trace/bottleneck.nsys-rep"
python -m benchmarks.deepseek_bottleneck_trace "$out/trace/bottleneck.sqlite"
```

[计时原始数据](../profiles/deepseek_compact_20261122/timing.json) · [fan-out sweep](../profiles/deepseek_compact_20261122/fanout_rep10.json) · [graph/eager](../profiles/deepseek_compact_20261122/graph/graph.json) · [Nsight 归属 JSON](../profiles/deepseek_compact_20261122/trace/trace.json) · [compact 正确性测试](../tests/test_moe_int4_compact.py)
