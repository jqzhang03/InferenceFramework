# DeepSeek-V2-Lite INT4 路由索引 GPU 打包与 expert 分段 dispatch

RTX 5060 Ti，WSL，PyTorch 2.8.0，CUDA 12.4/12.8 工具链，TP=1，BF16 激活与 MLA KV，INT4 group 64，流式载入。四组固定负载与 2026-11-18 瓶颈报告一致：`b1_ctx64`、`b4_ctx64`、`b8_ctx64`、`b4_ctx512`，每请求输出 4 token。计时每配置 6 轮，按 `cuda_loop`、`cuda_compact`（每层一次 host 打包）、`cuda_gpu_compact` 交错执行；每次仍执行真实 router 和 sampler。logits 校验先固定输入与路由再比较。

## 实现

`Config.moe_int4_backend="compact"` 新增一条 decode 专用路径：

- router 的 `[T, k]` 索引在 GPU 上稳定排序；`index_add_` 在固定大小直方图上生成 `counts`，`cumsum` 生成 `offsets`。同一份 `order` 用于行 gather、expert 权重指针选择和 `inverse` scatter；不再对每个 expert 做布尔掩码索引。
- [moe.py:271](../nanovllm/layers/moe.py#L271) 的 `_forward_compact_int4` 把 `offsets`/`counts` 与六张 expert 指针表传给 CUDA 扩展，由 [cuda_int4.cu:146](../nanovllm/layers/csrc/cuda_int4.cu#L146) 的 2D grid kernel 一次覆盖全部 expert；每个 expert 的 CTA 从设备内存读取 `counts/offsets`，空 expert 直接退出。gate/up 融合一次 launch，down 一次 launch。
- 每个 expert 最多 16 个 CTA，CTA 以 stride 遍历剩余输出 tile；这样空 expert 不会展开完整的 88/128-tile grid。decode 行数 ≤8 时使用 1/2/4/8 行模板，`rows` 由设备端 `counts` 决定。
- 最终按 `top_idx.argsort` 的 expert 升序逐 slot 累加，保持 `_forward_loop` 的累加顺序；`torch.bincount` 会因动态输出长度触发同步，已换成固定大小 `index_add_` 直方图。
- 大于 8 行的 prefill chunk 走 [moe.py:239](../nanovllm/layers/moe.py#L239) 的 `_forward_compact_host`：每层只读一次路由，调用原有 expert 线性层，保持 MMA 路径与位级输出；它仍然是每层一次 D2H，不是零同步路径。

`cuda_int4.cu` 的既有 `gemv_columns` 增加 `rows` 掩码；原 1–4 行 expert 融合和 `gemm_out`/MMA 路径保持行为不变。CUDA W4A16 的安装、TP=1、group size 和非 AWQ/非 `w_deq` 约束沿用原检查。

## 正确性

- `tests/test_moe_int4_compact.py` 31 项通过：top_k 1/2/6，token 数 1/2/3/4/5/8/9/16/17/32，输出与 `_forward_loop` 位级一致；T>8 的用例走 host 打包回退。同步调试模式下 decode forward 不再调用同步 API。
- 既有 `tests/test_cuda_int4.py` + `tests/test_moe_int4_segment.py` 28 项通过，说明 `gemv_columns` 的 `rows` 参数没有改变原融合/分段路径。
- 四组负载的 logits 对照见下表。`b1`、`b4`、`b4_ctx512` 与 `cuda_loop` 位级一致；`b8` 的相对 L2 为 4.856e-3，最大绝对差 0.28125。原因是 b8 中 5–8 行的 expert 在参考路径走 MMA（`launch` 的 M≤8 分支），compact kernel 走 GEMV；累加顺序一致，差异来自 BF16 舍入。若要求 b8 位级一致，应让 b8 走 host 打包回退或给 compact 增加 5–8 行 MMA 分支。

| 负载 | 位级一致 | 相对 L2 | 最大绝对差 |
|---|---|---:|---:|
| b1_ctx64 | 是 | 0 | 0 |
| b4_ctx64 | 是 | 0 | 0 |
| b8_ctx64 | 否（容差 1e-2） | 4.856e-3 | 0.28125 |
| b4_ctx512 | 是 | 0 | 0 |

## 端到端计时

单元格为 6 轮中位数；TTFT/TPOT 单位 ms，wall 为整批耗时。`cuda_compact` 是原有每层 host 打包诊断，作为对照。

| 负载 | 路径 | TTFT | TPOT | wall |
|---|---|---:|---:|---:|
| 1 × 64 | cuda_loop | 708.89 | 537.96 | 2,318.43 |
| 1 × 64 | cuda_compact | 237.98 | 40.51 | 376.30 |
| 1 × 64 | **cuda_gpu_compact** | **231.76** | **63.57** | **431.83** |
| 4 × 64 | cuda_loop | 880.61 | 533.52 | 2,515.92 |
| 4 × 64 | cuda_compact | 386.24 | 54.69 | 550.49 |
| 4 × 64 | **cuda_gpu_compact** | **384.94** | **61.13** | **568.46** |
| 8 × 64 | cuda_loop | 994.19 | 559.52 | 2,675.48 |
| 8 × 64 | cuda_compact | 483.70 | 74.23 | 706.00 |
| 8 × 64 | **cuda_gpu_compact** | **486.95** | **60.85** | **675.68** |
| 4 × 512 | cuda_loop | 1,470.09 | 618.70 | 3,407.33 |
| 4 × 512 | cuda_compact | 941.52 | 56.50 | 1,112.04 |
| 4 × 512 | **cuda_gpu_compact** | **937.88** | **57.59** | **1,111.72** |

相对 `cuda_loop`，`cuda_gpu_compact` 的 TPOT 配对中位降幅为 **−89.0%、−88.7%、−89.1%、−90.7%**，四组均 6/6 胜。相对 `cuda_compact`，compact 在 b8 为 −18.9%（6/6 胜），b4 为 +12.2%（0/6 胜），b1 为 +43.2%（1/6 胜），b4_ctx512 为 +3.2%（1/6 胜）。也就是说，零 D2H 的收益在 b8 最明显；b1/b4 的 host 打包因只启动非空 expert 并使用精确行数，decode 仍更快。TTFT 主要来自 prefill，compact 与 host 打包接近，均显著低于 loop。

## Routed expert 区间 D2H

Nsight Systems 采集 `b4_ctx64`、`b4_ctx512`，按 NVTX `component/routed_experts` 归属 CUDA memcpy 与 kernel。单元格为单个 step 的 D2H 次数 / routed CPU 区间 / GPU kernel 数。

| 采集 | step | cuda_loop | cuda_compact | cuda_gpu_compact |
|---|---|---:|---:|---:|
| b4_ctx64 | prefill | 4,992 / 1,255.89 ms / 31,820 | 26 / 567.51 ms / 9,069 | 26 / 452.34 ms / 9,173 |
| b4_ctx64 | decode ×3 | 4,992 / 712–783 ms / 20,950–21,112 | 26 / 35–94 ms / 1,466–1,547 | **0 / 65–253 ms / 1,144** |
| b4_ctx512 | prefill | 4,992 / 1,618.91 ms / 39,334 | — | 26 / 1,095.08 ms / 11,518 |
| b4_ctx512 | decode ×3 | 4,992 / 764–1,027 ms / 20,902–21,094 | — | **0 / 48–94 ms / 1,144** |

decode 的 compact 路径 D2H 为 0，routed expert 区间从 20k+ kernel 降到 1,144；prefill 仍为 26 次/层（host 回退），这是有意的精度取舍。b1/b8 的 decode 走同一 GPU kernel（T≤8），不额外读取路由，因此同样为 0 D2H；b8 的 logits 容差见上。

## 结论与后续

- GPU 路由把 decode 的 4,992 次 D2H/同步降为 0，TPOT 相对逐 expert 路径降低约 89–91%，与 2026-11-18 报告“每层 host 打包”的 87.5–91.8% 同量级；零 D2H 在 b8 还额外赢过 host 打包。
- 当前 compact 不是全面快于 host 打包：b1/b4 的 host 打包用精确行数和更少的空 CTA，decode 更快。若只追求 TPOT，host 打包仍可用；若目标是去掉同步、为后续 CUDA Graph 铺路，compact 更合适。
- 下一步：给 compact 增加 5–8 行 MMA 分支以恢复 b8 位级一致；把 prefill 也做成 GPU 分段（当前 host 回退仍有 26 D2H/层）；在固定容量后捕获 decode CUDA Graph；继续调 CTA fan-out（当前 16）和 `torch.zeros`/scatter 的分配开销。
- 默认仍是 `loop`；`compact` 需显式选择且要求 CUDA W4A16、TP=1、纯 INT4。

## 复现

```bash
out=profiles/deepseek_compact_20261122
bash benchmarks/run_deepseek_w4a16.sh bottleneck "$out"
# 或直接使用已归档结果：
python -m benchmarks.deepseek_bottleneck --root "$out" --repeats 6
/usr/local/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o "$out/trace/bottleneck" python -m benchmarks.deepseek_bottleneck \
  --root "$out/trace" --trace --repeats 1
/usr/local/bin/nsys export --type=sqlite -o "$out/trace/bottleneck.sqlite" \
  "$out/trace/bottleneck.nsys-rep"
python -m benchmarks.deepseek_bottleneck_trace "$out/trace/bottleneck.sqlite"
```

[计时原始数据](../profiles/deepseek_compact_20261122/timing.json) · [Nsight 归属 JSON](../profiles/deepseek_compact_20261122/trace/trace.json) · [compact 正确性测试](../tests/test_moe_int4_compact.py) · [实现](../nanovllm/layers/moe.py#L218)
