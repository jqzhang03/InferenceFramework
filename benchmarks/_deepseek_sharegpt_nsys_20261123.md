# DeepSeek-V2-Lite INT4 compact：ShareGPT 推理瓶颈 Nsight 报告

> 这是 GPU prefill 改造前的 baseline，保留用于对照。改造后的 final 数据和 before/after 见 [GPU prefill follow-up](_deepseek_sharegpt_gpu_prefill_20261123.md)。

RTX 5060 Ti 16GB，WSL，PyTorch 2.8.0，CUDA 12.4/12.8 工具链，TP=1，INT4 group 64，streaming load，`moe_int4_backend="compact"`，`max_num_seqs=32`，`max_model_len=1024`，`gpu_memory_utilization=0.85`。模型使用 [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)。

## 负载与采集

从 ShareGPT v3 清洗数据生成 32 条 burst 请求，保留真实用户 prompt；总 prompt 1,769 token（单请求 2–435），每请求固定输出 32 token。请求、输出预算与 token 哈希见 [sharegpt_profile_20261123.jsonl](../results/sharegpt_profile_20261123.jsonl)。

两次独立采集：

- **eager compact**：`enforce_eager=True`，便于用 NVTX 归因到 DeepSeek 模块；
- **graph compact**：`enforce_eager=False`，compact decode 捕获 batch 1/2/4/8/16/32 的 CUDA Graph。

每次采集均先做 8 条 shape-matched warmup，`nsys_capture` 只覆盖正式推理。33 个 engine step：**1 prefill + 1 mixed + 31 decode**；decode 行数共 992（31×32），所有请求固定 32 token。

采集命令（eager；graph 只多 `--cuda-graph`）：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o profiles/sharegpt_deepseek_20261123/eager/profile \
  python -m benchmarks.nsys_profile \
    --model ~/huggingface/DeepSeek-V2-Lite \
    --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
    --quantization int4 --int4-group-size 64 --streaming-load \
    --moe-int4-backend compact --moe-int4-cta-fanout 0 \
    --max-num-seqs 32 --max-model-len 1024 --gpu-memory-utilization .85 \
    --warmup-seqs 8 \
    --gpu-event-json profiles/sharegpt_deepseek_20261123/eager/events.json
```

> `nsys stats` 的 kernel/copy 汇总包含模型加载和初始化；本报告的 GPU 活跃、D2H、kernel 数均通过 `nsys_capture` NVTX 范围过滤，避免把 20s 级权重 H2D 误认为推理开销。

## 总览

| 采集 | capture wall | GPU active | GPU idle | engine steps |
|---|---:|---:|---:|---:|
| eager compact | 12,776.6 ms | 3,888.1 ms | 8,888.5 ms（69.6%） | 33 |
| graph compact | 6,474.4 ms | 4,167.6 ms | 2,306.8 ms（35.6%） | 33 |

eager 的 capture 被 **最后一个 decode step 的一次性编译**拉长（6,198 ms wall / 92 ms GPU）；graph 采集同样有尾部编译（1,792 ms / 58 ms GPU），但整体短一半。

## eager compact：按阶段

| 阶段 | steps | wall | GPU active | GPU idle | kernel 数 | D2H |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 893.3 ms | 734.5 ms | 158.8 ms | 12,851 | 27 / 2.20 MB |
| mixed | 1 | 1,799.3 ms | 139.1 ms | **1,660.2 ms** | 10,172 | 27 / 47.7 KB |
| decode（31） | 31 | 10,061.5 ms | 3,014.6 ms | 7,046.9 ms | 68,088 | 32 / 7.7 KB |
| decode 稳态（前 30） | 30 | 3,863.2 ms | 2,922.5 ms | 940.7 ms | — | 30 / 7.4 KB |
| decode 最后一步 | 1 | 6,198.3 ms | 92.1 ms | 6,106.2 ms | 2,458 | 2 / 12 B |

关键数字：

- **prefill 的 91.9% GPU 时间是 `w4a16_mma_kernel<BF16,32>`**：675.9 ms、5,180 次调用。它是 `_forward_compact_host` 回退到逐 expert 参考 MMA 的结果，不是 compact decode kernel。
- **mixed step 的 wall 为 1.80 s，但 GPU 只有 139 ms**；1.66 s 是 host/CPU 在等 Python 打包、逐 expert 调用和一次 late compile，属于 prefill/mixed 的 host 回退。
- **decode 稳态约 128.8 ms wall / 97.4 ms GPU per step**，GPU 活跃约 75.6%；CPU/采样/引擎税约 31 ms/step。
- decode 的 routed expert D2H 为 0；每步只有 1 次 sampler `.tolist()` 的 D2H，共 30–32 次。

## graph compact：按阶段

| 阶段 | steps | wall | GPU active | GPU idle | kernel/图节点 | D2H |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 1,072.4 ms | 727.2 ms | 345.2 ms | 12,851 | 27 / 2.20 MB |
| mixed | 1 | 290.3 ms | 141.7 ms | 148.7 ms | 9,969 | 27 / 47.7 KB |
| decode（31） | 31 | 5,108.5 ms | 3,298.8 ms | 1,809.7 ms | 966（31 次 graph replay） | 32 / 7.7 KB |
| decode 稳态（前 30） | 30 | 3,316.9 ms | 3,241.1 ms | **75.8 ms** | 30 次 replay | 30 / 7.4 KB |
| decode 最后一步 | 1 | 1,791.6 ms | 57.7 ms | 1,733.9 ms | 636 | 2 / 12 B |

- decode 稳态 **wall 110.6 ms / GPU 108.0 ms per step**，GPU 活跃约 **97.7%**；graph replay 本身约 100–107 ms/step。
- graph 省掉的是 decode 的 CPU/launch 税，不改变 GPU kernel 时长；eager 稳态 GPU 约 97 ms，graph 约 108 ms（graph 含固定容量 padding/额外节点），但 wall 从约 129 ms 降到 111 ms。
- prefill/mixed 仍是 eager/host 路径，graph 不覆盖；prefill GPU 时间两者一致（约 727–735 ms）。mixed 的 wall 差异（1.80 s vs 0.29 s）主要来自 eager 采集的首次 late compile/预热状态，不能当成 graph 对 mixed 的加速。

## top kernel 与归因

eager prefill 顶层 GPU kernel：

| kernel | 时间 | 次数 | 说明 |
|---|---:|---:|---|
| `w4a16_mma_kernel<BF16,32>` | 675.9 ms | 5,180 | 逐 expert 参考 MMA（host fallback） |
| `indexFuncLargeIndex` | 17.6 ms | 1,664 | host 回退的 index/gather |
| `flash_fwd_kernel` | 7.4 ms | 27 | prefill MLA attention |
| `elementwise_kernel` / `vectorized_elementwise` | 7.9 / 7.3 ms | 1,744 / 3,487 | router、SiLU、scatter 等 |

eager 稳态 decode 单步（step 2）GPU 约 91.9 ms：

| kernel | 时间 | 次数 | 说明 |
|---|---:|---:|---|
| `w4a16_compact_mma_kernel<BF16,8>` | 29.2 ms | 234 | 5–8 行 expert 的 compact MMA |
| `w4a16_compact_gate_up_kernel<BF16,4>` | 28.5 ms | 26 | 1–4 行 expert 的 GEMV gate/up |
| `w4a16_mma_kernel` | 17.2 ms | 215 | dense/shared 路径 |
| `w4a16_compact_down_kernel<BF16,4>` | 10.3 ms | 26 | compact down |
| `mla_decode_kernel` | 1.6 ms | 27 | MLA decode |

也就是说 decode 的 GPU 时间主要仍在 compact expert 投影（约 68 ms/step），MLA 本身很小；这部分已经没有 D2H，属于后续融合/调 tile 的空间。

## 尾部编译：不要当成稳态

eager 最后一步 6,198 ms wall / 92 ms GPU，同时出现：

- `cuModuleLoad` 13 次，37 ms；
- `cudaDeviceSynchronize` 27 次，50 ms；
- `cudaStreamIsCapturing` 1,113 次，event create/destroy 约 1,060 次；
- top kernel 变成 `vectorized_elementwise_kernel` 76 ms / 415 次，而不是 compact expert kernel。

这是**末步触发的 late Triton/ATen 编译/模块加载**（不是稳定的 matmul 计算）。graph 采集同样有 1.79 s 尾步。它说明 warmup 没有覆盖最终 batch/采样形状；报告中的 decode 稳态应排除最后一步。修复方式：在 capture 前用真实最终 batch 形状做一次完整 warmup，或在启动时预编译 sampler/softmax 变体。

## 结论

1. **当前 ShareGPT 的首要瓶颈是 prefill/mixed 的 host fallback，而不是 compact decode。** prefill 的 675.9 ms（5,180 次 `w4a16_mma_kernel<32>`）和 mixed 的 1.66 s CPU idle 都来自每层读一次路由、逐 expert 调参考 kernel；D2H 为 26 次/层（加 sampler 共 27）。
2. **decode 稳态已经接近 GPU 上限**：graph 下约 110.6 ms/step，GPU 活跃 97.7%；eager 下约 128.8 ms/step，GPU 活跃 75.6%。compact decode 的 routed D2H 为 0，剩余 D2H 只有每步 1 次 sampler。
3. **尾步的编译/模块加载是一次性噪声**，但会污染端到端 wall；必须把最终 batch/采样形状纳入 warmup。
4. **下一步优先级**：
   - 把 prefill 也做成 GPU 分段/MMA32（去掉 26 次 D2H/层和 5,180 次逐 expert 调用），这是 ShareGPT 负载的最大收益；
   - 完成最终形状 warmup，消除 6.2 s / 1.8 s 尾步；
   - decode 继续调 compact tile/fan-out，并考虑 gate/up/down 融合，减少 68 ms/step 的 expert 投影；
   - graph 只覆盖 decode，prefill 仍需单独的固定容量设计。

## 产物

- [eager Nsight 报告](../profiles/sharegpt_deepseek_20261123/eager/profile.nsys-rep) · [eager 分析 JSON](../profiles/sharegpt_deepseek_20261123/eager/analysis.json)
- [graph Nsight 报告](../profiles/sharegpt_deepseek_20261123/graph/profile.nsys-rep) · [graph 分析 JSON](../profiles/sharegpt_deepseek_20261123/graph/analysis.json)
- [CUDA Event 阶段摘要（eager）](../profiles/sharegpt_deepseek_20261123/eager/events.json)
- [ShareGPT 请求 trace](../results/sharegpt_profile_20261123.jsonl)
- 分析代码：[sharegpt_deepseek_nsys.py](sharegpt_deepseek_nsys.py)、[nsys_profile.py](nsys_profile.py)
