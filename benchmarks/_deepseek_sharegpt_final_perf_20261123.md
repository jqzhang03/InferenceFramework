# DeepSeek-V2-Lite 最终 ShareGPT 性能与 Nsight 瓶颈报告

RTX 5060 Ti 16GB，WSL，PyTorch 2.8.0，TP=1，INT4 group 64，streaming load，`moe_int4_backend="compact"`，`moe_int4_cta_fanout=0`（自动 8/128/64），`max_num_seqs=32`，`max_model_len=1024`，decode CUDA Graph 已启用。

## 1. 端到端性能

ShareGPT 32 条 burst 请求，总 prompt 1,769 token，每请求输出 32 token。`benchmarks/bench.py` 使用真实 request trace，`warmup-seqs=32`。

| 指标 | 冷启动首轮 | 预热/重复轮 |
|---|---:|---:|
| wall | 5.29 s | **4.36 s** |
| output throughput | 193.7 tok/s | **234.6 tok/s** |
| TTFT avg / p50 | 1,863.6 / 1,863.6 ms | **915.1 / 915.1 ms** |
| TPOT avg / p50 | 110.4 / 110.4 ms | **111.2 / 111.2 ms** |
| E2E avg | 5,285.5 ms | **4,363.2 ms** |
| prefill | 1,257 tok / 1 step | 1,257 tok / 1 step（1,374 tok/s） |
| decode | 992 tok / 8 engine steps | 992 tok / 8 engine steps（288 tok/s） |
| decode graph | 31 replays, 0 fallback | **31 replays, 0 fallback** |
| preemptions / swaps | 0 / 0 | 0 / 0 |
| prefix cache | 512 hit tokens, 2/32 requests | 同 |

首轮 TTFT 1.86 s 与重复轮 0.915 s 的差异来自**首次 prefill 形状/前缀缓存状态的 lazy 编译**；TPOT 在两轮都约 111 ms，decode CUDA Graph 均 31 次 replay、0 次 eager fallback。

## 2. Nsight capture

新采集：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o profiles/sharegpt_deepseek_20261123/final_nsys/profile \
  python -m benchmarks.nsys_profile \
    --model ~/huggingface/DeepSeek-V2-Lite \
    --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
    --quantization int4 --int4-group-size 64 --streaming-load \
    --moe-int4-backend compact --moe-int4-cta-fanout 0 \
    --max-num-seqs 32 --max-model-len 1024 --gpu-memory-utilization .85 \
    --warmup-seqs 32 --cuda-graph \
    --gpu-event-json profiles/sharegpt_deepseek_20261123/final_nsys/events.json
```

`nsys_capture` 过滤后：

| 阶段 | steps | wall | GPU active | GPU 占比 | kernel 数 | D2H |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 689.1 ms | 663.0 ms | 96.2% | 2,191 | 1（sampler） |
| mixed | 1 | 111.5 ms | 87.8 ms | 78.7% | 2,228 | 1（sampler） |
| decode | 31 | 3,452.3 ms | 3,371.9 ms | 97.7% | 341 | 31（每步 sampler） |
| **合计** | 33 | **4,256.2 ms** | **4,122.6 ms** | **96.9%** | — | 33 |

关键点：

- **GPU 活跃 96.9%**，capture 内 idle 只有约 134 ms；不再是 host D2H/launch 瓶颈。
- prefill/mixed 的 routed expert D2H 已被消除，只剩 sampler 的 1 次/阶段；decode 每步 1 次 sampler D2H（31 次）。
- decode 有 **31 次 CUDA Graph replay、0 fallback**；单步约 111 ms wall / 109 ms GPU。
- 没有旧报告里的 6.2 s 尾步编译；`warmup-seqs=32` 把 mixed/MLA 的 1.5 s lazy 编译移出了 capture。
- 没有旧的 layer0 MLA 1.5 s（warmup 32 已覆盖 batch 32 的 shape）。

## 3. 瓶颈归因

### 3.1 prefill：TTFT 的主因，且 GPU-bound

prefill 663 ms GPU 的 top kernel：

| kernel | 时间 | 次数 | 占 prefill GPU |
|---|---:|---:|---:|
| `w4a16_compact_mma_gate_up_kernel`（融合 gate/up） | 211.6 ms | 78 | 31.9% |
| `w4a16_mma_kernel`（dense/shared） | 175.9 ms | 188 | 26.5% |
| `w4a16_compact_mma_kernel`（MMA32 chunk） | 130.5 ms | 78 | 19.7% |
| `w4a16_compact_gate_up_kernel`（≤4 行 GEMV） | 53.5 ms | 26 | 8.1% |
| `vectorized_elementwise` | 18.4 ms | 497 | 2.8% |
| `w4a16_compact_down_kernel` | 17.8 ms | 26 | 2.7% |

结论：

- prefill 已经是 **GPU 计算瓶颈**，不是 D2H/host 循环；融合 gate/up（211.6 ms）和 dense/shared（175.9 ms）是最大两块。
- compact MMA32（130.5 ms）与逐 expert reference 的 MMA 基本同级；GEMV tail 仍有 53.5 ms。
- TTFT 约 0.9 s（warm），主要就是这段 prefill；要进一步降 TTFT，需要继续优化 prefill 的 gate/up + dense/shared + MMA tile，而不是消 D2H。

### 3.2 decode：GPU-bound，Graph 已把 launch 税消掉

- 31 个 decode step：3,452 ms wall / 3,372 ms GPU（97.7%），平均 **111 ms/step**。
- 单步 graph replay 约 96–109 ms；图外只剩 sampler/softmax 的小 kernel。
- TPOT 111 ms，对应每 forward 32 token，约 **3.5 ms/token**（batch 32 聚合口径）。
- decode 没有 routed D2H；每步只有 1 次 sampler D2H。
- 因而 decode 侧的优化空间在 **compact expert 投影本身**（gate/up、MMA、down 的 tile/融合），不是同步/launch。

### 3.3 mixed：已不再是主要瓶颈

mixed 111.5 ms wall / 87.8 ms GPU；top kernel 是 GEMV 28.4 ms、dense/shared 15.4 ms、融合 gate/up 14.0 ms、down 13.4 ms。  
此前 mixed 的 1.5 s 是 layer0 MLA 首次编译，`--warmup-seqs 32` 后消失。

## 4. 当前瓶颈判断

按数据和 nsys capture：

1. **TTFT/首 token 瓶颈 = prefill GPU**（663 ms GPU，prefill 内融合 gate/up 211.6 ms + dense/shared 175.9 ms + compact MMA 130.5 ms）。
2. **吞吐/TPOT 瓶颈 = decode GPU**（97.7% 活跃，111 ms/step；compact expert 投影为主）。
3. **不是** D2H、host 打包、kernel launch 或 CUDA Graph fallback；这些在此前版本已解决。
4. 冷启动首轮 TTFT 1.86 s 是一次性 prefill shape lazy 编译；预热/重复轮 0.915 s，应作为稳态 TTFT。

## 5. 下一步优化优先级

1. **prefill gate/up**：把 SiLU 融进融合 gate/up kernel（省掉 gate/up 写回和 elementwise），并评估更大的 N tile / 权重复用；
2. **dense/shared 的 175.9 ms**：这部分仍是参考 `w4a16_mma_kernel`，需要单独的 tile/fan-out 调优，不能被 compact expert 融合覆盖；
3. **prefill GEMV tail 53.5 ms**：多数 expert 行数远大于 4，但 kernel 仍要覆盖少量小 expert；考虑 device-side 小 expert 列表或更省的启动；
4. **decode**：保持 CUDA Graph；继续调 compact gate/up/down 的 tile 和 fan-out；b8 已从融合获得约 4.8%，但 b16/b32 基本持平；
5. 预热要覆盖 **prefix-cache 命中后的 prefill 形状**（本轮冷/热 TTFT 差异即来自此），避免首轮 TTFT 被一次性编译污染。

## 6. 产物

- [新 Nsight 报告](../profiles/sharegpt_deepseek_20261123/final_nsys/profile.nsys-rep)
- [Nsight 分析 JSON](../profiles/sharegpt_deepseek_20261123/final_nsys/analysis.json)
- [CUDA Event 阶段摘要](../profiles/sharegpt_deepseek_20261123/final_nsys/events.json)
- [端到端结果（冷启动）](../profiles/sharegpt_deepseek_20261123/bench_final.json)
- [端到端结果（重复轮）](../profiles/sharegpt_deepseek_20261123/bench_final_rep2.json)
- [ShareGPT trace](../results/sharegpt_profile_20261123.jsonl)
