# DeepSeek-V2-Lite：prefix-cache 命中 prefill 的 host-sync 与预热修正

承接 [最终 ShareGPT 性能报告](_deepseek_sharegpt_final_perf_20261123.md) §5 的第 5 条：
**预热要覆盖 prefix-cache 命中后的 prefill 形状，避免冷启动 TTFT 污染**。本轮实现了这一点，并在
验证过程中定位到命中态 prefill 里一个更大的问题：`MLAAttention._assemble` 逐行读 host 标量，
每个 cache 命中 prefill 产生 **4,213 次 4B D2H + stream sync、310 ms GPU 空转**。两处都修了。

## 1. 结论

冷 Triton cache、ShareGPT 32 请求、`warmup-seqs=32`、3 个计时 batch：

| 配置 | TTFT | prefill batch 0/1/2 | batch 0 wall |
|---|---:|---|---:|
| 旧：1 遍预热 + 旧 `_assemble` | 862.8 ms | 0.863 / 0.690 / 0.715 s | 4.267 s |
| 2 遍预热 + 旧 `_assemble` | 705.0 ms | 0.705 / 0.727 / 0.729 s | 4.133 s |
| 2 遍预热 + 向量化 `_assemble` | **582.6 ms** | **0.583 / 0.515 / 0.517 s** | **4.016 s** |

- 预热修正：TTFT **−18.3%**，首个计时 batch 不再比后续 batch 慢（+25.1% → −3.1%）；
- `_assemble` 修正：TTFT 再 **−17.4%**，稳态 prefill step 0.727 → 0.515 s（**−29%**）；
- 合计 **−32.5% TTFT**。decode TPOT 不变（110.7 vs 109.8 ms，噪声内）。

## 2. 预热覆盖 prefix-cache 命中形状

### 2.1 问题

`benchmarks/bench.py` 只用真实 prompts 预热一遍，此时 prefix cache 为空；计时 batch 0 命中预热
建立的 cache（ShareGPT trace 上 512 token / 2 个请求），走 cache-shaped prefill（`_assemble` +
分页 K/V），这条路径第一次运行时的 lazy 编译落进了计时区间。实测首个计时 batch 比后续慢 25%。

### 2.2 修正

- `benchmarks/bench.py`：新增 `--warmup-passes`（默认 **2**）。第一遍建 prefix cache，第二遍命中
  它，把 cache 命中 prefill 的形状也编译掉；预热参数加 `ignore_eos=True`，保证 8-token 预热
  decode 跑满、覆盖到 decode graph 的最大 batch。JSON 的 `benchmark_config` 记录
  `warmup_seqs` / `warmup_passes`。
- `benchmarks/nsys_profile.py`：新增 `--warmup-prefix-hits`（可选）。原预热用 token+1 的移位
  prompts，不会给真实 prompts 留下可命中的 prefix，因此抓取的一直是"完整冷 prefill"；打开该
  开关后，移位预热之后再用真实 prompts 跑两遍，命中态 prefill 路径在抓取前编译好，抓取看到的
  cache 状态与 bench.py 的计时 batch 一致（prefill query token 1,257）。默认仍保持旧语义
  （完整冷 prefill），便于继续做 cold-prefill 归因。events.json 记录 `warmup_prefix_hits`。

## 3. `_assemble` 的逐行 host 读

### 3.1 证据

用 `--warmup-prefix-hits` 在冷 cache 上抓取 cache 命中态 prefill（`nsys_hits`，修正前）：

| prefill 阶段 | 数值 |
|---|---:|
| wall / GPU active / 空转 | 761.1 / 450.7 / **310.4 ms** |
| 4B D2H（copyKind=2） | **4,213 次 / 33.9 KB** |
| `cudaStreamSynchronize` | **4,213 次 / 473.6 ms CPU** |
| kernel 数 | 3,543 |

按 NVTX 归属，每个 `DeepSeekV2.Layer[i].MLA [prefill]` 区间里 156 次 —— 正好是
`_assemble` 旧实现每个 prefill 行 5 次 `int(tensor)`（`seg_start`、`s_cum` 两端、`q_off`、`qlens`），
32 行 × 26 层 ≈ 4,160 次。每次 `.item()` 都要把 stream 排空，于是 prefill 有 310 ms GPU 空转。
只有 prefix 命中/spec 行走 `_assemble`（无命中时走 dense 路径），所以完整冷 prefill 不受影响，
这也解释了为什么“命中态 prefill”曾比“完整冷 prefill”更慢（761 vs 644 ms，尽管 token 更少）。

### 3.2 修正

`_assemble` 改为纯 device 端组装（`nanovllm/layers/attention_mla.py`）：

```text
行 r 目标段 = [seg_start[r], +s[r]) 缓存段 + 其后 f[r] 行 fresh 段
缓存段源 = s_cum[r] + local        （gathered 行主序）
fresh 段源 = s_total + q_off[r] + (local - s[r])
```

用 `repeat_interleave` 造 row id、`torch.where` 合并两段源索引，再对
`cat([gk_head, k_dense])` 做一次 `index_select`；每层只剩 1 次 `.item()`（输出行数，无法从
`k_dense.shape[0]` 推——decode 兜底路径 fresh 段为空）。逐行 host 读从 5/行 降到 0。

### 3.3 修正后（`nsys_fixed`，同样冷 cache + 命中态）

| prefill 阶段 | 修正前 | 修正后 |
|---|---:|---:|
| wall | 761.1 ms | **634.6 ms** |
| GPU active | 450.7 ms | 463.9 ms |
| GPU 空转 | 310.4 ms | **170.7 ms** |
| 4B D2H | 4,213 | **82**（每层 3 次） |
| `cudaStreamSynchronize` | 4,213 次 / 473.6 ms | 不在 top API |
| kernel 数 | 3,543 | 3,273 |

剩余 170 ms 空转主要来自：每层 1 次 `_assemble` 的 `.item()`（26 次）、其它 82 个 4B D2H、
以及 nsys NUPs 的 110 个 NVTX 区间开销（bench.py 无 NVTX 时稳态 prefill 是 0.515 s）。
要再降需要把组装行数通过 `Context` 传下来（扩展 per-step 契约），本轮未做。

## 4. 正确性

- 新增 CPU 单测 `test_mla_cache_assemble_matches_row_loop`（`tests/test_mla.py`）：
  新向量化 `_assemble` 与旧逐行循环**位级相同**，覆盖
  ①cache-shaped prefill（部分行有前缀、部分行 start=0）；②decode 兜底（`qlens` 全 0、
  fresh 段为空张量）。
- `tests/test_mla.py` 5 passed；INT4/MoE/model 回归 **128 passed**（唯一失败
  `test_benchmark_comparison.py::test_scheduling_goodput_reports_per_request_ttft_target_range`
  是既有问题：`scheduling_ablation.py:447` 读 `row["prompt_tokens"]`，测试夹具没有该字段，
  与本次改动无关）。
- 端到端：bench 的 prefill token 数、decode 步数、TPOT 与修正前一致；采样输出未做逐 token
  对比，但 `_assemble` 只做数据搬运，单测锁住了位级语义。

## 5. 复现

```bash
# 冷 Triton cache
rm -rf ~/.triton/cache
# ① 旧行为（首 batch 污染）：TTFT 862.8 ms
python benchmarks/bench.py --model ~/huggingface/DeepSeek-V2-Lite \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --quantization int4 --int4-group-size 64 --streaming-load \
  --moe-int4-backend compact --max-model-len 1024 --gpu-memory-utilization .85 \
  --warmup-seqs 32 --warmup-passes 1 --repeat-batches 3 \
  --output profiles/warmup_verify/old_single_pass.json
# ② 2 遍预热：TTFT 705.0 ms
python benchmarks/bench.py ... --warmup-passes 2 \
  --output profiles/warmup_verify/new_two_pass.json
# ③ + 向量化 _assemble：TTFT 582.6 ms
python benchmarks/bench.py ... --warmup-passes 2 \
  --output profiles/warmup_verify/assemble_fix.json

# nsys 命中态抓取（修正前/后各一次；保留 .nsys-rep）
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -o <dir>/profile \
  python -m benchmarks.nsys_profile --model ~/huggingface/DeepSeek-V2-Lite \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --quantization int4 --int4-group-size 64 --streaming-load \
  --moe-int4-backend compact --max-num-seqs 32 --max-model-len 1024 \
  --gpu-memory-utilization .85 --warmup-seqs 32 --warmup-prefix-hits \
  --cuda-graph --gpu-event-json <dir>/events.json
```

## 6. 产物与清理

保留的 nsys 原始 trace（`.nsys-rep`）：

- [命中态、修正前](../profiles/warmup_verify/nsys_hits/profile.nsys-rep) ·
  [analysis](../profiles/warmup_verify/nsys_hits/analysis.json)
- [命中态、修正后](../profiles/warmup_verify/nsys_fixed/profile.nsys-rep) ·
  [analysis](../profiles/warmup_verify/nsys_fixed/analysis.json)
- 之前两轮 SiLU 对比（完整冷 prefill）：[fused](../profiles/deepseek_silu_fusion_20261123/nsys_fused/profile.nsys-rep) ·
  [split](../profiles/deepseek_silu_fusion_20261123/nsys_split/profile.nsys-rep)

bench JSON：[old_single_pass](../profiles/warmup_verify/old_single_pass.json) ·
[new_two_pass](../profiles/warmup_verify/new_two_pass.json) ·
[assemble_fix](../profiles/warmup_verify/assemble_fix.json)

实验结束后删除了非必要产物：`nsys export` 的 sqlite（可由 `.nsys-rep` 重新导出，约 180 MB）、
WSL `/tmp` 下的运行日志与 nsys 临时目录、仓库内 `__pycache__`/`.pytest_cache`、临时探针脚本，
以及一份 359 MB 的旧 Triton JIT cache 备份。`.nsys-rep`、analysis/events JSON、bench/kernel JSON
全部保留。
