# DeepSeek-V2-Lite INT4：GEMV 延迟、prefill gate/up 流量、prefill down 反量化的优化路径

2026-11-24，RTX 5060 Ti 16GB / WSL2。承接
[上一轮 decode 优化](_deepseek_decode_opt_20261123.md) 的 §5.4/§5.5：
继续处理 decode gate/up GEMV 的 latency/ILP、prefill gate/up MMA32 的重复流量，
以及 prefill down MMA32 的 int4→bf16 反量化指令。

## 0. 结论

- **成功**：
  - `dequantize` 从 8 次标量 `cvt` 改为 4 次 packed `cvt.rn.bf16x2.f32`
    （FP16 为 `cvt.rn.f16x2.f32`），覆盖全部 W4A16 kernel；
  - decode gate/up GEMV 增加 **一 K-chunk 的权重/scale 预取**，让下一组
    packed word 在 dequant + FMA 当前 tile 时已在路上；
  - gate fan-out 64 / down fan-out 128（gate 64 是真实 decode sweep 的最优）。
- **端到端（ShareGPT 32 请求，3 batch）**：
  wall **3.22 s**，output **318.0 tok/s**，TTFT **465.4 ms**，
  TPOT **88.8 ms**，E2E **3,219.1 ms**。
  相对上一轮 baseline（93.6 ms / 519.7 ms / 298.8 tok/s）：
  **TPOT −5.1%，TTFT −10.5%，throughput +6.4%，wall −6.0%**。
- **kernel 级（nsys，按未改动的 dense kernel 归一化）**：
  decode expert 投影 **73.8 → 68.5 ms/step（−7.2%）**，
  其中 GEMV **59.7 → 56.1（−6.0%）**；
  decode GPU **89.5 → 84.7 ms/step（−5.4%）**；
  prefill GPU **392.1 → 383.3 ms（−2.2%）**。
- **prefill gate/up 的重复流量没有降下来**：chunks=1/2、fanout=22/32、
  max-L1 carveout 五种调度都做了 A/B，全部比 baseline 慢或持平；
  DRAM read 仍在 **1.34 GB** 左右。激活 x 被 44 个 channel CTA 重复读是主因，
  仅靠 grid 调度拿不掉，需要 shared-x / 多 channel-tile 的 tile 级重构。
  **最终保留 baseline 的 chunks=4 / 44 channel CTAs**，未把失败的调度合入。
- **prefill down**：packed dequant 保留；nsys 归一化
  **75.4 → 73.4 ms（−2.7%）**，NCU 单 kernel 因 run 间时钟波动有高有低。
- **正确性**：`tests/test_moe_int4_compact.py` + `test_cuda_int4.py` +
  `test_mla.py` **78 passed**；packed dequant 与 GEMV 预取都不改变每个输出的
  K 累加顺序，compact 与 per-expert loop 仍位级一致。

## 1. 改动

### 1.1 packed int4→bf16/fp16 反量化

[cuda_int4.cu](../nanovllm/layers/csrc/cuda_int4.cu#L36-L58)：

```cpp
for (int i = 0; i < 4; ++i) {
    low  = int(packed & 15) - 8;
    high = int((packed >> 4) & 15) - 8;
    packed >>= 8;
    *(__nv_bfloat162*)(elements + 2*i) = __floats2bfloat162_rn(low*scale, high*scale);
}
```

- 每个 32-bit packed word 的 8 个 int4 仍然按 low→high 顺序落到
  `elements[0..7]`，舍入仍是 `cvt.rn`，因此和旧的 8 次
  `from_float` 逐位一致（78 个 compact 测试验证）。
- 影响所有 W4A16 路径：compact GEMV/MMA、dense `w4a16_mma`。

### 1.2 GEMV 的 one-chunk 权重预取

[cuda_int4.cu](../nanovllm/layers/csrc/cuda_int4.cu#L95-L150)：
`gemv_columns` 把 `load_weights` 拆成 `load_packed` / `load_scale`，在循环里：

```text
preload  packed/scale for kk
for each kk:
    issue next packed/scale for kk+stride   # 与当前 dequant/FMA 重叠
    dequantize current, apply ROWS rows
    rotate preload
```

- 只预取 4B 的 packed word 和 2B scale，不预取 dequantized `uint4`，
  寄存器只增加约 4 个；NCU 里 gate kernel 的 registers 80 → 65，
  occupancy 49.6% → 59.6%。
- 每个输出的 K 累加顺序仍然是 `lane*8 + n*64` 的 even/odd 链，位级不变。

### 1.3 fan-out 选择

[moe.py](../nanovllm/layers/moe.py#L292-L299)：

```text
T ≤ 8       沿用原 fan-out
9 ≤ T ≤ 64  gate/up = 64，down = 128
T > 64      gate/up = 8， down = 8（prefill 小 expert 尾巴）
```

gate 64 来自真实 decode sweep（64 的 TPOT 87.0 / 128 的 87.5 / 8 的 95.7，
均为 3-batch），microbench 也显示 64 略优于 128。

### 1.4 没有合入的 prefill gate/up 调度

见 §5，所有尝试都保留在实验记录里，生产代码保持 `chunks=4`、44 个
channel CTA。

## 2. microbench

[benchmarks/int4_decode_microbench.py](../benchmarks/int4_decode_microbench.py)，
64 experts / 192 routes / gate-up 1408×2048 / down 2048×1408：

| 项目 | 优化前 | 本轮 | 变化 |
|---|---:|---:|---:|
| GEMV down/gate 8/8 | 2.682 ms | 2.581 ms | −3.8% |
| GEMV 128/64（生产） | 2.506 ms | **2.233 ms** | **−10.9%** |
| GEMV 128/128 | 2.541 ms | 2.281 ms | −10.2% |
| gate 5–16 band | 0.140 ms | 0.106 ms | −24% |
| down 5–16 band | 0.121 ms | 0.094 ms | −22% |
| 单层 expert 合计（8/8 + 三 band） | 2.943 ms | 2.781 ms | −5.5% |
| 单层 expert 合计（128/64 + band） | 2.767 ms | **2.433 ms** | **−12.1%** |

`band_gate_bitwise` / `band_down_bitwise` 都为 `true`。

## 3. NCU：decode

新报告：[decode rep](../profiles/deepseek_opt2_20261124/ncu/decode/profile.ncu-repz) /
[CSV](../profiles/deepseek_opt2_20261124/ncu/decode/report.csv)。
baseline 是 [上一轮 decode CSV](../profiles/deepseek_rerun_20261123/ncu/decode/report.csv)。

| kernel | baseline | 本轮 | 变化 | 本轮关键指标 |
|---|---:|---:|---:|---|
| GEMV gate/up `<1,4>` | 1.289 ms | 1.300 ms | +0.8% | DRAM 31.2%，SM 40.9%，occ 59.6%，regs 65，waves 16.1 |
| GEMV down `<1,4>` | 0.570 ms | **0.516 ms** | **−9.5%** | DRAM 42.1%，SM 69.8%，occ 65.2%，waves 28.4 |
| band gate/up | 0.243+0.105 | 0.376 ms | +8%（单次） | DRAM 33.8%，tensor 9.4%，occ 70.9% |
| band down | 0.125+0.111 | 0.153 ms | −35%（单次） | DRAM 41.9%，tensor 11.9%，occ 59.6% |
| MMA32（17+） | 0.005 ms | 0.005 ms | 空 band | — |

NCU 单次的 gate 时间与 baseline 基本持平，但 **occupancy 从 49.6% 升到 59.6%、
SM throughput 从 33.7% 升到 40.9%**；这来自 80→65 registers 和预取。
band kernel 的单次 NCU 有 ±10% 波动，整步收益以 nsys 为准。

## 4. NCU：prefill

新报告：[prefill rep](../profiles/deepseek_opt2_20261124/ncu/prefill/profile.ncu-repz) /
[CSV](../profiles/deepseek_opt2_20261124/ncu/prefill/report.csv)。
baseline：[上一轮 prefill CSV](../profiles/deepseek_rerun_20261123/ncu/prefill/report.csv)。

| kernel | baseline | 本轮 | DRAM read | 说明 |
|---|---:|---:|---:|---|
| gate/up MMA32 | 4.656 ms | 6.214 / 5.651 ms | **1.344 GB** | 两次层采样；流量未变，NCU 本轮时钟偏慢 |
| down MMA32 | 3.308 ms | 2.505 / 3.557 ms | 0.123 / 0.160 GB | 两次层采样，packed dequant 后混合 |
| dense `w4a16_mma` | 1.606 ms | 未抓（不变） | — | 对照 |

NCU 的绝对时间不能跨 run 直接比：本轮 nsys 里未改的 dense kernel 也从
9.45 → 10.33 ms/step（×1.093）。归一化后：

- prefill gate/up：135.4 → **131.2 ms（−3.1%）**；
- prefill down：75.4 → **73.4 ms（−2.7%）**；
- prefill GPU：392.1 → **383.3 ms（−2.2%）**。

**gate/up 的 1.344 GB DRAM read 没有下降**：packed dequant 只减少了指令，
没有改变激活 x 被 44 个 channel CTA 重复读的流量结构。

## 5. 优化路径：prefill gate/up 的失败尝试

目标是把 1.338 GB 的 DRAM read 降下来。尝试了五种调度，全部被数据否决：

| 方案 | 实测 | DRAM read | 结论 |
|---|---:|---:|---|
| chunks=1（1 CTA 走完 4 chunks） | nsys gate **190.8 ms** vs 135.4 | — | 并行度不足，−41% 性能，否决 |
| chunks=2 | NCU 5.47 / 4.73 ms vs 4.66 | 1.365 GB | 流量不变且更慢，否决 |
| channel fanout=22 | NCU 5.25 / 4.68 ms | **0.995 GB（−26%）** | 流量降了但时间不降，否决 |
| channel fanout=32 | NCU 4.63 / 5.51 ms | 1.127 GB | 平均更慢，否决 |
| `cudaSharedmemCarveoutMaxL1` | **19.2 / 20.1 ms**，occ 8.3% | 1.40 GB | shared carveout 配置反噬，否决 |

结论：

- 1.338 GB 主要是 **activation x 的重复读**（7542 行 × 2048 × 2 B × 44 个
  channel CTA ≈ 1.36 GB），不是 gate/up 权重；权重 184 MB 基本被 L2 吸收。
- 只改 grid/CTA 数量，会在“重复读下降”和“并行度/CTA 串行度”之间互相抵消；
  fanout=22 已经把 read 降 26%，但 kernel 仍不变快。
- 真正可行的下一步是 **把一个 32-row 的 x tile 缓存到 shared，并让 CTA 处理
  多个 channel tile**（或 N_TILE=2 的 K-outer tile），这需要重写
  `mma_gate_up_silu_wide_tile` 的循环结构；本轮没有把未验证的大改合入生产。

## 6. nsys：整步前后（归一化）

新 nsys：[profile.nsys-rep](../profiles/deepseek_opt2_20261124/nsys/profile.nsys-rep) /
[analysis.json](../profiles/deepseek_opt2_20261124/nsys/analysis.json)。
baseline：[analysis.json](../profiles/deepseek_rerun_20261123/nsys/analysis.json)。

| 阶段 | baseline | 本轮 raw | 归一化 |
|---|---:|---:|---:|
| capture wall | 3,459.5 ms | 3,702.0 ms | host 偏慢 |
| capture GPU | 3,244.2 ms | 3,364.8 ms | — |
| prefill GPU | 392.1 ms | 416.7 ms | 383.3（−2.2%） |
| decode GPU/step | 89.50 ms | 92.56 ms | **84.68（−5.4%）** |

decode 明细（除以未改 dense 的 10.33/9.45=1.0931）：

| kernel | baseline | 本轮 raw | 归一化 | 变化 |
|---|---:|---:|---:|---:|
| GEMV gate/up | 42.24 | 44.48 | 40.69 | −3.7% |
| GEMV down | 17.43 | 16.82 | 15.39 | −11.7% |
| band gate | 9.33 | 9.51 | 8.70 | −6.7% |
| band down | 4.79 | 4.07 | 3.72 | −22.3% |
| **expert 合计** | **73.79** | 74.88 | **68.50** | **−7.2%** |
| dense（对照） | 9.45 | 10.33 | 9.45 | 0% |
| **decode GPU/step** | **89.50** | 92.56 | **84.68** | **−5.4%** |

（归一化 factor 为未改动的 dense kernel：10.33/9.45=1.0931；表中所有
“归一化”列均除以该 factor。raw 的 74.88 与 baseline 73.79 接近，是因为本轮
run-wide clock 偏慢；归一化后才是可比的 kernel 收益。）

prefill 明细（除以 dense 的 74.7/68.7=1.0873）：

| kernel | baseline | 本轮 raw | 归一化 | 变化 |
|---|---:|---:|---:|---:|
| gate/up MMA32 | 135.4 | 142.7 | 131.2 | −3.1% |
| down MMA32 | 75.4 | 79.8 | 73.4 | −2.7% |
| dense（对照） | 68.7 | 74.7 | 68.7 | 0% |

## 7. 端到端

| 指标 | baseline | 本轮 |
|---|---:|---:|
| wall | 3.427 s | **3.220 s** |
| output throughput | 298.8 tok/s | **318.0** |
| TTFT p50 | 519.7 ms | **465.4** |
| TPOT p50 | 93.6 ms | **88.8** |
| E2E p50 | 3,421.6 ms | **3,219.1** |
| 相对变化 | — | wall −6.0%，throughput +6.4%，TTFT −10.5%，TPOT −5.1%，E2E −5.9% |

三个 batch 的 wall 为 3.22 / 3.15 / 3.20 s；decode 31 forwards，
graph 31 replay / 0 fallback，0 preemption / 0 swap。

## 8. 正确性与限制

- `pytest tests/test_moe_int4_compact.py tests/test_cuda_int4.py tests/test_mla.py -q`：
  **78 passed**（[pytest.log](../profiles/deepseek_opt2_20261124/pytest.log)）。
- packed dequant、GEMV 预取、band 合并、fan-out 改动均通过
  compact vs per-expert loop 的 `rtol=0, atol=0`。
- microbench 的新旧 band 输出 `torch.equal=True`。
- 限制：
  - prefill gate/up 的重复流量没有解决，仅通过 packed dequant 得到 ~3%；
  - NCU 单次时间受 run 间 clock/cache 影响，本轮 gate NCU 6.2 ms 与 nsys
    归一化 131 ms/26 层不一致，报告以 nsys 归一化和 end-to-end 为准；
  - 失败调度没有合入生产，相关 NCU 原始 rep 已在清理阶段删除，只保留
    本报告中的数值。

## 9. 产物与复现

- 最终 bench：[bench_final.json](../profiles/deepseek_opt2_20261124/bench_final.json) ·
  [log](../profiles/deepseek_opt2_20261124/bench_final.log)
- nsys：[profile.nsys-rep](../profiles/deepseek_opt2_20261124/nsys/profile.nsys-rep) ·
  [analysis.json](../profiles/deepseek_opt2_20261124/nsys/analysis.json) ·
  [events.json](../profiles/deepseek_opt2_20261124/nsys/events.json)
- NCU：[prefill rep](../profiles/deepseek_opt2_20261124/ncu/prefill/profile.ncu-repz) ·
  [prefill CSV](../profiles/deepseek_opt2_20261124/ncu/prefill/report.csv) ·
  [decode rep](../profiles/deepseek_opt2_20261124/ncu/decode/profile.ncu-repz) ·
  [decode CSV](../profiles/deepseek_opt2_20261124/ncu/decode/report.csv)
- microbench：[microbench.json](../profiles/deepseek_opt2_20261124/microbench/microbench.json) ·
  [log](../profiles/deepseek_opt2_20261124/microbench.log)
- 代码：[cuda_int4.cu](../nanovllm/layers/csrc/cuda_int4.cu)（packed dequant 36–58、
  GEMV 预取 95–150）、[moe.py](../nanovllm/layers/moe.py#L292-L299)、
  [int4_decode_microbench.py](../benchmarks/int4_decode_microbench.py)

复现：

```bash
cd /mnt/d/Project/nano-vllm
export PATH="/usr/local/cuda-12.8/bin:$HOME/miniconda3/envs/nano-vllm/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export PYTHONPATH=/mnt/d/Project/nano-vllm

python -m pytest tests/test_moe_int4_compact.py tests/test_cuda_int4.py -q
python -m benchmarks.int4_decode_microbench \
  --root profiles/deepseek_opt2_20261124/microbench2

python benchmarks/bench.py --model ~/huggingface/DeepSeek-V2-Lite \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --quantization int4 --int4-group-size 64 --streaming-load \
  --moe-int4-backend compact --max-model-len 1024 \
  --gpu-memory-utilization .85 --warmup-seqs 32 --warmup-passes 2 \
  --repeat-batches 3 \
  --output profiles/deepseek_opt2_20261124/bench_final.json

nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --force-overwrite=true --stats=false \
  -o profiles/deepseek_opt2_20261124/nsys/profile \
  python -m benchmarks.nsys_profile \
  --model ~/huggingface/DeepSeek-V2-Lite \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --quantization int4 --int4-group-size 64 --streaming-load \
  --moe-int4-backend compact --moe-int4-cta-fanout 0 \
  --max-num-seqs 32 --max-model-len 1024 --gpu-memory-utilization .85 \
  --warmup-seqs 32 --warmup-prefix-hits --cuda-graph \
  --gpu-event-json profiles/deepseek_opt2_20261124/nsys/events.json
nsys export --type sqlite --force-overwrite=true \
  --output profiles/deepseek_opt2_20261124/nsys/profile.sqlite \
  profiles/deepseek_opt2_20261124/nsys/profile.nsys-rep
python -m benchmarks.sharegpt_deepseek_nsys \
  profiles/deepseek_opt2_20261124/nsys/profile.sqlite \
  --events profiles/deepseek_opt2_20261124/nsys/events.json \
  --output profiles/deepseek_opt2_20261124/nsys/analysis.json

# NCU prefill gate/up + down
NCU=/usr/local/NVIDIA-Nsight-Compute-2026.3/ncu
"$NCU" --nvtx --nvtx-include ncu_prefill --kernel-name-base function \
  -k 'regex:.*w4a16_(compact_mma_gate_up_silu_kernel|compact_mma_kernel).*' -c 4 \
  --set detailed --cache-control all --force-overwrite \
  --export profiles/deepseek_opt2_20261124/ncu/prefill/profile \
  python -m benchmarks.dsv2_operator_ncu --phase prefill \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --output profiles/deepseek_opt2_20261124/ncu/prefill/harness.json

# NCU decode
"$NCU" --nvtx --nvtx-include ncu_decode --kernel-name-base function \
  -k 'regex:.*w4a16_(compact_gate_up_kernel|compact_down_kernel|compact_mma_gate_up_band_kernel|compact_mma_band_kernel|compact_mma_gate_up_silu_kernel|compact_mma_kernel).*' -c 8 \
  --set detailed --cache-control all --force-overwrite \
  --export profiles/deepseek_opt2_20261124/ncu/decode/profile \
  python -m benchmarks.dsv2_operator_ncu --phase decode \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --output profiles/deepseek_opt2_20261124/ncu/decode/harness.json
```

## 10. 清理

已删除：

- 失败的 prefill 调度 NCU 目录（`prefill_fan22` / `prefill_fan32` /
  `prefill_carveout`，约 2.4 MB）和中间 bench；
- nsys sqlite 导出（约 65 MB，可由 `.nsys-rep` 重新导出）；
- 临时 `_run_opt2_*.sh`、`__pycache__`、`.pytest_cache`；
- WSL `~/.nv/ComputeCache`（441 MB）、`pip_tmp`。

保留：最终的 `.nsys-rep`、`.ncu-repz`、CSV、bench/microbench/pytest 日志，
合计约 **20.4 MB**。
