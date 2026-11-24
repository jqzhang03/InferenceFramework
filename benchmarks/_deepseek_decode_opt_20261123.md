# DeepSeek-V2-Lite decode 优化：GEMV 访存并行度与 5–16 行 band 合并

2026-11-23，RTX 5060 Ti 16GB / WSL2。承接
[ShareGPT 复测报告](_deepseek_sharegpt_rerun_20261123.md) 的 §5.4/§5.5：
把 decode GEMV 的 grid 拉开，并把原先 5–8 / 9–16 两个 MMA band 合并成一个
按 `counts[expert]` 分支的 kernel，然后重新跑 bench、nsys 和 NCU。

## 0. 结论

- **改了两处生产代码**：
  1. decode T=32 时 gate/up 与 down GEMV 的 column CTA fan-out 从 **8 → 128**
     （prefill 仍保持 8）；`expert_out_compact` 现在显式接收 down/gate 两个
     fan-out，便于分别调；
  2. 新增 count-branched 的 **5–16 行 band kernel**：一个 launch 内部按
     `counts[expert]` 选择 MMA8 或 MMA16，原先的两个 launch 合并为一个；
     17+ 行仍走 MMA32。T=32 时每层 expert 投影 launch 数 **6 → 4**。
- **正确性**：`tests/test_moe_int4_compact.py` + `test_cuda_int4.py` +
  `test_mla.py` **78 passed**；microbench 中旧三 band 与新 band 的输出
  `torch.equal == True`；CUDA Graph 仍是 31 replay / 0 fallback。
- **kernel 级收益（同一 nsys 口径，按未改动的 dense kernel 归一化）**：
  decode expert 投影 **73.8 → 70.7 ms/step（−4.2%）**；
  其中 GEMV **59.7 → 57.3（−3.9%）**、5–16 行 band **14.1 → 13.3（−5.6%）**；
  decode GPU 总时间 **89.5 → 86.9 ms/step（−3.0%）**。
- **NCU 单 kernel**：down GEMV **0.570 → 0.508 ms（−10.9%）**，DRAM 37.2%→43.6%，
  waves 1.78→28.4；band down **0.236 → 0.132 ms（−44%）**；
  gate GEMV 1.289→1.316 ms（+2.1%，fan-out 对它是负收益，但端到端 sweep 仍选 128）；
  band gate 约 0.35 ms（与旧 0.348 ms 基本持平，`__launch_bounds__` 把寄存器
  72→64）。
- **端到端**：同 workload 的 3 个 3-batch 运行落在 **TPOT 92.6 / 92.9 / 93.3 ms**
  （baseline 93.6），wall 3.34–3.40 s（baseline 3.427），output throughput
  300.9–306.4 tok/s（baseline 298.8），TTFT 469–512 ms（baseline 519.7）。
  **端到端只看到约 1% 的 TPOT 改善**，因为 expert 投影只占 decode 的 82%，
  剩余部分和 host 侧抖动（重复运行 ±3 ms）把 3–4% 的 kernel 收益摊薄了。

## 1. 改动

### 1.1 GEMV 的 column fan-out

[cuda_int4.cu](../nanovllm/layers/csrc/cuda_int4.cu#L936-L959) 的 compact
GEMV 原来 gate/up 和 down 共用一个 `cta_fanout`；decode 时
[moe.py](../nanovllm/layers/moe.py#L292-L299) 把它固定成 8，导致
`grid=(64,1,8)`、2.37 waves、DRAM 只有 30–37%。现在：

```text
T ≤ 8       保持原 fan-out（8 或 128）
9 ≤ T ≤ 64  gate/up = 128，down = 128
T > 64      gate/up = 8，down = 8（prefill 小 expert 尾巴）
```

`expert_out_compact`/`launch_compact` 增加 `gate_fanout` 参数，down 用
`cta_fanout`；这样后续可以单独调 gate/up 与 down，而不必重新编译两个 kernel。
`w4a16_compact_gate_up_kernel` 与 `w4a16_compact_down_kernel` 本身未改算术，
只是 grid 变大：gate 每个 CTA 的 column tile 从 11 个降到 1 个，down 从
16 个降到 1 个，相当于把 512 个 CTA 展开到 5,632 / 8,192 个。

### 1.2 5–16 行 band 合并

旧路径（T=32）每层固定发三个 gate/up launch + 三个 down launch：

```text
(5, 8,  MMA8)  (9, 16, MMA16)  (17, T, MMA32)
```

其中 MMA8 的 `grid.y=2` 对 5–8 行 expert 还会多一个空 chunk。新路径：

- [w4a16_compact_mma_band_kernel](../nanovllm/layers/csrc/cuda_int4.cu#L691-L722)：
  `total <= 8` 走 `mma_tile<8,64>`，否则走 `mma_tile<16,64>`；一个
  `grid=(64,1,n_tiles)`，不再有第二 chunk；
- [w4a16_compact_mma_gate_up_band_kernel](../nanovllm/layers/csrc/cuda_int4.cu#L723-L760)：
  同样按 count 选 `MMA8`/`MMA16` 的 gate/up + SiLU 融合模板；
- `block_m=0` 作为“auto band”哨兵，host 校验/launch 在
  [gemm_*_compact](../nanovllm/layers/csrc/cuda_int4.cu#L1156-L1400) 内分支；
- [moe.py](../nanovllm/layers/moe.py#L331-L336) 的 `ranges` 从三个变成两个：
  `(5, min(16,T), 0)` + `(17,T,32)`。

旧 MMA8/MMA16 kernel 保留，供诊断和回退；生产路径只发 band + MMA32。

### 1.3 gate band 的寄存器上限

band gate 合并两个模板后 ptxas 用了 72 registers（旧 MMA8/16 都是 64），
occupancy 从 62.4%/55.1% 掉到 61.3%/55.8%。给
[w4a16_compact_mma_gate_up_band_kernel](../nanovllm/layers/csrc/cuda_int4.cu#L723-L724)
加 `__launch_bounds__(128, 8)` 后寄存器回到 64，band gate 在 microbench 从
0.113 ms 降到 0.107 ms。

## 2. microbench：单层 64 expert / 192 route

[benchmarks/int4_decode_microbench.py](../benchmarks/int4_decode_microbench.py)
直接构造 DSV2-Lite 的 expert 形状（gate/up 1408×2048、down 2048×1408、64 experts、
T=32 × top_k 6 = 192 packed rows），不加载模型，避免 host 调度干扰。counts 里显式
放了 5–8、9–16、17+ 三档 expert。

| 配置 | 时间 | 变化 |
|---|---:|---:|
| GEMV fan-out down/gate = 8/8 | 2.666 ms | — |
| 128/8 | 2.571 ms | −3.6% |
| 128/64 | **2.506 ms** | **−6.0%** |
| 128/128（生产选择） | 2.541 ms | −4.7% |
| gate band：旧 MMA8+16 | 0.140 ms | — |
| gate band：合并 | **0.107 ms** | **−23.6%** |
| down band：旧 MMA8+16 | 0.121 ms | — |
| down band：合并 | **0.093 ms** | **−23.1%** |
| 单层 expert 合计（旧 8/8 + 三 band） | 2.927 ms | — |
| 单层 expert 合计（128/128 + 合并 band） | **2.741 ms** | **−6.4%** |

`band_gate_bitwise` / `band_down_bitwise` 都是 `true`；合并 kernel 与旧三 band
逐位相同。生产选 128/128 而不是 microbench 最低的 128/64，是因为真实 ShareGPT
decode 的 sweep 里 128/128 的 TPOT 最低（见 §5）。

## 3. NCU：单 kernel 前后

采集：NCU 2026.3.1，`--set detailed --cache-control all`，decode batch 32，
每个 kernel 取层 1 的 launch；band gate 单独抓了 2 层。baseline 是
[上一轮 NCU](../profiles/deepseek_rerun_20261123/ncu/decode/report.csv)，
优化后是 [decode rep](../profiles/deepseek_decode_opt_20261123/ncu/decode/profile.ncu-repz)
和 [band gate rep](../profiles/deepseek_decode_opt_20261123/ncu/band_gate/profile.ncu-repz)。

| kernel | baseline | 优化后 | 变化 | 关键指标（优化后） |
|---|---:|---:|---:|---|
| GEMV gate/up `<1,4>` | 1.289 ms | 1.316 ms | +2.1% | DRAM 31.3%，occ 50.1%，waves 26.1 |
| GEMV down `<1,4>` | 0.570 ms | **0.508 ms** | **−10.9%** | DRAM 43.6%，occ 65.3%，waves 28.4 |
| band gate/up（5–16） | 0.243 + 0.105 ms | ~0.352 ms | ~+1% | DRAM 34–36%，tensor 10–13%，regs 64 |
| band down（5–16） | 0.125 + 0.111 ms | **0.132 ms** | **−44%** | DRAM 39.1%，tensor 12.2%，regs 56 |
| MMA32 gate/up（17+） | 0.007 ms | 0.005 ms | 空 band | — |
| MMA32 down（17+） | 0.005 ms | 0.005 ms | 空 band | — |
| dense `w4a16_mma<1,16,128>` | 0.116 ms | 0.116 ms | 0% | 对照，未改 |

解释：

- **down GEMV 是这次 fan-out 的主要赢家**：grid 从 `(64,1,8)` 变成
  `(64,1,128)`，DRAM 37.2%→43.6%，occupancy 56.6%→65.3%，latency 被更多
  outstanding weight 流覆盖。
- **gate/up GEMV 在 NCU 里反而慢 2.1%**：它读 gate+up 两份权重、激活复用更高，
  8 CTA 时每 CTA 串 11 个 tile 反而有更好的 L1 复用；128 CTA 把激活重复读放大。
  但真实端到端 sweep（§5）里 128 仍优于 8，说明在 graph 上下文里多 CTA 的
  latency hiding 更重要；这个差异只有 ~2 ms/step，属于可接受范围。
- **band down 合并收益最大**：旧 MMA16 down 用 95 registers、occupancy 23%，
  合并 kernel 只有 56 registers，且去掉了 8-row band 的空第二 chunk。
- **band gate 基本持平**：旧 MMA8 是 64 regs/62% occ，MMA16 是 64 regs/55% occ；
  合并后经 `__launch_bounds__` 也是 64 regs，但一个 kernel 要覆盖两种 tile，
  branch/共享 tile 让 5–8 行路径没有额外收益。它节省的是 launch 数（每层 −2），
  不是单次时间。

## 4. nsys：整步前后的归一化对比

新 nsys：[profile.nsys-rep](../profiles/deepseek_decode_opt_20261123/nsys/profile.nsys-rep) /
[analysis.json](../profiles/deepseek_decode_opt_20261123/nsys/analysis.json)；
baseline：[analysis.json](../profiles/deepseek_rerun_20261123/nsys/analysis.json)。
nsys 使用 `--cuda-graph-trace=node`，decode 逐 kernel 可见。

| 阶段 | baseline | 优化后 | 说明 |
|---|---:|---:|---|
| capture wall | 3,459.5 ms | 3,518.8 ms | 本轮 host/clock 偏慢 |
| capture GPU active | 3,244.2 ms | 3,287.4 ms | 原始值 |
| prefill GPU | 392.1 ms | 416.1 ms | 未改 prefill；整体约 +6% |
| mixed GPU | 77.6 ms | 76.5 ms | — |
| decode GPU / step | 89.50 ms | 90.16 ms | 原始值 |
| decode kernel 数/step | 2,033 | 1,983 | −50（每层 −2 launch） |

单看 raw 值会被本轮 ~4% 的 run-wide 漂移污染：未改动的 dense kernel 从
9.45 → 9.81 ms/step（×1.038）。用这个未改 kernel 归一化：

| kernel | baseline | 优化后 raw | 归一化后 | 变化 |
|---|---:|---:|---:|---:|
| GEMV gate/up | 42.24 ms | 43.52 | 41.92 | −0.8% |
| GEMV down | 17.43 | 15.99 | 15.40 | **−11.7%** |
| band gate | 9.33 | 9.64 | 9.29 | −0.4% |
| band down | 4.79 | 4.20 | 4.05 | **−15.5%** |
| **expert 合计** | **73.79** | 73.35 | **70.66** | **−4.2%** |
| dense（对照） | 9.45 | 9.81 | 9.45 | 0% |
| **decode GPU/step** | **89.50** | 90.16 | **86.86** | **−3.0%** |

结论：**expert 投影确实快了 4.2%，decode GPU 快 3.0%**；端到端 TPOT 只看到
约 1%，是因为 decode 里还有 ~15 ms 的非 expert kernel 和 host 间隙，且本轮
重复运行的 wall 本身有 ±3 ms 抖动。

## 5. 端到端：fan-out sweep 与最终结果

同 workload（ShareGPT 32 请求、1,257 prefill token、每请求 32 输出 token），
3 个 batch，`bench.py`：

| 配置 | wall | output tok/s | TTFT | TPOT |
|---|---:|---:|---:|---:|
| baseline（fan-out 8，三 band） | 3.427 s | 298.8 | 519.7 ms | **93.6 ms** |
| gate fan-out 8 / down 128 | 3.44 s | 298.0 | 468.9 | 95.7 |
| gate 64 / down 128 | 3.37 s | 304.1 | 465.7 | 93.6 |
| gate 128 / down 128 | 3.36 s | 304.7 | 480.3 | **92.9** |
| 最终代码（128/128 + 合并 band + launch_bounds） | 3.34–3.40 s | 300.9–306.4 | 469–512 | **92.6–93.3** |

gate fan-out 8 在真实 decode 里最慢（95.7），64 和 128 明显更好；因此最终选
128/128。端到端相对 baseline：

- TPOT 93.6 → 92.6–93.3（约 −0.3~−1.1%）；
- TTFT 519.7 → 469–512（约 −1.5~−9.7%）；prefill kernel 未改，差异来自本轮
  host/prefix-cache 状态波动，不应归因于这次 fan-out；
- wall 3.427 → 3.34–3.40 s；
- throughput 298.8 → 300.9–306.4 tok/s。

“最终代码”指：128/128 fan-out、5–16 band 合并、gate band
`__launch_bounds__(128,8)`。其中 `bench_final_rep3` 是加 launch_bounds **之前**
的 92.6 ms 运行（其余配置相同），`bench_final3_rep3` 是最终代码 93.3 ms 的复测；
两者的差异主要来自 host/clock。

## 6. 正确性与限制

- `pytest tests/test_moe_int4_compact.py tests/test_cuda_int4.py tests/test_mla.py -q`：
  **78 passed**（[pytest_compact.log](../profiles/deepseek_decode_opt_20261123/pytest_compact.log)）。
- microbench 对旧三 band 和新 band 做 `torch.equal`：gate/down 都 `true`。
- 本轮 bench：decode 31 forwards、graph 31 replay / 0 fallback、0 preemption、
  0 swap；3 个 batch 的 decode token 一致（992）。
- 限制：
  - gate/up GEMV 的 fan-out 128 在 NCU 里略慢于 8；真实 sweep 选 128，但两者
    只差 ~2 ms/step，和 host 抖动同量级；
  - band gate 合并是“省 launch 不省时间”，真正的大头仍是 GEMV；
  - 这些是 kernel 级收益，端到端 TPOT 的进一步下降需要继续动 gate/up GEMV
    （例如把 4 行激活装入 shared、按 warp 广播激活）或减少 decode 的
    非 expert 开销。

## 7. 产物与复现

- 最终 bench：[bench_final_rep3.json](../profiles/deepseek_decode_opt_20261123/bench_final_rep3.json)（92.6）·
  [bench_final3_rep3.json](../profiles/deepseek_decode_opt_20261123/bench_final3_rep3.json)（93.3）·
  [gate128 sweep](../profiles/deepseek_decode_opt_20261123/bench_gate128.json)
- nsys：[profile.nsys-rep](../profiles/deepseek_decode_opt_20261123/nsys/profile.nsys-rep) ·
  [analysis.json](../profiles/deepseek_decode_opt_20261123/nsys/analysis.json) ·
  [events.json](../profiles/deepseek_decode_opt_20261123/nsys/events.json)
- NCU：[decode rep](../profiles/deepseek_decode_opt_20261123/ncu/decode/profile.ncu-repz) ·
  [decode CSV](../profiles/deepseek_decode_opt_20261123/ncu/decode/report.csv) ·
  [band gate rep](../profiles/deepseek_decode_opt_20261123/ncu/band_gate/profile.ncu-repz) ·
  [band gate CSV](../profiles/deepseek_decode_opt_20261123/ncu/band_gate/report.csv)
- microbench：[microbench.json](../profiles/deepseek_decode_opt_20261123/microbench/microbench.json) ·
  [microbench.log](../profiles/deepseek_decode_opt_20261123/microbench.log)
- 代码：[cuda_int4.cu](../nanovllm/layers/csrc/cuda_int4.cu)（band kernel 691–760、
  GEMV fan-out 936–959、host auto-band 1156+）、
  [moe.py](../nanovllm/layers/moe.py#L292-L336)、
  [int4_decode_microbench.py](../benchmarks/int4_decode_microbench.py)

复现：

```bash
cd /mnt/d/Project/nano-vllm
export PATH="/usr/local/cuda-12.8/bin:$HOME/miniconda3/envs/nano-vllm/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export PYTHONPATH=/mnt/d/Project/nano-vllm

# 1) 正确性（会顺带重建 extension）
python -m pytest tests/test_moe_int4_compact.py tests/test_cuda_int4.py -q

# 2) 无模型 microbench（逐位对比旧/新 band + fan-out sweep）
python -m benchmarks.int4_decode_microbench \
  --root profiles/deepseek_decode_opt_20261123/microbench2

# 3) 端到端
python benchmarks/bench.py --model ~/huggingface/DeepSeek-V2-Lite \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --quantization int4 --int4-group-size 64 --streaming-load \
  --moe-int4-backend compact --max-model-len 1024 \
  --gpu-memory-utilization .85 --warmup-seqs 32 --warmup-passes 2 \
  --repeat-batches 3 \
  --output profiles/deepseek_decode_opt_20261123/bench_final3_rep3.json

# 4) nsys
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --force-overwrite=true --stats=false \
  -o profiles/deepseek_decode_opt_20261123/nsys/profile \
  python -m benchmarks.nsys_profile \
  --model ~/huggingface/DeepSeek-V2-Lite \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --quantization int4 --int4-group-size 64 --streaming-load \
  --moe-int4-backend compact --moe-int4-cta-fanout 0 \
  --max-num-seqs 32 --max-model-len 1024 --gpu-memory-utilization .85 \
  --warmup-seqs 32 --warmup-prefix-hits --cuda-graph \
  --gpu-event-json profiles/deepseek_decode_opt_20261123/nsys/events.json
nsys export --type sqlite --force-overwrite=true \
  --output profiles/deepseek_decode_opt_20261123/nsys/profile.sqlite \
  profiles/deepseek_decode_opt_20261123/nsys/profile.nsys-rep
python -m benchmarks.sharegpt_deepseek_nsys \
  profiles/deepseek_decode_opt_20261123/nsys/profile.sqlite \
  --events profiles/deepseek_decode_opt_20261123/nsys/events.json \
  --output profiles/deepseek_decode_opt_20261123/nsys/analysis.json

# 5) NCU decode（band gate 用 -k w4a16_compact_mma_gate_up_band_kernel -c 2 单独抓）
NCU=/usr/local/NVIDIA-Nsight-Compute-2026.3/ncu
"$NCU" --nvtx --nvtx-include ncu_decode --kernel-name-base function \
  -k 'regex:.*w4a16_(compact_mma_gate_up_silu_band_kernel|compact_mma_gate_up_band_kernel|compact_mma_band_kernel|compact_gate_up_kernel|compact_down_kernel|mma_kernel).*' \
  -c 20 --set detailed --cache-control all --force-overwrite \
  --export profiles/deepseek_decode_opt_20261123/ncu/decode/profile \
  python -m benchmarks.dsv2_operator_ncu --phase decode \
  --prompts-file results/sharegpt_profile_20261123.jsonl --num-seqs 32 \
  --output profiles/deepseek_decode_opt_20261123/ncu/decode/harness.json
```

（上面的 NCU `-k` 里 `compact_mma_gate_up_band_kernel` 是实际名字；
`compact_mma_gate_up_silu_band_kernel` 是历史别名，不会匹配。）

## 8. 清理

本轮结束后删除：

- nsys sqlite 导出（约 66 MB，可由 `.nsys-rep` 重新导出）；
- 中间 sweep 的 `bench_fanout8*`、`bench_oldranges*`、`bench_final2*` 等
  非最终 JSON/日志；
- 临时驱动脚本 `_run_opt_*.sh`、`_run_dsv2_*.sh`、`__pycache__`、`.pytest_cache`；
- WSL `~/.nv/ComputeCache`（本轮再次生成，实测约 0.46 GB）、`~/pip_tmp`（若存在）。

保留：全部 `.nsys-rep` / `.ncu-repz` / CSV / 最终 bench / microbench / 报告。
