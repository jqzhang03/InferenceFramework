# DeepSeek-V2-Lite CUDA MLA decode 流水线优化

RTX 5060 Ti（SM120），PyTorch 2.8.0+cu128，CUDA 12.8。CUDA 将共享内存中
Q 与 KV 的行跨度分别从 576、592 调为 584 个 FP16 元素。SM120 的 BF16 路径
默认提前一块读取下一组 16 token 的 KV 到寄存器，将这部分全局读与当前块的
QK、softmax 和 PV 重叠。三种候选都沿用原来的 FP16 MMA 和 FP32 累加。
`cp.async` 候选通过额外的 BF16 共享内存暂存区实现异步读取。

## 独立 CUDA Graph 计时

同一组输入分别运行原 CUDA、行填充、行填充加寄存器预取、行填充加
`cp.async` 和 Triton。筛选种子为 17/42/99，复测种子为 7/23/61。每轮
打乱执行顺序，每个样本摊销 64 次 MLA 调用。下表为复测中各个种子
中位数的中位数，单位为 µs，含 query 拼接及各自的注意力和归约。
[独立计时及逐种子数据](../profiles/cuda_mla_pipeline_20261128/pipeline_timing.json)

| 行数 | KV token | 原 CUDA | 行填充 | 寄存器预取 | `cp.async` | Triton |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 11.24 | 10.61 | 9.33 | 9.60 | 28.81 |
| 1 | 1,024 | 16.02 | 14.98 | 12.31 | 12.96 | 104.03 |
| 1 | 4,096 | 42.04 | 37.24 | 25.72 | 28.83 | 404.78 |
| 8 | 128 | 11.85 | 11.62 | 10.36 | 10.64 | 29.39 |
| 8 | 1,024 | 36.95 | 30.83 | 26.70 | 38.33 | 104.53 |
| 8 | 4,096 | 150.87 | 130.94 | 104.60 | 114.79 | 399.44 |
| 16 | 128 | 16.46 | 14.98 | 14.36 | 17.36 | 29.89 |
| 16 | 1,024 | 59.76 | 48.26 | 40.00 | 60.55 | 105.55 |
| 16 | 4,096 | 285.05 | 248.51 | 199.73 | 224.35 | 401.56 |
| 32 | 128 | 28.69 | 25.74 | 23.66 | 30.57 | 30.60 |
| 32 | 1,024 | 154.68 | 135.97 | 111.20 | 118.80 | 109.89 |
| 32 | 4,096 | 550.91 | 478.70 | 389.72 | 422.88 | 407.27 |

72 个形状和种子对照中，寄存器预取比单独行填充快 4.0%–35.4%，
比原 CUDA 快 11.5%–39.6%。CUDA 候选对原 CUDA 逐位一致，Triton
通过 `atol=0.002, rtol=0.008`。32 行、1024 token 时两套后端的优势
随种子变化，Triton 在复测中位数领先 1.2%。自动分派在 16 行以内选
CUDA，超过 16 行选 Triton。显式选择 CUDA 可在 32 行使用优化内核。

## Nsight Compute

32 行、4096 token，同一随机输入预热后采集注意力主核。Shared load bank
conflict 为事件计数，占用资源为每 CTA 分配字节数。DRAM 使用
`dram__cycles_active.avg.pct_of_peak_sustained_elapsed`，Tensor 使用
`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`。
Long Scoreboard 使用按发射周期归一化的比值。
[原 CUDA](../profiles/cuda_mla_pipeline_20261128/ncu/b32_l4096_cuda_none.ncu-repz) ·
[行填充](../profiles/cuda_mla_pipeline_20261128/ncu/b32_l4096_cuda_padding_none.ncu-repz) ·
[寄存器预取](../profiles/cuda_mla_pipeline_20261128/ncu/b32_l4096_cuda_registers_none.ncu-repz) ·
[`cp.async`](../profiles/cuda_mla_pipeline_20261128/ncu/b32_l4096_cuda_async_none.ncu-repz) ·
[Triton](../profiles/cuda_mla_pipeline_20261128/ncu/b32_l4096_triton_none.ncu-repz)

| 指标 | 原 CUDA | 行填充 | 寄存器预取 | `cp.async` | Triton |
|---|---:|---:|---:|---:|---:|
| Shared load bank conflicts | 10,761,669 | 270,339 | 275,615 | 274,285 | 0 |
| Long Scoreboard | 8.36 | 9.31 | 4.70 | 1.10 | 0.26 |
| 可发射 warp / scheduler / 活跃周期 | 0.249 | 0.274 | 0.331 | 0.285 | 0.295 |
| DRAM 持续峰值利用率 | 67.32% | 81.08% | 94.16% | 87.36% | 88.26% |
| Tensor 管线活跃率 | 17.24% | 19.04% | 24.71% | 22.73% | 58.12% |
| 每 CTA 分配共享内存（字节） | 47,360 | 47,360 | 47,360 | 65,792 | 74,752 |
| 每线程寄存器 | 66 | 66 | 108 | 75 | 152 |
| 可驻留 CTA / SM | 2 | 2 | 2 | 1 | 1 |

行填充使 shared load bank conflict 减少 97.5%。再加寄存器预取后，
Long Scoreboard 比行填充版降低 49.5%，最热 `PRMT` 的未发射样本
从 19,176 减到 6,512，寄存器增至 108 个且 local load/store sector
仍为零。`cp.async` 的 SASS 含全局到共享的 `LDGSTS`，能进一步减少
Long Scoreboard。其额外 18,432 字节使每个 SM 只能驻留一个 CTA，
可发射 warp 和 DRAM 利用率低于寄存器预取，32 行、4096 token 的
独立 Graph 延迟多用 33.16 µs。
[寄存器预取 SourceCounters](../profiles/cuda_mla_pipeline_20261128/ncu/cuda_registers_source.csv) ·
[`cp.async` SourceCounters](../profiles/cuda_mla_pipeline_20261128/ncu/cuda_async_source.csv)

## 整模回放

DeepSeek-V2-Lite 使用 INT4 group64 权重，输入为 128 token，
每条请求贪心生成 16 token。每组运行三次，每次执行 15 次 decode
Graph 回放。首轮将启动阶段采集的优化版 Graph 与启动后重新采集
的原版 Graph 配对。8 行的 384 个生成 token 逐一匹配。1 行在
三个配对中各有 2 个 token 不同，最早在第 3 个 token。
原版两个候选的 BF16 logits 都是 24.375，优化版则为 24.250 和
24.375。该轮的每 token 延迟中位数为 19.03→18.50 ms（1 行）
和 52.39→52.01 ms（8 行）。
[整模独立 Graph 回放](../results/cuda_mla_pipeline_20261128_model.json)

诊断版在同一张 1 行 Graph 内把每层的同一组 query、KV、页表和长度
同时送入原版和优化版 MLA。27 层 × 15 步共 405 次真实模型输入对照，
注意力输出全部逐位一致，包括第 3 个 token 对应的 decode 步。
[同一 Graph 逐层诊断](../results/cuda_mla_pipeline_20261128_model_diag.json)

复测将两版 Graph 都放在引擎启动后采集。1 行的 48 个和 8 行的
384 个生成 token 全部匹配。每 token 延迟中位数为 16.24→16.04 ms
（1 行）和 45.56→45.77 ms（8 行）。这组 128-token 输入下，整模
延迟的变化幅度约 1%，算子加速见上面的独立计时。
[对称采图整模复测](../results/cuda_mla_pipeline_20261128_model_matched.json)

## 复现

```bash
python -m benchmarks.cuda_mla_profile --mode pipeline --output profiles/cuda_mla_pipeline_run
python -m benchmarks.cuda_mla_decode_study --model /path/to/DeepSeek-V2-Lite --model-baseline original --diagnose-original --output results/cuda_mla_pipeline_model_run.json
pytest -q tests/test_cuda_mla_decode.py tests/test_mla.py tests/test_linear_backend_default.py
```
