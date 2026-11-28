# DeepSeek-V2-Lite 三种 4 位权重量化对照

## 量化格式与 CUDA 路径

| 格式 | 权重编码 | 每组权重数 | 缩放 | 打包矩阵平均位数 |
| --- | --- | ---: | --- | ---: |
| INT4 | 等间隔的对称整数，取值 −7 至 7 | 64 | BF16 组尺度 | 4.25 |
| NVFP4 | E2M1 非等间隔浮点 | 16 | E4M3 组尺度 × FP32 矩阵尺度 | 4.50 + 矩阵尺度开销 |
| MXFP4 | E2M1 非等间隔浮点 | 32 | E8M0 的 2 次幂组尺度 | 4.25 |

NVFP4 复用仓库已有的权重编码。MXFP4 使用块最大绝对值的二进制指数减 2
来选择组尺度，并将块内数值按最近偶数规则量化到 E2M1。参见
[NVIDIA NVFP4 格式说明](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference)
和 [OCP MX v1.1 格式规范](https://www.opencompute.org/documents/ocp-microxcaling-mx-specification-v1-1-0/)。

两种新增 CUDA 路径均在内核内解包和反量化，1–4 行使用 warp 分组 GEMV，更多行
使用共享内存分块和 BF16/FP16 Tensor Core MMA。量化激活的 cuBLASLt NVFP4
W4A4 路径可单独选择。本实验三组均为 W4A16。FP4 路由专家按专家连续打包，
由 GPU 根据各专家实际行数选择 GEMV 或 MMA。INT4 对照使用已有的 compact
专家和 CUDA 投影，包括已优化的 gate/up 和 SiLU 融合。FP4 专家分别执行三次
投影，因此整模速度包含两条算子实现与当前融合程度的差异。

三组都用 BF16 原始权重流式量化，同样保留原精度的路由器、词嵌入和 LM head。
MLA 的 `kv_b` 在三组都保留各自的反量化视图供吸收式解码使用。流式加载现在
和普通加载一样遵循路由器的量化排除标记。

## 实验设置

- RTX 5060 Ti 16 GB（SM120），WSL2，PyTorch 2.8.0+cu128，TP=1，BF16 激活和 KV。
  三组固定使用 Triton MLA decode。
- 精度测试从本地 ShareGPT 留出集以种子 42 选取 16 段不同文本，每段取前 128 个
  token。将同一模型按层加载到 GPU 计算 BF16 参考，在 2,032 个真实 next-token
  位置上计算困惑度、BF16 到量化模型的平均 KL 散度及 top-1 一致率。
- 推理测试使用另外 8 条文本。B=1 和 B=8 各使用 128 token 输入、16 token 贪心
  输出、一次同形状预热和三次计时。Prefill 和 mixed 使用 eager，decode 使用
  CUDA Graph。统一显存预算为 0.85，关闭多步解码与 KV swap。
- 权重占用统计模型参数和量化权重缓冲区，包括 MLA 的反量化视图。运行期显存
  峰值还受到 KV 容量、CUDA Graph 池和临时激活影响。

## 精度

| 格式 | 困惑度 ↓ | KL(BF16 ‖ 量化) ↓ | 与 BF16 的 top-1 一致率 ↑ | 权重 GiB |
| --- | ---: | ---: | ---: | ---: |
| BF16 参考 | 11.928 | — | — | — |
| INT4 | 12.951 | 0.0881 | 87.70% | 8.455 |
| NVFP4 | 12.444 | 0.0390 | 90.70% | 8.900 |
| MXFP4 | 13.235 | 0.0912 | 86.71% | 8.455 |

在这 2,032 个位置，NVFP4 的三项指标均最接近 BF16。相对 INT4，它把平均
KL 降低约 56%，把 top-1 一致率提高 3.00 个百分点，增加约 0.445 GiB 权重
显存。MXFP4 与 INT4 的权重占用相同，困惑度高约 2.2%，top-1 一致率低
0.98 个百分点。精度结果适用于这组文本及这里的加载时量化方案。

## 推理效率

下表使用三次计时的中位数。吞吐包含 prefill、decode 和引擎调度，TTFT 是
首 token 延迟，TPOT 是首 token 后的每 token 平均间隔。

| 格式 | 并发请求 | 输出 token/s ↑ | TTFT ms ↓ | TPOT ms ↓ |
| --- | ---: | ---: | ---: | ---: |
| INT4 | 1 | 42.33 | 129.09 | 16.54 |
| NVFP4 | 1 | 45.47 | 85.68 | 17.83 |
| MXFP4 | 1 | 43.42 | 92.88 | 18.11 |
| INT4 | 8 | 129.38 | 287.94 | 45.70 |
| NVFP4 | 8 | 129.47 | 262.52 | 47.45 |
| MXFP4 | 8 | 127.45 | 281.78 | 47.15 |

单请求下，NVFP4 相对 INT4 的首 token 延迟下降 33.6%，推高端到端吞吐
7.4%。MXFP4 首 token 延迟下降 28.0%，吞吐增加 2.6%。INT4 的 TPOT
在两种负载下都最低。8 请求时三者吞吐中位数在 127.45–129.47 token/s，
NVFP4 的三次结果跨度为 109.82–129.73 token/s，INT4 为 128.93–130.19
token/s。8 请求下 NVFP4 与 INT4 的中位数差约 0.07%。

18 次计时的请求和输出长度均符合设置，每次 15 次 decode Graph 重放，
零前缀复用、零抢占。B=1 每次先 prefill 128 token。B=8 每次先 prefill
512 token，再执行包含 512 个 prefill token 和 4 个 decode token 的 mixed
批次。同一格式、同一批量的三次贪心输出一致。显存预算下实际分配的 KV
块数为 INT4 365、NVFP4 252、MXFP4 353，块数受权重、启动预热峰值和
当时的设备显存使用共同影响。

## 复现

```bash
python -m benchmarks.deepseek_fp4 \
  --model ~/huggingface/DeepSeek-V2-Lite \
  --data results/supplemental/sharegpt_500_heldout250_20261016.jsonl \
  --graph --variants nvfp4 mxfp4 int4 --output results/deepseek_fp4_run.json
python -m pytest tests/test_cuda_fp4.py tests/test_cuda_int4.py \
  tests/test_nvfp4.py tests/test_moe_int4_compact.py \
  tests/test_mla.py tests/test_linear_backend_default.py
```

124 项回归测试通过，包括 BF16/FP16 的 CUDA GEMV 与 MMA、尾块、非默认 stream，
以及改变路由输入后的 CUDA Graph 重放。

原始结果含逐次延迟、吞吐、显存和输出 token，以及文本标识和 token 指纹。
[实验数据](../results/deepseek_fp4_20261127_matched.json)
