# Qwen3-0.6B DFlash 与 EAGLE 配对测试

## 草稿与验证

DFlash 从 Qwen3-0.6B 的残差流取检查点指定的三组特征。特征融合后，批量投影成各草稿层的 K/V 并写到独立分页池。草稿用一个已输出的 anchor token 加七个 mask，执行一次批量双向 attention。对照最多送四个候选给目标模型。目标走现有因果验证路径，最长匹配前缀和补偿 token 都按目标模型的采样结果提交。分页草稿缓存只提交已验证的目标特征，临时草稿块跨页时使用独立暂存页。DFlash 使用私有前缀块，抢占后重算。 [DFlash 论文](https://arxiv.org/abs/2602.06036) · [运行代码](../nanovllm/layers/dflash.py)

使用 `arnabwithab/Qwen3-0.6B-DFlash-noshift-b8` 的 3 层、32,000 草稿词表权重，revision `b4a61819293fdb9106e648b0422c38dbf90bb3f4`。本地 `model.safetensors` 已核验 SHA-256 为 `5084f318a312a5aa8c999caa6b86bd644d3fa7846e7f0c45c545009fa852ea68`。三代 EAGLE 沿用上一轮的权重及验证配置。 [固定版本草稿](https://huggingface.co/arnabwithab/Qwen3-0.6B-DFlash-noshift-b8/commit/b4a61819293fdb9106e648b0422c38dbf90bb3f4) · [上一轮 EAGLE 报告](_eagle_comparison_20261119.md)

## 测试条件

RTX 5060 Ti、TP=1、BF16、eager、默认 `cuda_decode`、普通分页 KV，显存预算 50%，关闭跨请求前缀复用。每个工作负载测试 B=1/8/32、seed 17/42/99，greedy 忽略 EOS，每请求生成 32 token。重复文本上下文上限 256，ShareGPT held-out 上下文上限 1024。EAGLE-1 最大草稿长度 4，EAGLE-2/3 树深 4、top-k 2、最多 8 个候选节点，DFlash 最多验证 4 个候选。

重复文本试跑时，共用 RMSNorm 编译入口因新增草稿形状达到 PyTorch 重编译上限。正式对照给 DFlash 的 RMSNorm 和 SwiGLU 使用独立编译入口，重新运行五种模式。试跑产物 `paired_repeat.json` 保留用于核对，正式重复文本数据为 `paired_repeat_final.json`。

18 组正式测试共 90 次模式运行。已核对全部组和上一轮的输入指纹，每种模式均生成 7,872 个 token，期间无请求抢占。`test_dflash.py`、EAGLE 树、投机解码及默认后端回归共 58 项通过。

## 端到端速度

表中为三个种子的吞吐中位数，单位 tok/s。最后一列先计算 DFlash 相对本轮无投机的逐 seed 吞吐变化，再取中位数。DFlash 与 EAGLE 的相对速度同样按配对计算。

| 工作负载 | B | 无投机 | EAGLE-1 | EAGLE-2 | EAGLE-3 | DFlash | DFlash vs 无投机 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 重复文本 | 1 | 42.5 | 37.0 | 78.9 | 112.3 | 92.0 | +172.3% |
| 重复文本 | 8 | 581.9 | 413.2 | 212.4 | 133.3 | 498.3 | +25.8% |
| 重复文本 | 32 | 1319.3 | 2068.9 | 270.7 | 232.1 | 1757.8 | +28.0% |
| ShareGPT | 1 | 37.7 | 64.1 | 52.8 | 67.0 | 41.5 | +61.9% |
| ShareGPT | 8 | 380.3 | 202.7 | 158.9 | 133.6 | 411.5 | +26.5% |
| ShareGPT | 32 | 715.2 | 963.1 | 231.3 | 274.4 | 832.5 | +16.4% |

重复文本 B32 三组 DFlash 都快于各自的无投机基线，变化为 +28.0%、+33.2%、+14.2%。ShareGPT B32 的变化为 −46.9%、+84.9%、+16.4%，配对比值中位数为 EAGLE-1 的 0.86×、EAGLE-2 的 3.60×、EAGLE-3 的 2.87×。ShareGPT B8 的 DFlash 在三个种子下均快于 EAGLE-1，在三个种子下也均快于 EAGLE-2/3。小 batch 的端到端吞吐波动较大，例如重复文本 B1 的无投机在 23.1–90.4 tok/s 间变化。

对照上一轮同输入数据，ShareGPT B32 的无投机和 EAGLE-1/2/3 中位吞吐是 1222.7、1509.0、207.7、180.5 tok/s，重复文本 B32 则是 1945.1、1661.5、242.5、202.5 tok/s。两轮的无投机基线发生了波动，以上的速度比来自本轮同 seed 的配对。 [上一轮完整数字](_eagle_comparison_20261119.md#端到端吞吐) · [重复文本 JSON](../results/dflash_comparison_20261119/paired_repeat_final.json) · [ShareGPT JSON](../results/dflash_comparison_20261119/paired_sharegpt.json)

## 验证效率和输出一致性

下表是 ShareGPT B32 三种子中位值。接受率按提议的草稿 token 计。每请求每轮输出含一个目标模型选出的 token，有效验证行等于这些行加已接受草稿行，占总验证行的比例。目标前向数按完整 batch 计。

| 模式 | 草稿接受率 | 有效验证行 | 每请求每次验证输出 token | 目标 decode 前向数 | TTFT p50 (ms) | TPOT p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 无投机 | — | — | — | 32 | 58.71 | 43.14 |
| EAGLE-1 | 13.9% | 33.1% | 1.48 | 27 | 58.30 | 25.53 |
| EAGLE-2 | 8.7% | 20.0% | 1.61 | 25 | 58.88 | 136.69 |
| EAGLE-3 | 16.6% | 26.5% | 2.24 | 18 | 57.76 | 108.51 |
| DFlash | 2.0% | 23.0% | 1.07 | 31 | 62.07 | 36.44 |

DFlash 在重复文本 B32 的接受率为 25.0%，每请求每次验证输出 1.94 个 token，目标 decode 前向从 32 次减至 17 次，有效验证行 40.8%。因此相同的草稿实现在两种负载上的收益不同。

Greedy 输出按每请求完整的 32 个 token 比对。重复文本的 123/123 在四种投机模式上均一致。ShareGPT 的一致率为 EAGLE-1 118/123、EAGLE-2 101/123、EAGLE-3 99/123、DFlash 119/123。DFlash 的四个分歧均在 B8，B32 为 96/96。上一轮三代 EAGLE 的 ShareGPT 对照分别为 119/123、100/123、98/123。

另一次 ShareGPT logits 诊断中，DFlash 的 123 个请求输出与主测完全一致，无投机自身有 2/123 个请求变化。诊断轮的 6 个首分歧位置，两侧的 top-2 token 集合均相同，两侧 top-1 与 top-2 间距均不超过 0.125 logit。这些数字衡量与目标 greedy 路径的逐 token 一致性。 [诊断 JSON](../results/dflash_comparison_20261119/logits.json)

## 分段诊断

取与上一轮 EAGLE 分段诊断相同的 seed 17 和输入。CUDA Event 区间包括提交工作时可能的设备空等。目标侧为纯 decode、spec 与 mixed 前向之和，纯 prefill 单独记录。单位为毫秒。

| 工作负载 | B | DFlash 草稿 | DFlash 目标 | 无投机目标 |
| --- | ---: | ---: | ---: | ---: |
| 重复文本 | 1 | 123.0 | 368.8 | 679.0 |
| 重复文本 | 8 | 70.5 | 262.6 | 1308.3 |
| 重复文本 | 32 | 110.7 | 389.9 | 796.9 |
| ShareGPT | 1 | 91.4 | 389.4 | 1320.7 |
| ShareGPT | 8 | 208.0 | 666.7 | 1451.6 |
| ShareGPT | 32 | 131.3 | 494.6 | 383.1 |

ShareGPT B32 的草稿区间为 131.3 ms，上一轮 EAGLE-1/2/3 的同条件草稿区间分别是 339.4、4436.9、3895.6 ms。DFlash 的批量并行草稿明显降低了这一阶段的成本。该种子下 DFlash 的接受率仍只有约 2%，目标前向区间为 494.6 ms，对照的纯 decode 与 mixed 为 383.1 ms。端到端吞吐以无计时钩子的正式配对为准。 [分段 JSON](../results/dflash_comparison_20261119/phases.json) · [上一轮分段数据](../results/eagle_comparison_20261118/phases.json)

## 复现

```bash
draft=results/dflash_qwen3_0.6b/verified
hf download arnabwithab/Qwen3-0.6B-DFlash-noshift-b8 config.json model.safetensors \
  --revision b4a61819293fdb9106e648b0422c38dbf90bb3f4 --local-dir "$draft"

trace=results/supplemental/sharegpt_heldout_chat_template_100_20261016.jsonl
common=(--speculative none,eagle,eagle2,eagle3,dflash --batch-sizes 1,8,32
  --seeds 17,42,99 --sampling-mode greedy --cold-prefix --max-draft-len 4
  --min-input-len 64 --max-input-len 192 --max-output-len 32 --gpu-memory-utilization .5
  --dflash-path "$draft")

python -m pytest tests/test_dflash.py tests/test_eagle_tree.py tests/test_spec_decode.py tests/test_linear_backend_default.py -q
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --styles repeat \
  --max-model-len 256 --output results/dflash_comparison_20261119/paired_repeat_final.json
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --styles sharegpt \
  --prompts-file "$trace" --max-model-len 1024 \
  --output results/dflash_comparison_20261119/paired_sharegpt.json
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --speculative none,dflash \
  --styles repeat,sharegpt --prompts-file "$trace" --max-model-len 1024 --seeds 17 \
  --phase-timing --output results/dflash_comparison_20261119/phases.json
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --speculative none,dflash \
  --styles sharegpt --prompts-file "$trace" --max-model-len 1024 \
  --logit-diagnostics --output results/dflash_comparison_20261119/logits.json
```
