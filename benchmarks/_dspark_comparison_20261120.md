# Qwen3-0.6B DSpark 投机解码配对测试

## 实现与权重

DSpark 将 Qwen3-0.6B 的第 2、14、25 个残差流边界特征融合，投影并写入分页草稿 KV。三层草稿骨干使用因果滑窗注意力，对锚点和七个 mask 执行一次批量前向，从锚点预测第一个候选。随后在 GPU 按位置批量计算 rank-256 Markov 偏置，使用上一个目标词表 token 的 embedding 修正下一位置的 logits。置信度头使用归一化隐藏状态和同一 Markov embedding。候选和置信度在每轮一起传回 CPU，目标模型只验证选出的连续前缀。草稿 KV 只提交已验证位置的目标特征，跨页的临时 KV 使用独立暂存页。 [实现](../nanovllm/layers/dspark.py) · [原论文](https://arxiv.org/abs/2607.05147) · [Speculators 参考](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/)

使用公开权重 `atutej/dspark_qwen3_0_6b_nemotron-terminal-corpus_targetgen-20261012`，revision `eacfaf2f3d52917e07f938a45e17f30cf04aaef0`，SHA-256 `ac9b0ed2d31370af533e2057d2902a959f323cf1fa5a7affd20f39b8903d1fc4`。该草稿的候选词表为 32,000，训练数据来自 Nemotron Terminal Corpus 的目标模型重生成输出。词嵌入复用本地 Qwen3-0.6B 权重。测试对比各方法已有的公开或本地草稿及各自的优化实现。 [固定版本权重](https://huggingface.co/atutej/dspark_qwen3_0_6b_nemotron-terminal-corpus_targetgen-20261012/tree/eacfaf2f3d52917e07f938a45e17f30cf04aaef0) · [DFlash 原有对照](_dflash_comparison_20261119.md)

置信度调度先计算各位置的前缀存活率，再按存活率跨请求排序。以预期提交 token 数除以实测拟合的目标验证步长估算吞吐，在第一次无收益时停止扩展。采用 128-token 缓存前缀、1～32 条活动请求、每请求 0～4 个候选。验证和 greedy 采样的 GPU 时间用 CUDA Graph 回放测量，以减弱 WSL 主机调度抖动。拟合曲线只按验证行数建模，正式端到端仍走 eager。目标采样和最长匹配前缀提交沿用引擎现有实现。 [验证成本脚本](dspark_verifier_cost.py) · [原始取样及拟合值](../results/dspark_comparison_20261120/verifier_cost.json)

35 个验证形状每组采集五次，每次摊销 16 次回放。拟合耗时为 `4.561 + 0.0515 × 验证行数` ms，表覆盖 1～160 行。

## 测试条件

RTX 5060 Ti 16GB，TP=1，BF16，eager，目标投影走默认 `cuda_decode`，显存预算 50%，关闭跨请求前缀复用。greedy，忽略 EOS，每请求固定输出 32 token。两种负载分别跑 B1、B8、B32，使用 seed 17、42、99。重复文本的上下文上限是 256，ShareGPT held-out 的上下文上限是 1024。每轮草稿深度最多 4，EAGLE-2/3 固定树宽 top-k 为 2，候选节点总预算 8。另测 ShareGPT B8/B32 的自适应 EAGLE 树。墙钟和吞吐正式测量关闭阶段计时，单独用 seed 17 采集阶段区间。

固定树主测 18 组、108 次模式运行，输入 token 指纹与历史对照 18/18 一致。每种模式每个负载共生成 3,936 个 token，全部按预算结束，无抢占。67 项回归通过，覆盖 Markov 词表映射、置信度截断、滑窗跨页数值对照、真实草稿权重的混合批次与分块 Prefill，以及原有投机解码和默认线性后端。 [测试代码](../tests/test_dspark.py)

## 端到端耗时和吞吐

单元格为三个 seed 的中位吞吐（tok/s）与整批耗时（s）。墙钟包括请求入队、Prefill、草稿与验证、调度和输出整理。模型加载和预热在区间之外。

| 负载 | B | 无投机 | EAGLE-1 | EAGLE-2 | EAGLE-3 | DFlash | DSpark |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 重复文本 | 1 | 43.7 / 0.732 | 108.2 / 0.296 | 61.0 / 0.524 | 70.9 / 0.451 | 118.6 / 0.270 | **146.1 / 0.219** |
| 重复文本 | 8 | 411.1 / 0.623 | 430.7 / 0.594 | 309.2 / 0.828 | 321.8 / 0.795 | 344.4 / 0.743 | **1281.9 / 0.200** |
| 重复文本 | 32 | 1266.6 / 0.808 | 2107.2 / 0.486 | 1998.5 / 0.512 | 2150.6 / 0.476 | 2048.5 / 0.500 | **3872.0 / 0.264** |
| ShareGPT | 1 | 23.2 / 1.382 | 75.6 / 0.423 | 66.1 / 0.484 | 90.2 / 0.355 | 65.8 / 0.487 | **127.8 / 0.250** |
| ShareGPT | 8 | 414.2 / 0.618 | 327.2 / 0.782 | 460.8 / 0.556 | 330.7 / 0.774 | 312.4 / 0.819 | **732.5 / 0.349** |
| ShareGPT | 32 | 1212.6 / 0.844 | 1511.5 / 0.677 | 1168.4 / 0.876 | 1836.2 / 0.558 | 1065.1 / 0.961 | **2414.9 / 0.424** |

下表先计算 DSpark 相对同一 seed 对照的吞吐变化，再取中位数。括号内是 DSpark 获胜的 seed 数。ShareGPT B8 和 B32 相对无投机的胜率均为 2/3，逐 seed 变化范围分别为 −3.8%～+80.8%、−32.0%～+241.4%。B1 的基线波动更大。

| 负载 | B | 无投机 | EAGLE-1 | EAGLE-2 | EAGLE-3 | DFlash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 重复文本 | 1 | +179.2%（3） | +23.1%（3） | +100.1%（3） | +78.0%（3） | +87.0%（3） |
| 重复文本 | 8 | +139.8%（3） | +175.0%（3） | +263.4%（3） | +243.2%（3） | +291.4%（2） |
| 重复文本 | 32 | +81.6%（3） | +61.0%（3） | +88.9%（3） | +75.9%（3） | +60.6%（3） |
| ShareGPT | 1 | +451.9%（2） | +69.0%（2） | +75.4%（3） | +41.7%（2） | +99.2%（3） |
| ShareGPT | 8 | +30.1%（2） | +32.8%（3） | +66.5%（2） | +120.9%（3） | +119.2%（3） |
| ShareGPT | 32 | +117.7%（2） | +55.3%（3） | +125.9%（3） | +36.2%（2） | +147.8%（3） |

[重复文本 JSON](../results/dspark_comparison_20261120/repeat.json) · [重复文本 CSV](../results/dspark_comparison_20261120/repeat.csv) · [ShareGPT JSON](../results/dspark_comparison_20261120/sharegpt.json) · [ShareGPT CSV](../results/dspark_comparison_20261120/sharegpt.csv)

### 自适应 EAGLE 树复测

ShareGPT B8/B32 另测 6 组、36 次模式运行，和历史自适应对照的输入指纹 6/6 一致。下表沿用吞吐（tok/s）与墙钟（s）口径。这一轮只有 EAGLE-2、EAGLE-3 使用自适应树。该设置下 B8 的 DSpark 中位吞吐落后于无投机和 EAGLE-2/3，逐 seed 配对分别为 −28.0%、−30.0%、−37.1%，各赢 1/3。B32 分别为 +113.9%、+21.0%、+119.6%，各赢 2/3。两轮 DSpark 的逐 seed 草稿接受率相同，墙钟差异反映重复测量波动。

| B | 无投机 | EAGLE-1 | 自适应 EAGLE-2 | 自适应 EAGLE-3 | DFlash | DSpark |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 541.4 / 0.473 | 278.3 / 0.920 | 583.3 / 0.439 | 611.0 / 0.419 | 266.9 / 0.959 | 408.5 / 0.627 |
| 32 | 1186.8 / 0.863 | 1422.3 / 0.720 | 2097.5 / 0.488 | 1156.4 / 0.886 | 1045.9 / 0.979 | **2538.8 / 0.403** |

自适应对照中，DSpark 相对 DFlash 的配对吞吐变化是 B8 +53.0%（3/3）、B32 +54.4%（2/3）。 [自适应 JSON](../results/dspark_comparison_20261120/sharegpt_adaptive.json) · [自适应 CSV](../results/dspark_comparison_20261120/sharegpt_adaptive.csv)

## 草稿接受与验证

接受率是目标实际接受的草稿 token 占提出 token 的比例，有效行占比还包括每轮生成一个目标 token 的必要验证行。目标前向数用 `decode_iterations`，多步 decode 中的每次目标前向单独计数。下表是三个 seed 的中位数。

| 负载 | B | 接受率 DSpark / DFlash | 有效行 DSpark / DFlash | 目标前向 无投机 / DFlash / DSpark |
| --- | ---: | ---: | ---: | ---: |
| 重复文本 | 8 | 92.8% / 25.0% | 94.3% / 40.8% | 34 / 19 / 10 |
| 重复文本 | 32 | 96.6% / 25.0% | 97.4% / 40.8% | 32 / 17 / 9 |
| ShareGPT | 8 | 48.8% / 1.9% | 62.3% / 22.9% | 34 / 32 / 17 |
| ShareGPT | 32 | 60.1% / 2.0% | 74.0% / 23.0% | 32 / 31 / 19 |

ShareGPT B32 每请求每轮平均提出的草稿数由 DFlash 的 3.67 降至 DSpark 的 1.87，接受的草稿 token 总数中位数由 68 增至 525。该负载上，EAGLE-3 的目标前向中位数是 18 次、有效验证行占比 26.5%，与 DSpark 的 19 次和 74.0% 对照。收益同时受各草稿权重的训练数据、候选分布和实现差异影响。

固定树对照中，重复文本 123/123 个 greedy 请求和无投机逐 token 一致。ShareGPT 一致数为 EAGLE-1 118/123、EAGLE-2 102/123、EAGLE-3 99/123、DFlash 119/123、DSpark 118/123。DSpark 的 5 个分歧分布在 B1（1 个）和 B8（4 个），B32 全部 96/96 一致。

独立 logits 诊断复现了全部 5 个首分歧位置。两边目标模型的 top-2 token 集合相同，每个分歧都是一侧并列、另一侧领先 0.125 logit。该现象符合 BF16 近并列下的 argmax 翻转。数据只覆盖 greedy 解码。 [Logits 诊断 JSON](../results/dspark_comparison_20261120/sharegpt_logits.json)

## 分段耗时

下表为 seed 17 的独立 CUDA Event 诊断，单元格为草稿区间 / 目标区间，单位 ms。目标区间汇总 decode、verify 与 mixed 模型前向，纯 Prefill 单列在原始结果。草稿区间含元数据准备、缓存更新、草稿前向、取回候选及选前缀，区间可能包含设备空等。正式吞吐取自上方无诊断钩子的实验。

| 负载 | B | 无投机 | EAGLE-1 | EAGLE-2 | EAGLE-3 | DFlash | DSpark |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 重复文本 | 8 | — / 411.7 | 150.6 / 446.2 | 95.9 / 249.0 | 98.6 / 241.3 | 254.7 / 435.5 | **51.0 / 151.3** |
| 重复文本 | 32 | — / 504.0 | 353.0 / 720.4 | 193.1 / 590.3 | 109.4 / 306.0 | 58.6 / 238.1 | **35.6 / 128.8** |
| ShareGPT | 8 | — / 794.0 | 204.0 / 586.3 | 125.2 / 312.3 | 167.9 / 345.5 | 156.7 / 550.9 | 270.9 / 753.8 |
| ShareGPT | 32 | — / 378.6 | 412.2 / 780.6 | 225.2 / 562.3 | 375.9 / 472.6 | 175.4 / 693.5 | **69.3 / 250.7** |

ShareGPT B32 的 DSpark 草稿与目标区间分别为 69.3 ms 和 250.7 ms，低于同轮 DFlash 的 175.4 ms 和 693.5 ms。ShareGPT B8 这个诊断轮次的 DSpark 草稿区间为 270.9 ms，目标侧相对无投机只节省了 40.2 ms，草稿开销超过这部分收益。 [重复文本阶段 JSON](../results/dspark_comparison_20261120/repeat_phases.json) · [ShareGPT 阶段 JSON](../results/dspark_comparison_20261120/sharegpt_phases.json)

## 复现

```bash
root=results/dspark_comparison_20261120
draft=results/dspark_qwen3_0.6b/verified
hf download atutej/dspark_qwen3_0_6b_nemotron-terminal-corpus_targetgen-20261012 \
  config.json model.safetensors --revision eacfaf2f3d52917e07f938a45e17f30cf04aaef0 \
  --local-dir "$draft"

bash benchmarks/run_qwen_cuda.sh --dspark-cost --dspark-path "$draft" \
  --output "$root/verifier_cost.json"
common=(--speculative none,eagle,eagle2,eagle3,dflash,dspark --batch-sizes 1,8,32
  --seeds 17,42,99 --sampling-mode greedy --cold-prefix --max-draft-len 4
  --min-input-len 64 --max-input-len 192 --max-output-len 32 --gpu-memory-utilization .5
  --dspark-path "$draft" --dspark-step-costs "$root/verifier_cost.json")
repeat=(--styles repeat --max-model-len 256)
sharegpt=(--styles sharegpt --max-model-len 1024
  --prompts-file results/supplemental/sharegpt_heldout_chat_template_100_20261016.jsonl)

bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" "${sharegpt[@]}" \
  --output "$root/sharegpt.json"
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" "${repeat[@]}" \
  --output "$root/repeat.json"
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" "${sharegpt[@]}" \
  --batch-sizes 8,32 --eagle-tree-policy adaptive --output "$root/sharegpt_adaptive.json"
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" "${sharegpt[@]}" \
  --batch-sizes 8,32 --seeds 17 --phase-timing --output "$root/sharegpt_phases.json"
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" "${repeat[@]}" \
  --batch-sizes 8,32 --seeds 17 --phase-timing --output "$root/repeat_phases.json"
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" "${sharegpt[@]}" \
  --speculative none,dspark --batch-sizes 1,8 --seeds 17,99 --logit-diagnostics \
  --output "$root/sharegpt_logits.json"
```

通过 Python API 使用时，传入 `LLM(model_dir, speculative="dspark", dspark_path=draft_dir)`。传入 `dspark_step_ms=tuple(costs["step_ms"])` 可启用实测验证成本曲线。缺少覆盖当前批次的曲线时，使用每请求前缀存活率阈值，默认 0.3。
