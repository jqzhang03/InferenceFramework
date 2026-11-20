# Qwen3-0.6B EAGLE-1 / EAGLE-2 / EAGLE-3 对比

## 实现与测试条件

- `speculative="eagle"` 使用本地 EAGLE-1 草稿层逐步生成单链。`speculative="eagle2"` 复用同一 `.pt` 权重，按累计置信度动态扩展候选树。`speculative="eagle3"` 从目标模型低、中、高层提取特征，使用独立 draft head 和逐请求的草稿 KV。
- EAGLE-2/3 在一次目标前向中验证整棵树，每一行注意力读取共享前缀和自己的祖先链。验证根据目标模型的采样结果选中一条路径，再将选中的 KV 行压紧写回逻辑序列。GPU 回归测试覆盖跨页 KV 压紧、兄弟分支隔离、真实检查点增量草稿 KV，以及分块 prefill 和混合批次。
- RTX 5060 Ti 16 GB、CUDA 12.8、PyTorch 2.8.0、FlashAttention 2.8.3.post1。四种模式都用 TP=1、BF16、普通 paged KV、默认 `cuda_decode`、eager，关闭跨请求的前缀共享。EAGLE-1 最大草稿长度 4，EAGLE-2/3 树深 4、每次扩展 top-k 2、全树最多 8 个候选节点。EAGLE-1/2 使用 `results/eagle_layer.pt`。EAGLE-3 使用本地 `results/eagle3_qwen3_0.6b` 中的公开草稿权重。EAGLE-2 和 EAGLE-3 之间的差异同时包含模型权重与执行路径。 [EAGLE-3 检查点及固定 revision](https://huggingface.co/Nicolassuez/Qwen3-0.6B-eagle3/commit/00b7c41f32655e90ee65ac5b90bb381905ce1700)
- 每个工作负载运行 B=1/8/32、seed 17/42/99，四模式共享完全相同的输入 token。Greedy、忽略 EOS、每请求输出 32 token。重复文本的实际输入为 60–61 token，ShareGPT held-out 的预渲染输入为 9–489 token。重复文本用 256 上下文上限，ShareGPT 用 1024。每模式跨运行合计 246 个请求和 7,872 个输出 token。123 个 ShareGPT 请求包含 47 个不同的 prompt ID。主测试中未发生请求抢占。

## 端到端吞吐

数值为三个种子的 `tok/s` 中位数。括号内为各自同 seed 对基线吞吐比的中位变化，和两列吞吐中位数之比可能不同。计时覆盖从提交到生成完毕，不含模型加载和预热。

| 工作负载 | B | 无投机 | EAGLE-1 | EAGLE-2 | EAGLE-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 重复文本 | 1 | 43.2 | 72.3 (+68.9%) | 75.0 (+103.3%) | 47.8 (+10.8%) |
| 重复文本 | 8 | 480.1 | 545.0 (+13.5%) | 201.5 (−52.1%) | 196.6 (−65.6%) |
| 重复文本 | 32 | 1945.1 | 1661.5 (−5.0%) | 242.5 (−87.6%) | 202.5 (−89.6%) |
| ShareGPT | 1 | 79.1 | 46.4 (−47.8%) | 46.3 (+25.2%) | 69.6 (−12.1%) |
| ShareGPT | 8 | 587.8 | 313.8 (−47.8%) | 102.5 (−70.8%) | 137.6 (−60.8%) |
| ShareGPT | 32 | 1222.7 | 1509.0 (−6.5%) | 207.7 (−78.1%) | 180.5 (−81.8%) |

ShareGPT B1 的无投机三次测量为 32.0、79.1、91.6 tok/s，EAGLE-2 相对变化分别为 +44.9%、−59.7%、+25.2%。该组波动较大。B8 和 B32 下，两种树路径的吞吐在每个种子都低于各自基线。

## 验证效率

下表为 ShareGPT 三个种子的中位数。接受率是接受的草稿 token / 提议的草稿 token。有效验证行比例是（接受的草稿行 + 每请求每轮的一行目标输出）/ 总验证行。目标 decode 前向数包含批量验证和与解码混合的前向。TTFT 和 TPOT 显示每次运行请求 p50 的跨种子中位数，完整 p95 在原始文件中。

| B | 模式 | 接受率 | 有效验证行 | decode 前向数 | TTFT p50 (ms) | TPOT p50 (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 无投机 | — | — | 31 | 13.68 | 12.60 |
| 1 | EAGLE-1 | 15.7% | 34.4% | 20 | 15.75 | 20.86 |
| 1 | EAGLE-2 | 7.6% | 18.9% | 20 | 13.27 | 21.72 |
| 1 | EAGLE-3 | 21.6% | 31.0% | 12 | 18.39 | 14.23 |
| 8 | 无投机 | — | — | 34 | 30.17 | 12.40 |
| 8 | EAGLE-1 | 14.0% | 33.2% | 26 | 48.25 | 23.16 |
| 8 | EAGLE-2 | 8.7% | 20.1% | 23 | 30.91 | 77.26 |
| 8 | EAGLE-3 | 15.8% | 25.7% | 20 | 140.90 | 52.45 |
| 32 | 无投机 | — | — | 32 | 59.41 | 24.34 |
| 32 | EAGLE-1 | 13.9% | 33.1% | 27 | 57.72 | 16.42 |
| 32 | EAGLE-2 | 8.7% | 20.0% | 25 | 58.18 | 149.47 |
| 32 | EAGLE-3 | 16.6% | 26.5% | 18 | 59.05 | 173.98 |

ShareGPT B32 的每次目标 decode 前向产出从 31.0 token 增至 EAGLE-1 的 36.8、EAGLE-2 的 39.7 和 EAGLE-3 的 55.2，按整个 batch 计。EAGLE-2/3 每个请求每次验证分别提出约 7.05 和 7.51 个候选，有效利用了约五分之一至四分之一的验证行。该 B32 条件的中位峰值已分配显存为 6.31、6.30、6.32、6.34 GiB，包含按可用预算预分配的目标 KV 块池。

## 草稿与目标模型耗时

独立诊断用 seed 17，在相同 CUDA 流上给草稿生成和目标模型前向分别加 CUDA Event。下表单位为毫秒，斜线左右依次为草稿和目标 decode/spec + mixed 计时，纯 prefill 另行记录。Event 区间包含提交工作时可能的设备空等。端到端吞吐采用上面的无钩子结果。 [分段 JSON](../results/eagle_comparison_20261118/phases.json) · [分段 CSV](../results/eagle_comparison_20261118/phases.csv)

| 工作负载 | B | 无投机 | EAGLE-1 | EAGLE-2 | EAGLE-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 重复文本 | 1 | — / 590 | 90 / 291 | 73 / 401 | 322 / 562 |
| 重复文本 | 8 | — / 1465 | 222 / 735 | 633 / 289 | 1267 / 372 |
| 重复文本 | 32 | — / 588 | 211 / 444 | 3160 / 539 | 4263 / 321 |
| ShareGPT | 1 | — / 429 | 151 / 499 | 120 / 326 | 148 / 264 |
| ShareGPT | 8 | — / 655 | 208 / 590 | 764 / 481 | 808 / 511 |
| ShareGPT | 32 | — / 385 | 339 / 636 | 4437 / 699 | 3896 / 531 |

两个 B32 负载上，EAGLE-2/3 草稿区间占本轮生成墙钟的 84%–92%。当前跨请求逐个建树，扩展每层要把置信度和候选 token 从 GPU 取回 Python。EAGLE-3 还要拼接分支的草稿 KV。计时定位到了草稿阶段，Python 同步、草稿前向和分支 KV 操作的各自占比需进一步拆分。 [建树代码](../nanovllm/engine/eagle_tree.py) · [草稿调度代码](../nanovllm/engine/llm_engine.py)

## EAGLE-3 窄树对照

ShareGPT 的 B1/B8 和同一组三个种子另跑窄树，保留深度 4，将 top-k 降为 1、最多候选降为 4。吞吐变化均按各配置所在批次的无投机对照配对计算。

| B | 树配置 | 平均候选数的中位值 | 有效验证行 | 目标 decode 前向数 | EAGLE-3 吞吐 (tok/s) | 相对各自基线 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | top-k 2，最多 8 | 7.33 | 31.0% | 12 | 69.6 | −12.1% |
| 1 | top-k 1，最多 4 | 3.58 | 45.6% | 15 | 30.4 | −28.9% |
| 8 | top-k 2，最多 8 | 7.49 | 25.7% | 20 | 137.6 | −60.8% |
| 8 | top-k 1，最多 4 | 3.63 | 39.1% | 21 | 114.4 | −79.7% |

窄树提高验证行利用比例，同时增加目标前向次数。六个窄树配对均慢于各自的基线。两种配置在不同轮次测量，吞吐差异受测量波动影响。 [窄树 JSON](../results/eagle_comparison_20261118/narrow.json) · [窄树 CSV](../results/eagle_comparison_20261118/narrow.csv)

Seed 17 的独立分段对照中，B1 的草稿区间由 148 降至 72 ms，目标 spec + mixed 由 264 降至 195 ms。B8 的目标 spec + mixed 由 511 降至 361 ms，草稿区间仍为约 800 ms（宽树 808 ms，窄树 800 ms）。这是单种子的分段诊断。 [窄树分段 JSON](../results/eagle_comparison_20261118/narrow_phases.json)

## Greedy 输出一致性

对每个请求比较完整的 32 个输出 token。三种 EAGLE 在重复文本上都是 123/123。ShareGPT 分别为 119/123（96.7%）、100/123（81.3%）、98/123（79.7%）。总计分别为 242/246、223/246、221/246。该指标衡量与无投机解码的逐 token 一致性。

ShareGPT 的 9 组输入在独立诊断中复跑，四种模式各记录 3,936 个输出位置的 top-2 logits。三代分别有 7、21、23 个请求分歧，两侧首个分歧位置的 top-1 与 top-2 差距均不超过 0.25 logit。49/51 处的两个候选相同。B1 seed 42 的第 28 个 token 上，无投机的 token 653 和 1598 同为 20.75，EAGLE-3 对两者分别给出 20.625 和 20.875。无投机自身在主测和诊断复跑之间也有 24/123 个请求的输出不同。结果支持首个分歧发生在 BF16 近并列位置，具体算子来源尚未定位。所有诊断行均已核对，记录到的候选包含该行实际输出。 [诊断 JSON](../results/eagle_comparison_20261118/logits.json)

`--eagle-pytest` 正式复跑通过 56 项测试。其中完整 Qwen3 树验证的未扰动兄弟路径保持逐位相同，独立串行路径对照的 logits 和提交后 KV 相对 L2 误差低于 2.5%，测试行 argmax 相同。EAGLE-3 检查点的增量/KV 与稠密参考测试也通过。

## 原始数据与复现

- 本地 EAGLE-1/2 权重 SHA-256 为 `38eec801724345adcc16d21d8d523cee8b1f553145e96a20a35ff92a56f0601a`。EAGLE-3 的 `model.safetensors` SHA-256 为 `99863f46e5cf6bb0cccecad1099ef0609a3424a142965413ad3cf9a411810860`。
- 重复文本取 `style=repeat` 的全部 9 组。首次执行的 256 上下文上限容纳不下部分 ShareGPT prompt，文件因此保留 `running` 状态。该文件中的 ShareGPT 试跑行不纳入上述汇总。 [JSON](../results/eagle_comparison_20261118/paired.json) · [CSV](../results/eagle_comparison_20261118/paired.csv)
- ShareGPT 在 1024 上下文上限下独立执行完 9 组。原始 JSON 附每对输入指纹、完整生成 token ID、首个分歧和性能指标。 [JSON](../results/eagle_comparison_20261118/paired_sharegpt.json) · [CSV](../results/eagle_comparison_20261118/paired_sharegpt.csv)

```bash
trace=results/supplemental/sharegpt_heldout_chat_template_100_20261016.jsonl
common=(--batch-sizes 1,8,32 --seeds 17,42,99 --sampling-mode greedy --cold-prefix
  --min-input-len 64 --max-input-len 192 --max-output-len 32 --gpu-memory-utilization .5)
modes=none,eagle,eagle2,eagle3

bash benchmarks/run_qwen_cuda.sh --eagle-pytest
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --speculative "$modes" \
  --styles repeat --max-model-len 256 --output results/eagle_repeat_repro.json
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --speculative "$modes" \
  --styles sharegpt --prompts-file "$trace" --max-model-len 1024 \
  --output results/eagle_sharegpt_repro.json
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --speculative "$modes" \
  --styles sharegpt --prompts-file "$trace" --max-model-len 1024 --logit-diagnostics \
  --output results/eagle_logits_repro.json
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --speculative "$modes" \
  --styles repeat,sharegpt --prompts-file "$trace" --max-model-len 1024 --seeds 17 \
  --phase-timing --output results/eagle_phases_repro.json
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --speculative none,eagle3 \
  --styles sharegpt --prompts-file "$trace" --max-model-len 1024 --batch-sizes 1,8 \
  --eagle-tree-top-k 1 --eagle-tree-max-nodes 4 --output results/eagle_narrow_repro.json
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --speculative none,eagle3 \
  --styles sharegpt --prompts-file "$trace" --max-model-len 1024 --batch-sizes 1,8 \
  --eagle-tree-top-k 1 --eagle-tree-max-nodes 4 --seeds 17 --phase-timing \
  --output results/eagle_narrow_phases_repro.json
```

## EAGLE-2/3 批量建树与分页草稿 KV

每个深度把所有活跃请求的前沿拼成一个草稿批次。候选 token、累计 log probability、父节点和最终保留集合都在 GPU 上确定。每轮结束才把树交给 CPU 调度器。单链直接用普通因果验证。EAGLE-2 使用验收路径的最后一个节点作为下一轮草稿输入。

EAGLE-3 的草稿 KV 池复用目标块表的物理页编号，显存预算按目标和草稿每页合计计算。已提交的前缀 KV 只写一份，分支注意力通过索引读取前缀和祖先，每层新候选各写一行临时 KV。目标验证后，用接受路径上的目标特征追加正式草稿 KV。被抢占的请求从 prefill 重建草稿 KV。

复用上一轮 ShareGPT held-out 的 B8/B32、seed 17/42/99、BF16、单卡、eager、greedy、固定 32 输出 token、50% 显存预算和 1024 上下文上限。宽树为深度 4、宽度 2、最多 8 个候选。窄树为深度 4、宽度 1、最多 4 个候选。各配置都独立配对无投机，原始数据包含端到端吞吐、TTFT、TPOT、目标前向次数、候选接受数、输出 token 和输入指纹。

固定宽树的 EAGLE-2/3 在 ShareGPT B32 分别达到 916.4、1350.5 tok/s。原实现的同指纹数字分别为 207.7、180.5 tok/s。两次测量的无投机吞吐也有变化。EAGLE-2 同时修正了接受节点的索引，两轮差值包含这个行为变化。

窄树的 B32 三种子中位有效验证行比例分别为 EAGLE-2 的 33.1%、EAGLE-3 的 38.8%，宽树为 21.0%、26.5%。EAGLE-2 在 B32 的请求级 greedy 一致数从宽树的 79/96 变为窄树的 96/96，EAGLE-3 从 79/96 变为 95/96。两种验证形状分别与各自同轮无投机做比较。 [宽树数据](../results/eagle_batched_20261119/wide.json) · [窄树数据](../results/eagle_batched_20261119/narrow_linear.json)

| B | 配置 | 无投机 tok/s | EAGLE-2 tok/s | EAGLE-3 tok/s |
| ---: | --- | ---: | ---: | ---: |
| 8 | 宽树 | 171.0 | 294.0 (+59.1%) | 229.6 (+7.1%) |
| 8 | 窄树，因果验证 | 551.1 | 460.2 (−8.1%) | 402.4 (−18.6%) |
| 8 | 自适应 | 410.9 | 349.4 (−15.0%) | 323.2 (−21.3%) |
| 32 | 宽树 | 2179.2 | 916.4 (−56.1%) | 1350.5 (−41.3%) |
| 32 | 窄树，因果验证 | 1849.6 | 1256.5 (−32.1%) | 1548.0 (−22.4%) |
| 32 | 自适应 | 1284.7 | 1923.8 (+6.2%) | 1512.1 (+17.7%) |

吞吐列是三个种子的中位数，括号取同 seed 相对各行基线的变化中位数。B8 无投机基线跨配置有明显波动。B32 自适应各赢两组，EAGLE-2 的范围为 −5.9% 至 +57.8%，EAGLE-3 为 −50.9% 至 +45.6%。B32 自适应时 EAGLE-2/3 分别有 95/96、96/96 个请求与同轮无投机逐 token 一致。 [自适应数据](../results/eagle_batched_20261119/adaptive.json)

`eagle_tree_policy="adaptive"` 按活跃请求数分到小于 8、8–31、至少 32 三档。默认探测窄树，最大深度分别为 3、3、2，受用户配置上限约束。每档更新近期每请求接受的草稿数，EWMA 当前样本权重为 0.25。至少 8 个活跃请求且平均接受不足 0.6 个草稿时，暂停该轮投机，每八轮探测一次。小于 32 个请求且平均接受至少 1.25 个草稿时，可用配置的宽树。B32 本轮测量中 EAGLE-2/3 分别有 17–24、24–25 次暂停草稿的决策。这些是实验阈值，API 默认仍使用 `"fixed"`。

单独用 seed 17 采集宽树的 CUDA Event 区间。B8 的 EAGLE-2 草稿从 763.5 降至 196.0 ms，EAGLE-3 从 807.8 降至 367.4 ms。B32 的 EAGLE-2 从 4436.9 降至 250.5 ms，EAGLE-3 从 3895.6 降至 499.7 ms。B32 的目标 spec/decode 加 mixed 前向区间为 EAGLE-2 的 830.9 ms、EAGLE-3 的 455.5 ms、无投机的 462.1 ms。此时 EAGLE-2 仍付出更多目标验证时间，EAGLE-3 的目标时间接近无投机且另有草稿成本。CUDA Event 区间可包含设备等待主机提交的时间。 [分段数据](../results/eagle_batched_20261119/wide_phases.json) · [上一轮分段数据](../results/eagle_comparison_20261118/phases.json)

B32 seed 17 的自适应诊断在两种模式中均采到了普通 `target_decode` 前向。暂停建树时，EAGLE-3 仍要将接受的目标特征投影写入分页草稿 KV，供后续探测使用。这次诊断的草稿区间合计为 EAGLE-2 的 25.7 ms 和 EAGLE-3 的 504.2 ms，包含各自的提交与调度开销。 [自适应分段数据](../results/eagle_batched_20261119/adaptive_phases.json)

RTX 5060 Ti 上的 EAGLE、DFlash、投机验收和默认后端回归共 63 项通过。批量树与串行参考比较了 EOS、不同深度和预算。EAGLE-3 的分页根节点与两层分支和稠密 BF16 参考的隐藏状态相对 L2 差异均低于 1.2%，且分支前向保持已提交前缀逐位相同。

### 批量建树复现

```bash
trace=results/supplemental/sharegpt_heldout_chat_template_100_20261016.jsonl
common=(--speculative none,eagle2,eagle3 --styles sharegpt --prompts-file "$trace"
  --batch-sizes 8,32 --seeds 17,42,99 --sampling-mode greedy --cold-prefix
  --max-draft-len 4 --max-output-len 32 --min-input-len 64 --max-input-len 192
  --max-model-len 1024 --gpu-memory-utilization .5)
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --output results/eagle_wide_repro.json
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --eagle-tree-top-k 1 \
  --eagle-tree-max-nodes 4 --output results/eagle_linear_repro.json
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --eagle-tree-policy adaptive \
  --output results/eagle_adaptive_repro.json
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --seeds 17 --phase-timing \
  --output results/eagle_batched_phases_repro.json
bash benchmarks/run_qwen_cuda.sh --eagle-study "${common[@]}" --batch-sizes 32 --seeds 17 \
  --eagle-tree-policy adaptive --phase-timing --output results/eagle_adaptive_phases_repro.json
```
