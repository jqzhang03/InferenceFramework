# Qwen3-0.6B DFlash 与 EAGLE 配对测试

## 草稿与验证

DFlash 从 Qwen3-0.6B 的残差流取检查点指定的三组特征。特征融合后，批量投影成各草稿层的 K/V 并写到独立分页池。草稿用一个已输出的 anchor token 加七个 mask，执行一次批量双向 attention。对照最多送四个候选给目标模型。目标走现有因果验证路径，最长匹配前缀和补偿 token 都按目标模型的采样结果提交。分页草稿缓存只提交已验证的目标特征，临时草稿块跨页时使用独立暂存页。DFlash 使用私有前缀块，抢占后重算。 [DFlash 论文](https://arxiv.org/abs/2602.06036) · [运行代码](../nanovllm/layers/dflash.py)

使用 `arnabwithab/Qwen3-0.6B-DFlash-noshift-b8` 的 3 层、32,000 草稿词表权重，revision `b4a61819293fdb9106e648b0422c38dbf90bb3f4`。本地 `model.safetensors` 已核验 SHA-256 为 `5084f318a312a5aa8c999caa6b86bd644d3fa7846e7f0c45c545009fa852ea68`。三代 EAGLE 沿用上一轮的权重及验证配置。 [固定版本草稿](https://huggingface.co/arnabwithab/Qwen3-0.6B-DFlash-noshift-b8/commit/b4a61819293fdb9106e648b0422c38dbf90bb3f4) · [上一轮 EAGLE 报告](_eagle_comparison_20261119.md)

## 测试条件

RTX 5060 Ti、TP=1、BF16、eager、默认 `cuda_decode`、普通分页 KV，显存预算 50%，关闭跨请求前缀复用。每个工作负载测试 B=1/8/32、seed 17/42/99，greedy 忽略 EOS，每请求生成 32 token。重复文本上下文上限 256，ShareGPT held-out 上下文上限 1024。EAGLE-1 最大草稿长度 4，EAGLE-2/3 树深 4、top-k 2、最多 8 个候选节点，DFlash 最多验证 4 个候选。

重复文本试跑时，共用 RMSNorm 编译入口因新增草稿形状达到 PyTorch 重编译上限。正式对照给 DFlash 的 RMSNorm 和 SwiGLU 使用独立编译入口，重新运行五种模式。试跑产物 `paired_repeat.json` 保留用于核对，正式重复文本数据为 `paired_repeat_final.json`。

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
```
