from pathlib import Path

import pytest
import torch
from torch import nn


def test_markov_chain_uses_preceding_target_token():
    from nanovllm.layers.dspark import ConfidenceHead, DSparkDraft, MarkovHead

    with torch.no_grad():
        draft = DSparkDraft.__new__(DSparkDraft)
        nn.Module.__init__(draft)
        draft.lm_head = nn.Linear(1, 2, bias=False)
        draft.lm_head.weight.zero_()
        draft.markov_head = MarkovHead(5, 2, 1)
        draft.markov_head.markov_w1.weight.copy_(torch.tensor([[1.], [1.], [0.], [1.], [-1.]]))
        draft.markov_head.markov_w2.weight.copy_(torch.tensor([[-1.], [1.]]))
        draft.confidence_with_markov = True
        draft.confidence_head = ConfidenceHead(2)
        draft.confidence_head.proj.weight.copy_(torch.tensor([[0., 1.]]))
        draft.confidence_head.proj.bias.zero_()
        draft.register_buffer("target_ids", torch.tensor([1, 4]))
        block = draft._markov_proposals(torch.tensor([1, 4]), torch.zeros(2, 3, 1))
    assert block[..., 0].tolist() == [[4, 1, 4], [1, 4, 1]]
    torch.testing.assert_close(block[0, :, 1], torch.tensor([1., -1., 1.]).sigmoid())


def test_confidence_scheduler_accounts_for_prefixes_and_per_request_caps():
    from nanovllm.layers.dspark import DSparkDraft

    draft = DSparkDraft.__new__(DSparkDraft)
    proposals = [[[7, .9], [8, .9], [9, 1.]], [[10, .7], [11, .1], [12, .9]]]
    assert draft.select_prefixes(proposals, [2, 3]) == [[7, 8], [10]]
    draft.step_ms = (1., 1., 1.1, 1.2, 2., 2.1, 2.2)
    assert draft.select_prefixes(proposals, [1, 3]) == [[7], [10]]
    draft.step_ms = (1., 1., 1.1, 1.2, 1.3, 1.4, 1.5)
    assert draft.select_prefixes(proposals, [2, 3]) == [[7, 8], [10]]
    assert draft.select_prefixes(proposals, [2, 3], eos=7) == [[], [10]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_causal_sliding_paged_block_matches_dense_attention():
    from nanovllm.layers.dflash import DFlashDecoderLayer
    from nanovllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb

    torch.manual_seed(17)
    layer = DFlashDecoderLayer(
        dict(hidden_size=32, intermediate_size=64, num_attention_heads=2,
             num_key_value_heads=1, head_dim=16, rms_norm_eps=1e-6),
        causal=True, window_size=(7, 0),
    ).cuda().bfloat16()
    k_cache = torch.randn(2, 256, 1, 16, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([255], dtype=torch.int32, device="cuda")
    positions = torch.arange(255, 263, device="cuda")
    rope = RotaryEmbedding(16, 16, 512, 1000000).cuda().cos_sin_cache[positions].chunk(2, dim=-1)
    hidden = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)

    q, k, v = layer.qkv(layer.input_layernorm(hidden)).split((32, 16, 16), dim=-1)
    q = apply_rotary_emb(layer.q_norm(q.reshape(-1, 2, 16)), *rope)
    k = apply_rotary_emb(layer.k_norm(k.reshape(-1, 1, 16)), *rope).squeeze(1)
    keys = torch.cat((k_cache[0, :255, 0], k))
    values = torch.cat((v_cache[0, :255, 0], v))
    distance = positions[:, None] - torch.arange(263, device="cuda")[None]
    scores = (torch.einsum("qhd,kd->hqk", q.float(), keys.float()) / 4).masked_fill(
        (distance < 0) | (distance > 7), float("-inf"))
    attention = torch.einsum("hqk,kd->qhd", scores.softmax(-1), values.float())
    residual = hidden + layer.o_proj(attention.flatten(1).bfloat16())
    gate, up = layer.gate_up(layer.post_attention_layernorm(residual)).chunk(2, dim=-1)
    expected = residual + layer.down(torch.nn.functional.silu(gate) * up)
    actual = layer(hidden, rope, k_cache, v_cache, table, lengths)
    torch.testing.assert_close(actual.float(), expected.float(), atol=.015, rtol=.015)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pinned_dspark_checkpoint_batches_chunked_mixed_generation(monkeypatch):
    import gc

    from nanovllm import LLM, SamplingParams

    target = Path.home() / "huggingface/Qwen3-0.6B"
    drafter = Path("results/dspark_qwen3_0.6b/verified")
    if not ((target / "config.json").is_file() and (drafter / "model.safetensors").is_file()):
        pytest.skip("local target and pinned DSpark checkpoint required")
    gc.collect()
    torch.cuda.empty_cache()
    llm = LLM(str(target), speculative="dspark", dspark_path=str(drafter),
              dspark_step_ms=(1.,) * 20, max_model_len=512, max_num_batched_tokens=256,
              max_num_seqs=4, gpu_memory_utilization=.5, latency_aware_scheduling=False,
              slo_aware_scheduling=False, multi_step_decode=False)
    try:
        llm.model_runner.sampler.forward = lambda logits, _: logits.argmax(dim=-1)
        draft = llm.model_runner.dflash
        assert draft.first_draft_slot == 0
        assert draft.embed_tokens.weight.data_ptr() == llm.model_runner.model.model.embed_tokens.weight.data_ptr()
        original = draft.forward
        batches = []

        def record(anchors, lengths, table, cache, depth):
            batches.append(anchors.numel())
            block = original(anchors, lengths, table, cache, depth)
            assert block.shape == (anchors.numel(), depth, 2)
            return block

        monkeypatch.setattr(draft, "forward", record)
        params = SamplingParams(max_tokens=12, ignore_eos=True)
        llm.add_request([1045, 295, 1947] * 100, params)
        llm.step()
        assert 0 < llm.scheduler.waiting[0].num_cached_tokens < 300
        assert not batches
        llm.step()
        llm.add_request([1045, 295] * 8, params)
        assert llm.step()[1] == "mixed"
        for _ in range(24):
            if llm.is_finished():
                break
            llm.step()
        assert llm.is_finished() and 2 in batches
        assert llm.scheduler.block_manager.no_share and not llm.scheduler.kv_swap
    finally:
        llm.exit()
        gc.collect()
        torch.cuda.empty_cache()
