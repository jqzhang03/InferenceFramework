from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_dflash_bidirectional_paged_block_matches_dense_attention():
    from nanovllm.layers.dflash import DFlashDecoderLayer
    from nanovllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb

    torch.manual_seed(17)
    cfg = dict(hidden_size=32, intermediate_size=64, num_attention_heads=2,
               num_key_value_heads=1, head_dim=16, rms_norm_eps=1e-6)
    layer = DFlashDecoderLayer(cfg).cuda().bfloat16()
    for norm in (layer.input_layernorm, layer.post_attention_layernorm, layer.q_norm, layer.k_norm):
        norm.weight.fill_(1)
    q_len, page = 8, 256
    k_cache = torch.randn(5, page, 1, 16, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    tables = [[1, 3], [0, 4]]
    lengths = [255, 256]
    cache_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    block_table = torch.tensor(tables, device="cuda", dtype=torch.int32)
    positions = (cache_lengths[:, None] + torch.arange(q_len, device="cuda")).flatten()
    rope = RotaryEmbedding(16, 16, 512, 1000000).cuda().cos_sin_cache[positions].chunk(2, dim=-1)
    hidden = torch.randn(2 * q_len, 32, device="cuda", dtype=torch.bfloat16)
    original_k, original_v = k_cache.clone(), v_cache.clone()

    q, k, v = layer.qkv(layer.input_layernorm(hidden)).split((32, 16, 16), dim=-1)
    q = apply_rotary_emb(layer.q_norm(q.reshape(-1, 2, 16)), *rope)
    k = apply_rotary_emb(layer.k_norm(k.reshape(-1, 1, 16)), *rope)
    expected_attention = []
    for i, (table, length) in enumerate(zip(tables, lengths)):
        rows = slice(i * q_len, (i + 1) * q_len)
        keys = torch.cat((original_k[table].flatten(0, 1)[:length], k[rows]), dim=0)
        values = torch.cat((original_v[table].flatten(0, 1)[:length], v[rows, None]), dim=0)
        scores = torch.einsum("qhd,kd->hqk", q[rows].float(), keys[:, 0].float()) / 4
        reference = torch.einsum("hqk,kd->qhd", scores.softmax(-1), values[:, 0].float())
        expected_attention.append(reference.flatten(1))
    residual = hidden + layer.o_proj(torch.cat(expected_attention).bfloat16())
    gate, up = layer.gate_up(layer.post_attention_layernorm(residual)).chunk(2, dim=-1)
    expected = residual + layer.down(torch.nn.functional.silu(gate) * up)

    actual = layer(hidden, rope, k_cache, v_cache, block_table, cache_lengths)
    torch.testing.assert_close(actual.float(), expected.float(), atol=.015, rtol=.015)
    for i, (table, length) in enumerate(zip(tables, lengths)):
        rows = slice(i * q_len, (i + 1) * q_len)
        updated_k = k_cache[table].flatten(0, 1)
        updated_v = v_cache[table].flatten(0, 1)
        torch.testing.assert_close(updated_k[:length], original_k[table].flatten(0, 1)[:length])
        torch.testing.assert_close(updated_v[:length], original_v[table].flatten(0, 1)[:length])
        torch.testing.assert_close(updated_k[length:length + q_len], k[rows])
        torch.testing.assert_close(updated_v[length:length + q_len], v[rows, None])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dflash_checkpoint_chunked_prefill_mixed_and_batched_drafting(monkeypatch):
    import gc

    from nanovllm import LLM, SamplingParams

    target = Path.home() / "huggingface/Qwen3-0.6B"
    draft_path = Path("results/dflash_qwen3_0.6b/verified")
    if not ((target / "config.json").is_file() and (draft_path / "model.safetensors").is_file()):
        pytest.skip("local target and pinned DFlash checkpoint required")
    gc.collect()
    torch.cuda.empty_cache()
    llm = LLM(str(target), speculative="dflash", dflash_path=str(draft_path),
              max_model_len=512, max_num_batched_tokens=256, max_num_seqs=4,
              gpu_memory_utilization=.5, latency_aware_scheduling=False,
              slo_aware_scheduling=False, multi_step_decode=False)
    llm.model_runner.sampler.forward = lambda logits, temperatures: logits.argmax(dim=-1)
    draft = llm.model_runner.dflash
    proposal_batches = []
    original = draft.forward

    def forward(anchors, lengths, table, cache, draft_len):
        proposal_batches.append((anchors.numel(), draft_len))
        return original(anchors, lengths, table, cache, draft_len)

    monkeypatch.setattr(draft, "forward", forward)
    try:
        params = SamplingParams(max_tokens=12, ignore_eos=True)
        llm.add_request([1045, 295, 1947] * 100, params)
        llm.step()
        assert 0 < llm.scheduler.waiting[0].num_cached_tokens < 300
        assert not proposal_batches
        llm.step()
        llm.add_request([1045, 295] * 8, params)
        assert llm.step()[1] == "mixed"
        for _ in range(24):
            if llm.is_finished():
                break
            llm.step()
        assert llm.is_finished()
        assert any(batch == 2 for batch, _ in proposal_batches)
        assert all(1 <= length <= 4 for _, length in proposal_batches)
        assert not llm.scheduler.kv_swap and llm.scheduler.block_manager.no_share
    finally:
        llm.exit()
        gc.collect()
        torch.cuda.empty_cache()
