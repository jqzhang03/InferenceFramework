import gc
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.eagle_tree import TreeAttentionBatch, build_tree, tree_depths, tree_mask, verify_tree


def test_dynamic_tree_has_ancestor_closed_budget():
    def advance(nodes, parents, tokens):
        return torch.tensor([[0., 1., 8.] if token == 0 else [9., 1., 0.] for token in tokens])

    drafts, parents = build_tree(torch.tensor([2.0, 1.9, -5.0]), advance, None, 2, 3, 4, 9)
    assert 0 < len(drafts) <= 4
    assert len(set(zip(parents, drafts))) == len(drafts)
    assert max(tree_depths(parents)) <= 3
    mask = tree_mask(parents)
    assert all(row[i] and row[0] and not any(row[i + 1:]) for i, row in enumerate(mask))


def test_tree_accepts_only_target_sampled_path():
    drafts = [10, 20, 11, 21]
    parents = [0, 0, 1, 2]
    assert verify_tree(drafts, parents, [20, 11, 21, 99, 88]) == ([20, 21, 88], [0, 2, 4])
    assert verify_tree(drafts, parents, [10, 21, 21, 99, 88]) == ([10, 21], [0, 1])
    assert verify_tree(drafts, parents, [7, 11, 21, 99, 88]) == ([7], [0])
    assert verify_tree(drafts, parents, [20, 11, 21, 99, 88], eos=20) == ([20], [0])


def test_tree_sampling_preserves_target_paths():
    # Enumerate the independent draws at three target rows. The unused
    # rows must integrate out, leaving the baseline distribution at the root.
    from itertools import product

    outcomes = {}
    distributions = ((0.3, 0.7), (0.8, 0.2), (0.4, 0.6))
    for a, b, c in product(range(2), repeat=3):
        tokens, path = verify_tree([0, 1], [0, 0], [a, b, c])
        assert path == [0, a + 1]
        key = tuple(tokens)
        outcomes[key] = outcomes.get(key, 0) + distributions[0][a] * distributions[1][b] * distributions[2][c]
    for a, b in product(range(2), repeat=2):
        assert outcomes[a, b] == pytest.approx(distributions[0][a] * distributions[a + 1][b])


def test_eagle_uses_predecessor_hidden_after_first_rejection():
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams

    prefilling = Sequence([0])
    rejected = Sequence([2], SamplingParams(max_tokens=4))
    rejected.is_prefill = False
    rejected.append_token(0)
    model = SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda x: x),
                            lm_head=SimpleNamespace(weight=torch.eye(3)))
    runner = SimpleNamespace(model=model, eagle_layer=lambda h, _: h)
    engine = SimpleNamespace(model_runner=runner,
                             config=SimpleNamespace(max_draft_len=1, eos=99, speculative="eagle"))
    hidden = torch.eye(3)
    LLMEngine._eagle_drafts(engine, [prefilling, rejected], hidden, [0, 1], [0, 2], prefill_tokens=1)
    assert rejected.draft_tokens == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tree_attention_matches_individual_branches_and_kv_compaction():
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.layers.attention import Attention

    torch.manual_seed(241)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    attention = Attention(4, 128, 128 ** -.5, 2)
    cache = torch.randn(2, 1, 2, 256, 2, 128, device=device, dtype=dtype)
    attention.k_cache, attention.v_cache = cache[0, 0], cache[1, 0]
    q = torch.randn(5, 4, 128, device=device, dtype=dtype)
    k = torch.randn(5, 2, 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    mask = tree_mask([0, 0, 1, 2])
    ctx = SimpleNamespace(tree_attention=TreeAttentionBatch(
        torch.arange(5, device=device)[None], torch.arange(5, device=device),
        torch.tensor([mask], device=device), torch.tensor([257], device=device, dtype=torch.int32),
        torch.tensor([[1, 0]], device=device, dtype=torch.int32), 0, 0))
    actual = attention._tree_rows(q, k, v, ctx)
    pk = torch.cat((attention.k_cache[1], attention.k_cache[0, :1]))
    pv = torch.cat((attention.v_cache[1], attention.v_cache[0, :1]))
    for i, allowed in enumerate(mask):
        indices = [j for j, keep in enumerate(allowed) if keep]
        kk = torch.cat((pk, k[indices])).repeat_interleave(2, dim=1).float()
        vv = torch.cat((pv, v[indices])).repeat_interleave(2, dim=1).float()
        reference = torch.einsum("nh,hnd->hd", (
            (kk * q[i].float()).sum(-1) * attention.scale).softmax(0), vv.transpose(0, 1))
        torch.testing.assert_close(actual[i].float(), reference, atol=0.008, rtol=0.015)

    from nanovllm.engine.sequence import Sequence
    seq = Sequence([1] * 256)
    seq.block_table = [1, 0]
    seq.num_scheduled_tokens = 5
    span = [511, 0, 1, 2, 3]
    before = cache.flatten(2, 3).clone()
    runner = SimpleNamespace(kv_cache=cache, block_size=256)
    runner._span_slots = ModelRunner._span_slots.__get__(runner)
    ModelRunner.compact_tree_kv(runner, [(seq, [0, 2, 4])])
    torch.testing.assert_close(cache.flatten(2, 3)[:, :, span[:3]], before[:, :, [511, 1, 3]], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_qwen_tree_logits_and_committed_kv_match_linear_verification():
    from pathlib import Path

    from nanovllm import LLM
    from nanovllm.engine.sequence import Sequence
    from nanovllm.utils.context import reset_context

    checkpoint = Path.home() / "huggingface/Qwen3-0.6B"
    if not (checkpoint / "config.json").is_file():
        pytest.skip("local Qwen3 checkpoint unavailable")
    llm = LLM(str(checkpoint), enforce_eager=True, max_model_len=512,
              max_num_seqs=4, max_num_batched_tokens=512, gpu_memory_utilization=.5)
    runner = llm.model_runner
    try:
        seq = Sequence([1045, 295, 1947] * 85)
        seq.block_table = [0, 1]
        seq.num_scheduled_tokens = len(seq)
        inputs = runner.prepare_prefill([seq])
        runner.run_model(*inputs, "prefill")
        seq.num_cached_tokens = len(seq)
        seq.append_token(1045)
        seq.is_prefill = False
        draft_tokens = [293, 271, 1536, 854]
        parents = [0, 0, 1, 2]
        seq.draft_tokens = draft_tokens
        seq.draft_parents = parents
        seq.num_scheduled_tokens = 5
        actual = runner.run_model(*runner.prepare_spec([seq]), "spec").float()
        seq.draft_tokens = [999, draft_tokens[1], 1912, draft_tokens[3]]
        changed_sibling = runner.run_model(*runner.prepare_spec([seq]), "spec").float()
        torch.testing.assert_close(actual[[0, 2, 4]], changed_sibling[[0, 2, 4]], rtol=0, atol=0)
        runner.compact_tree_kv([(seq, [0, 2, 4])])
        cache = runner.kv_cache.flatten(2, 3)
        committed = cache[:, :, 255:258].clone()

        for path in ([1, 3], [2, 4]):
            seq.draft_parents = None
            seq.draft_tokens = [draft_tokens[i - 1] for i in path]
            seq.num_scheduled_tokens = 3
            expected = runner.run_model(*runner.prepare_spec([seq]), "spec").float()
            got = actual[[0, *path]]
            # This comparison changes both the BF16 attention reduction and
            # target projection shape. Sibling isolation above is bitwise.
            assert ((got - expected).norm() / expected.norm()).item() < .025
            assert (got.argmax(-1) == expected.argmax(-1)).all()
        assert ((committed.float() - cache[:, :, 255:258].float()).norm()
                / committed.float().norm()).item() < .025
    finally:
        reset_context()
        llm.exit()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_public_eagle3_checkpoint_logits_and_incremental_cache():
    from pathlib import Path
    from torch.nn import functional as F
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    from nanovllm.layers.eagle3 import Eagle3Layer

    checkpoint = Path("results/eagle3_qwen3_0.6b")
    if not (checkpoint / "model.safetensors").is_file():
        pytest.skip("public EAGLE-3 checkpoint unavailable")
    draft = Eagle3Layer.load(str(checkpoint)).cuda().bfloat16()
    torch.manual_seed(42)
    features = torch.randn(1, 13, 3072, device="cuda", dtype=torch.bfloat16)
    embeddings = torch.randn(1, 13, 1024, device="cuda", dtype=torch.bfloat16)
    whole, full_kv = draft(features, embeddings, None)
    _, past = draft(features[:, :10], embeddings[:, :10], None)
    tail, kv = draft(features[:, 10:], embeddings[:, 10:], past)
    actual = draft.logits(tail).float()
    reference = draft.logits(whole[:, 10:]).float()
    assert ((actual - reference).norm() / reference.norm()).item() < .009
    assert torch.equal(draft.target_ids[actual.argmax(-1)], draft.target_ids[reference.argmax(-1)])
    for a, b in zip(kv, full_kv):
        torch.testing.assert_close(a, b, atol=.01, rtol=.015)

    # Reference the published draft layout using HF RoPE and dense causal scores.
    def rms(x, weight):
        value = x.float()
        return (value * (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt()).to(x.dtype) * weight

    def linear(x, name):
        return F.linear(x, getattr(draft, name).weight)

    fused = linear(features, "fc")
    combined = torch.cat((rms(embeddings, draft.input_layernorm.weight),
                          rms(fused, draft.hidden_norm.weight)), dim=-1)
    q = linear(combined, "q_proj").reshape(1, 13, 16, 128).transpose(1, 2)
    k = linear(combined, "k_proj").reshape(1, 13, 8, 128).transpose(1, 2)
    v = linear(combined, "v_proj").reshape(1, 13, 8, 128).transpose(1, 2)
    positions = torch.arange(13, device=q.device)
    freq = 1 / (1e6 ** (torch.arange(0, 128, 2, device=q.device).float() / 128))
    angles = torch.outer(positions.float(), freq).repeat(1, 2)[None]
    q, k = apply_rotary_pos_emb(q, k, angles.cos().bfloat16(), angles.sin().bfloat16())
    scores = (q.float() @ k.float().repeat_interleave(2, dim=1).transpose(-1, -2)) / (128 ** .5)
    scores.masked_fill_(positions[:, None] < positions[None, :], -torch.inf)
    mixed = scores.softmax(-1) @ v.float().repeat_interleave(2, dim=1)
    residual = fused + linear(mixed.transpose(1, 2).reshape(1, 13, -1).bfloat16(), "o_proj")
    normalized = rms(residual, draft.post_attention_layernorm.weight)
    result = residual + linear(F.silu(linear(normalized, "gate_proj")) * linear(normalized, "up_proj"), "down_proj")
    expected_logits = linear(rms(result, draft.norm.weight), "lm_head").float()
    full_logits = draft.logits(whole).float()
    assert ((full_logits - expected_logits).norm() / expected_logits.norm()).item() < .009
    assert torch.equal(draft.target_ids[full_logits.argmax(-1)], draft.target_ids[expected_logits.argmax(-1)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_eagle3_chunked_prefill_and_mixed_verification():
    from pathlib import Path

    from nanovllm import LLM, SamplingParams

    target = Path.home() / "huggingface/Qwen3-0.6B"
    draft = Path("results/eagle3_qwen3_0.6b")
    if not ((target / "config.json").is_file() and (draft / "model.safetensors").is_file()):
        pytest.skip("local target and draft weights required")
    # KV sizing uses device-wide free memory, including earlier tests' reservations.
    gc.collect()
    torch.cuda.empty_cache()
    llm = LLM(str(target), speculative="eagle3", eagle_path=str(draft),
              max_model_len=512, max_num_batched_tokens=256, max_num_seqs=4,
              gpu_memory_utilization=.5, latency_aware_scheduling=False,
              slo_aware_scheduling=False, multi_step_decode=False)
    llm.model_runner.sampler.forward = lambda logits, temperatures: logits.argmax(dim=-1)
    try:
        params = SamplingParams(max_tokens=9, ignore_eos=True)
        llm.add_request([1045, 295, 1947] * 100, params)
        llm.step()
        queued = llm.scheduler.waiting[0]
        assert queued.eagle3_kv[0].size(2) == queued.num_cached_tokens < 300
        assert queued.draft_tokens is None
        llm.step()
        assert llm.scheduler.running[0].eagle3_kv[0].size(2) == 300
        llm.add_request([1045, 295] * 8, params)
        assert llm.step()[1] == "mixed"
        for _ in range(18):
            if llm.is_finished():
                break
            llm.step()
        assert llm.is_finished()
        assert queued.eagle3_kv is None
    finally:
        llm.exit()
