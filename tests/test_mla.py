"""DeepSeek-V2 / MLA CPU 单测（无需 GPU）。

覆盖：
- 结构：参数路径与 HF checkpoint 直配（self_attn.*、mlp.gate/experts.{i}.*/
  shared_experts.*、dense mlp.*）；MoE 继承布局不重复注册；
- 语义自洽：无引擎 CPU 前向（手工稠密注意力分支）确定性、无 NaN、形状正确；
- MoE routed_scaling_factor：默认 1.0 恒等、非 1 时路由概率被缩放；
- RMSNorm 非破坏性（_mla_check.py 定位的真实 bug：fp32 下 x.float() 别名输入，
  残差流中间张量被原位归一化）——CPU fp32 模型 forward 与"逐层克隆重算"一致。
"""
import torch

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.moe import MoE
from nanovllm.models.deepseek_v2 import DeepseekV2ForCausalLM


def _toy_cfg():
    from transformers import DeepseekV2Config
    cfg = DeepseekV2Config(
        vocab_size=2048,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,          # 层0 dense；层1 MoE
        num_attention_heads=16,
        first_k_dense_replace=1,
        kv_lora_rank=64,
        q_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        routed_scaling_factor=1.0,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        mlp_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=True,
    )
    return cfg


def test_deepseek_param_names_match_hf():
    """参数路径与 HF checkpoint 直配（loader 零映射、零改名）。"""
    cfg = _toy_cfg()
    m = DeepseekV2ForCausalLM(cfg)
    names = {n for n, _ in m.named_parameters()}
    want = [
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",          # dense 层
        "model.layers.1.mlp.gate.weight",               # MoE router
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.3.down_proj.weight",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
        "model.embed_tokens.weight",
    ]
    for w in want:
        assert w in names, f"缺参数 {w}"
    # MoE 继承不重复注册（mlp.moe.* 不应出现）
    assert not any(n.startswith("model.layers.1.mlp.moe") for n in names)


def test_cpu_forward_manual_dense():
    """无引擎 CPU 前向：确定性、无 NaN、行数与词表形状正确。"""
    torch.manual_seed(0)
    cfg = _toy_cfg()
    m = DeepseekV2ForCausalLM(cfg).to(torch.float32).eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2:
                p.uniform_(-0.05, 0.05)
            else:
                p.fill_(1.0)          # norm 权重（RMSNorm ×w）
    with torch.no_grad():
        ids = torch.randint(0, cfg.vocab_size, (13,))
        h1 = m.compute_logits(m(ids, torch.arange(13)))
        h2 = m.compute_logits(m(ids, torch.arange(13)))
    assert h1.shape == (13, cfg.vocab_size)
    assert torch.isfinite(h1).all()
    assert torch.allclose(h1, h2, atol=1e-4), "同权重同输入应确定"


def test_rmsnorm_no_inplace_on_input():
    """RMSNorm fp32 不得修改输入（残差流复用中间张量的正确性前提）。

    回归场景：_mla_check.py 定位——fp32 下 x.float() 别名输入，mul_ 把调用方
    的残差流张量原地归一化；bf16 路径 .float() 拷贝所以引擎无感。
    """
    torch.manual_seed(1)
    norm = RMSNorm(64, eps=1e-6)
    x = torch.randn(5, 64)
    ref = x.clone()
    with torch.no_grad():
        y = norm(x)
    assert torch.equal(x, ref), "输入被 RMSNorm 原位修改"
    # 与逐元素重算一致
    with torch.no_grad():
        var = x.float().pow(2).mean(-1, keepdim=True)
        y2 = (x.float() * torch.rsqrt(var + norm.eps)) * norm.weight
    assert torch.allclose(y.float(), y2, atol=1e-6)


def test_routed_scaling_factor():
    torch.manual_seed(2)
    x = torch.randn(7, 48)
    m1 = MoE(48, 96, num_experts=6, top_k=2)
    with torch.no_grad():
        for p in m1.parameters():
            p.uniform_(-0.05, 0.05)
    m2 = MoE(48, 96, num_experts=6, top_k=2, routed_scaling_factor=2.0)
    m2.load_state_dict(m1.state_dict())
    with torch.no_grad():
        v1, i1 = m1._route(x)
        v2, i2 = m2._route(x)
    assert torch.equal(i1, i2)
    assert torch.allclose(v2.float(), v1.float() * 2.0, atol=1e-6)
    y1 = m1(x)
    y2 = m2(x)
    # 默认 1.0 时与旧行为一致（数值同 MoE reference）
    y1r = m1.reference(x)
    assert torch.allclose(y1, y1r, atol=1e-5)
    assert not torch.allclose(y1, y2, atol=1e-3), "factor=2 应改变输出"


def _row_loop_assemble(attn, k_dense, v_dense, starts, qlens, q_off, gather_idx):
    """旧版逐行组装（5 次 host int()/行）；仅作为 `_assemble` 的位级参考。"""
    flat = attn.mla_cache.reshape(-1, attn.mla_cache.shape[-1])
    gathered = flat.index_select(0, gather_idx.long())
    gc, gk = gathered.split([attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    gk_head, gv = attn._expand(gc, gk, gc.shape[0])
    s, f = starts.long(), qlens.long()
    total = int((s + f).sum())
    k_out = torch.empty(total, *gk_head.shape[1:], dtype=k_dense.dtype)
    v_out = torch.empty(total, *gv.shape[1:], dtype=v_dense.dtype)
    seg_start = torch.cat([torch.zeros(1, dtype=torch.long),
                           torch.cumsum(s + f, 0)[:-1]])
    s_cum = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(s, 0)])
    for r in range(s.numel()):
        dst = int(seg_start[r])
        cs, ce = int(s_cum[r]), int(s_cum[r + 1])
        qs = int(q_off[r])
        qe = qs + int(f[r])
        if ce - cs:
            k_out[dst:dst + ce - cs] = gk_head[cs:ce]
            v_out[dst:dst + ce - cs] = gv[cs:ce]
        if qe - qs:
            k_out[dst + ce - cs:dst + ce - cs + qe - qs] = k_dense[qs:qe]
            v_out[dst + ce - cs:dst + ce - cs + qe - qs] = v_dense[qs:qe]
    return k_out, v_out


def test_mla_cache_assemble_matches_row_loop():
    """向量化 `_assemble` 与旧逐行循环位级相同（cache-shaped prefill / decode 兜底）。

    旧实现每行读 5 个 host 标量，cache 命中 prefill 会变成 4,160 次 4B D2H + sync
    （RTX 5060 Ti 上 310 ms GPU 空转），这里锁住替换后的数值语义。
    """
    torch.manual_seed(4)
    cfg = _toy_cfg()
    m = DeepseekV2ForCausalLM(cfg)
    attn = m.model.layers[0].self_attn
    with torch.no_grad():
        attn.kv_b_proj.weight.uniform_(-0.05, 0.05)
    blocks, block_size = 8, 16
    dim = attn.kv_lora_rank + attn.qk_rope_head_dim
    attn.mla_cache = torch.randn(blocks, block_size, dim)
    attn.mla_fp8 = False
    heads = attn.num_heads
    dk = attn.qk_nope_head_dim + attn.qk_rope_head_dim
    dv = attn.v_head_dim

    # 1) cache-shaped prefill：部分行有前缀（start>0），部分行 start=0
    starts = torch.tensor([0, 3, 0, 5, 2], dtype=torch.int32)
    qlens = torch.tensor([2, 4, 1, 2, 3], dtype=torch.int32)
    tokens = int(qlens.sum())
    q_off = torch.cat([torch.zeros(1, dtype=torch.int32),
                       torch.cumsum(qlens, 0)[:-1]])
    gather = torch.randint(0, blocks * block_size, (int(starts.sum()),),
                           dtype=torch.long)
    k_dense = torch.randn(tokens, heads, dk)
    v_dense = torch.randn(tokens, heads, dv)
    with torch.no_grad():
        got = attn._assemble(k_dense, v_dense, starts, qlens, q_off, gather)
        want = _row_loop_assemble(attn, k_dense, v_dense, starts, qlens, q_off, gather)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])

    # 2) decode 兜底：qlens 全 0、fresh 段传 [0,1,1] 空张量
    starts = torch.tensor([5, 2, 7], dtype=torch.int32)
    qlens = torch.zeros(3, dtype=torch.int32)
    q_off = torch.zeros(3, dtype=torch.int32)
    gather = torch.randint(0, blocks * block_size, (int(starts.sum()),),
                           dtype=torch.long)
    q = torch.randn(3, heads, dk)
    with torch.no_grad():
        got = attn._assemble(q, q.new_empty(0, 1, 1), starts, qlens, q_off, gather)
        want = _row_loop_assemble(attn, q, q.new_empty(0, heads, dv),
                                  starts, qlens, q_off, gather)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
