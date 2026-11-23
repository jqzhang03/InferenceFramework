"""Correctness coverage for the route-major packed-INT4 MoE CUDA path."""
from __future__ import annotations

import pytest
import torch

from nanovllm.layers.cuda_int4 import install_cuda_int4
from nanovllm.layers.moe import MoE


def _quantized_moe(*, top_k: int = 2, experts: int = 8) -> MoE:
    torch.manual_seed(23)
    moe = MoE(128, 256, num_experts=experts, top_k=top_k,
              int4_backend="compact").cuda().to(torch.bfloat16)
    with torch.no_grad():
        for param in moe.parameters():
            param.uniform_(-0.05, 0.05)
    for expert in moe.experts:
        for linear in (expert.gate_proj, expert.up_proj, expert.down_proj):
            linear.quantize_int4(dense_path=False, group_size=64)
    install_cuda_int4(moe)
    return moe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("top_k", [1, 2, 6])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 8, 9, 16, 17, 32])
def test_compact_int4_matches_per_expert_loop(top_k: int, tokens: int):
    moe = _quantized_moe(top_k=top_k)
    x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        moe.int4_backend = "loop"
        expected = moe._forward_loop(x)
        moe.int4_backend = "compact"
        actual = moe(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compact_int4_has_no_host_sync_after_warmup():
    moe = _quantized_moe(top_k=6)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        moe(x)  # pointer-table initialization
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            moe(x)
        finally:
            torch.cuda.set_sync_debug_mode("default")
