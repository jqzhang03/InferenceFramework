import math
from types import SimpleNamespace

import pytest
import torch

from nanovllm.layers import attention_mla
from nanovllm.layers.attention_mla import mla_decode_attention
from nanovllm.layers.cuda_mla_decode import (
    CUDA_MLA_AUTO_MAX_ROWS, cuda_mla_decode_attention, load_cuda_mla_decode,
)


@pytest.fixture(scope="module")
def extension():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires SM80+ CUDA")
    ext = load_cuda_mla_decode(torch.device("cuda"), required=True)
    ext.prepare_async(torch.cuda.current_device())
    return ext


def make_inputs(lengths, dtype, page_size=256):
    batch = len(lengths)
    num_pages = max(1, math.ceil(max(lengths) / page_size))
    pages = torch.randperm(batch * num_pages, device="cuda", dtype=torch.int32).view(batch, num_pages)
    cache = torch.randn(batch * num_pages, page_size, 576, device="cuda").to(dtype)
    query = torch.randn(batch, 16, 576, device="cuda", dtype=torch.float16)
    q_abs, q_pe = query.split((512, 64), dim=-1)
    lengths_cuda = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return q_abs, q_pe, cache, pages, lengths_cuda


def reference(q_abs, q_pe, cache, pages, lengths, softmax_scale, c_scale, r_scale):
    query = torch.cat((q_abs, q_pe), dim=-1).float()
    outputs = []
    for i, length in enumerate(lengths.tolist()):
        if not length:
            outputs.append(query.new_zeros((16, 512)))
            continue
        rows = cache.float().index_select(0, pages[i].long()).reshape(-1, 576)[:length]
        c, r = rows.split((512, 64), dim=-1)
        c, r = (c * c_scale).half().float(), (r * r_scale).half().float()
        score = ((query[i, :, :512] @ c.T + query[i, :, 512:] @ r.T) * softmax_scale)
        outputs.append(score.softmax(dim=-1) @ c)
    return torch.stack(outputs).half()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
def test_decode_paged_cache(extension, dtype):
    torch.manual_seed(42)
    args = make_inputs([1, 15, 16, 17, 127, 256, 257, 1093], dtype)
    scales = (192**-0.5, .5, 1.5) if dtype == torch.float8_e4m3fn else (192**-0.5, 1., 1.)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = cuda_mla_decode_attention(*args, *scales, extension)
        expected = reference(*args, *scales)
        triton = mla_decode_attention(*args, *scales)
        torch.testing.assert_close(actual, expected, atol=1.8e-3, rtol=.008)
        torch.testing.assert_close(actual, triton, atol=2e-3, rtol=.008)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("variant", ["auto", "padded", "registers", "async"])
def test_decode_graph_changes_paging_and_lengths(extension, variant):
    torch.manual_seed(33)
    args = make_inputs([0, 1, 257, 63], torch.bfloat16)
    scales = (192**-0.5, 1., 1.)
    cuda_mla_decode_attention(*args, *scales, extension, variant=variant)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = cuda_mla_decode_attention(*args, *scales, extension, variant=variant)
    graph.replay()
    torch.testing.assert_close(out, reference(*args, *scales), atol=1.8e-3, rtol=.008)

    q_abs, q_pe, cache, pages, lengths = args
    q_abs.copy_(torch.randn_like(q_abs))
    q_pe.copy_(torch.randn_like(q_pe))
    cache.copy_(torch.randn_like(cache))
    pages.copy_(pages.flip((0, 1)))
    lengths.copy_(torch.tensor([489, 0, 15, 1], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.testing.assert_close(out, reference(*args, *scales), atol=1.8e-3, rtol=.008)
    assert not out[1].count_nonzero()


def test_bf16_prefetch_preserves_accumulation_and_tail(extension):
    torch.manual_seed(23)
    args = make_inputs([0, 1, 31, 33, 256, 257, 777, 4096], torch.bfloat16)
    scales = (192**-0.5, .75, 1.25)
    want = cuda_mla_decode_attention(*args, *scales, extension, variant="original")
    for variant in ("padded", "registers", "async", "auto"):
        got = cuda_mla_decode_attention(*args, *scales, extension, variant=variant)
        torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_decode_dispatch_auto_and_explicit_cuda(monkeypatch):
    calls = []

    def record(name):
        def decode(q, *_):
            calls.append(name)
            return q
        return decode

    monkeypatch.setattr(attention_mla, "get_context",
                        lambda: SimpleNamespace(block_tables=None, context_lens=None))
    monkeypatch.setattr(attention_mla, "mla_decode_attention", record("triton"))
    monkeypatch.setattr(attention_mla, "cuda_mla_decode_attention", record("cuda"))
    weights = torch.ones(16, 1, 1)
    attn = SimpleNamespace(_wuk_t=weights, _wuv_t=weights, num_heads=16, v_head_dim=1,
                           mla_cache=None, scaling=1., mla_c_scale=1., mla_r_scale=1.,
                           _cuda_mla_decode_ext=object(),
                           _cuda_mla_decode_max_rows=CUDA_MLA_AUTO_MAX_ROWS)
    for rows in (16, 32):
        q = torch.ones(rows, 16, 1)
        attention_mla.MLAAttention._decode_kernel(attn, q, q)
    attn._cuda_mla_decode_max_rows = None
    attention_mla.MLAAttention._decode_kernel(attn, q, q)
    assert calls == ["cuda", "triton", "cuda"]
