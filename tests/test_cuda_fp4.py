import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.cuda_int4 import cuda_fp4_gemm
from nanovllm.layers.linear import (
    dequantize_mxfp4_tensor,
    dequantize_nvfp4_tensor,
    quantize_mxfp4_tensor,
    quantize_nvfp4_tensor,
)


def test_mxfp4_scale_and_ties():
    weights = torch.zeros(2, 64)
    weights[0, :8] = torch.tensor([6, -6, 0.25, 0.75, 1.25, 1.75, 2.5, -3.5])
    weights[0, 32] = 2
    packed, scales = quantize_mxfp4_tensor(weights)
    assert scales.tolist() == [[127, 126], [127, 127]]
    assert packed[0, :4].tolist() == [0xF7, 0x20, 0x42, 0xE4]
    restored = dequantize_mxfp4_tensor(packed, scales, 64)
    assert restored[0, :8].tolist() == [6, -6, 0, 1, 1, 2, 2, -4]
    assert restored[0, 32] == 2
    assert not restored[1].count_nonzero()
    weights[1, 0] = torch.inf
    packed, scales = quantize_mxfp4_tensor(weights)
    assert scales[1, 0] == 255
    assert dequantize_mxfp4_tensor(packed, scales, 64)[1, :32].isnan().all()


@pytest.fixture
def gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires SM80+ CUDA")
    return torch.device("cuda")


@pytest.mark.parametrize("group", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp4_cuda_dense(gpu, group, dtype):
    torch.manual_seed(9124)
    # N and K exercise column and shared-memory tile tails.
    weight = torch.randn((67, 160), device=gpu, dtype=dtype) / 8
    if group == 16:
        packed, scales, global_scale = quantize_nvfp4_tensor(weight)
        dense = dequantize_nvfp4_tensor(packed, scales, global_scale, 160).to(dtype)
    else:
        packed, scales = quantize_mxfp4_tensor(weight)
        global_scale = None
        dense = dequantize_mxfp4_tensor(packed, scales, 160).to(dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for rows in (0, 1, 2, 3, 4, 5, 17, 67):
            x = torch.randn((160, rows), device=gpu, dtype=dtype).T
            actual = cuda_fp4_gemm(x, packed, scales, global_scale, group)
            torch.testing.assert_close(actual, F.linear(x.contiguous(), dense), atol=.017, rtol=.008)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
def test_fp4_cuda_moe_replays_changed_routes(gpu, quantization):
    from nanovllm.layers.moe import MoE

    torch.manual_seed(432)
    moe = MoE(64, 96, 8, 2).to(device=gpu, dtype=torch.bfloat16)
    with torch.no_grad():
        moe.gate.weight.normal_(std=.2)
        for expert in moe.experts:
            for layer in (expert.gate_proj, expert.up_proj, expert.down_proj):
                layer.weight.normal_(std=.2)
                if quantization == "nvfp4":
                    layer.quantize_nvfp4(backend="cuda")
                else:
                    layer.quantize_mxfp4()
        moe.prepare_cuda_fp4(quantization)
        x = torch.randn((23, 64), device=gpu, dtype=torch.bfloat16)
        replacement = torch.randn_like(x)
        assert not torch.equal(moe._route(x)[1], moe._route(replacement)[1])
        for sizes in (1, 4, 23):
            torch.testing.assert_close(moe(x[:sizes]), moe._forward_loop(x[:sizes]),
                                       atol=.025, rtol=.015)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = moe(x)
        x.copy_(replacement)
        graph.replay()
        torch.testing.assert_close(replayed, moe(replacement), atol=0, rtol=0)
