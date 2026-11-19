"""Regression tests for native W4A16 with the runtime's INT4 buffers."""
import pytest
import torch

from nanovllm.layers.cuda_int4 import (
    cuda_int4_gemm, install_cuda_int4, load_cuda_int4, set_cuda_int4_expert_fusion,
)
from nanovllm.layers.linear import ReplicatedLinear, int4_gemm
from nanovllm.layers.moe import ExpertFFN, MoE


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="requires CUDA SM80+")


@pytest.fixture(scope="module")
def extension():
    return load_cuda_int4()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,n,k,group", [(0, 33, 64, 64), (1, 33, 64, 64),
    (2, 19, 128, 128), (3, 17, 160, 32), (4, 33, 10944, 64),
    (7, 33, 320, 64), (13, 65, 256, 128), (33, 67, 192, 64)])
def test_same_packed_weights(extension, dtype, m, n, k, group):
    rng = torch.Generator(device="cuda").manual_seed(17)
    a = torch.randn((m, k), generator=rng, device="cuda", dtype=dtype)
    packed = torch.randint(-128, 128, (n, k // 2), generator=rng,
                           device="cuda", dtype=torch.int8)
    scales = torch.rand((n, k // group), generator=rng, device="cuda", dtype=dtype) / 32
    actual = cuda_int4_gemm(a, packed, scales, extension)
    assert actual.shape == (m, n) and actual.dtype == dtype
    if m:
        codes = packed.int()
        signed = torch.stack(((codes & 15) - 8, ((codes >> 4) & 15) - 8), dim=-1)
        weights = (signed.reshape(n, -1, group) * scales.float().unsqueeze(-1)).to(dtype)
        expected = (a.double() @ weights.reshape(n, k).double().T).to(dtype)
        tolerance = .008 if dtype == torch.bfloat16 else .001
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(actual, int4_gemm(a, packed, scales),
                                   atol=tolerance, rtol=tolerance)


@torch.inference_mode()
def test_linear_bias_dense_route_and_noncontiguous_fallback():
    layer = ReplicatedLinear(64, 33, bias=True).cuda().bfloat16()
    layer.weight.normal_(std=.03)
    layer.bias.normal_(std=.02)
    layer.quantize_int4(group_size=64, dense_path=False)
    x = torch.randn((64, 7), device="cuda", dtype=torch.bfloat16).T
    baseline_strided = layer(x)
    baseline_contiguous = layer(x.contiguous())
    packed_ptr = layer.w_int4.data_ptr()
    assert install_cuda_int4(layer) == 1
    assert layer.w_int4.data_ptr() == packed_ptr
    assert torch.equal(layer(x), baseline_strided)
    torch.testing.assert_close(layer(x.contiguous()), baseline_contiguous, rtol=.008, atol=.008)

    dense = ReplicatedLinear(64, 33, bias=True).cuda().bfloat16()
    dense.weight.normal_(std=.03)
    dense.bias.normal_(std=.02)
    dense.quantize_int4(group_size=64, dense_path=True)
    expected = dense(x)
    install_cuda_int4(dense)
    assert torch.equal(dense(x), expected)


@torch.inference_mode()
def test_moe_loop_reuses_packed_weights_and_skips_empty_experts():
    torch.manual_seed(88)
    moe = MoE(128, 64, num_experts=4, top_k=1).cuda().bfloat16()
    for param in moe.parameters():
        param.normal_(std=.035)
    moe.gate.weight.zero_()
    moe.gate.weight[0].fill_(1 / 128)
    x = torch.rand((3, 128), device="cuda", dtype=torch.bfloat16)
    assert (moe._route(x)[1] == 0).all().item()
    linears = [lin for triple in moe._expert_linears() for lin in triple]
    for lin in linears:
        lin.quantize_int4(group_size=64, dense_path=False)
    packed_ptrs = [lin.w_int4.data_ptr() for lin in linears]
    expected = moe(x)
    assert install_cuda_int4(moe) == len(linears)
    actual = moe(x)
    assert packed_ptrs == [lin.w_int4.data_ptr() for lin in linears]
    assert moe._grouped_ok is False
    torch.testing.assert_close(actual, expected, atol=.001, rtol=.02)


@torch.inference_mode()
@pytest.mark.parametrize("m", [3, 13])
def test_current_stream_and_changed_input_graph_replay(extension, m):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        a = torch.full((m, 64), 1, device="cuda", dtype=torch.bfloat16)
        packed = torch.full((37, 32), -103, device="cuda", dtype=torch.int8)
        scales = torch.full((37, 1), .125, device="cuda", dtype=torch.bfloat16)
        out = torch.empty((m, 37), device="cuda", dtype=torch.bfloat16)
        extension.gemm_out(a, packed, scales, out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            extension.gemm_out(a, packed, scales, out)
        a.fill_(-2)
        graph.replay()
    stream.synchronize()
    assert torch.equal(out, torch.full_like(out, -16))


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_expert_fusion_rounding_and_fallbacks(dtype):
    torch.manual_seed(21)
    expert = ExpertFFN(192, 96).to(device="cuda", dtype=dtype)
    for linear in (expert.gate_proj, expert.up_proj, expert.down_proj):
        linear.weight.uniform_(-.12, .12)
        linear.quantize_int4(group_size=32, dense_path=False)
    assert install_cuda_int4(expert) == 3
    inputs = [torch.randn(m, 192, device="cuda", dtype=dtype) for m in (0, 1, 2, 3, 4, 7)]
    inputs.append(torch.rand(192, 3, device="cuda", dtype=dtype).T)
    inputs.append(torch.rand(3 * 192 + 1, device="cuda", dtype=dtype)[1:].reshape(3, 192))
    expected = [expert(x) for x in inputs]
    for mode in ("gate_up", "full"):
        assert set_cuda_int4_expert_fusion(expert, mode) == 1
        for x, target in zip(inputs, expected):
            if x.is_contiguous() and x.data_ptr() % 16 == 0 and 0 < x.size(0) <= 4:
                gate, up, down = expert.gate_proj, expert.up_proj, expert.down_proj
                hidden = x.new_empty((x.size(0), 96))
                out = torch.empty_like(x)
                launched = gate._cuda_int4_ext.expert_out(
                    x, gate.w_int4, gate.w_int4_scale, up.w_int4, up.w_int4_scale,
                    down.w_int4, down.w_int4_scale, hidden, out, mode == "full")
                if launched:
                    torch.testing.assert_close(hidden, torch.nn.functional.silu(gate(x)) * up(x),
                                               rtol=.002, atol=.0005)
                    assert torch.equal(out, expert(x))
                else:
                    assert mode == "full"
            torch.testing.assert_close(expert(x), target, rtol=.002, atol=.0005)
