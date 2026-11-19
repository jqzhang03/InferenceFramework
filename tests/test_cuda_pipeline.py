"""Boundary, stream and graph checks for new native CUDA pipeline candidates."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_warp_gemv_candidates(dtype):
    from nanovllm.layers.cuda_linear import load_decode_extension
    ext = load_decode_extension()
    stream = torch.cuda.Stream()
    torch.manual_seed(811)
    with torch.cuda.stream(stream):
        for m in range(1, 5):
            for n, k in ((1, 8), (17, 144), (71, 520), (129, 1024)):
                x = torch.randn(m, k, device="cuda", dtype=dtype)
                w = torch.randn(n, k, device="cuda", dtype=dtype)
                out = torch.empty(m, n, device="cuda", dtype=dtype)
                reference = x.double() @ w.double().T
                limit = .001 if dtype == torch.float16 else .006
                for config in range(38, 50):
                    ext.decode_out(x, w, out, config)
                    assert torch.isfinite(out).all()
                    assert (out.double()-reference).norm()/reference.norm().clamp_min(1e-12) < limit
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        ext.decode_out(x, w, out, config)
                    x.add_(.125)
                    w.mul_(-1)
                    graph.replay()
                    reference = x.double() @ w.double().T
                    assert (out.double()-reference).norm()/reference.norm().clamp_min(1e-12) < limit
                empty_x = torch.empty(m, 0, device="cuda", dtype=dtype)
                empty_w = torch.empty(n, 0, device="cuda", dtype=dtype)
                ext.decode_out(empty_x, empty_w, out, 38)
                assert torch.count_nonzero(out) == 0
    stream.synchronize()


def test_warp_gemv_rejects_large_batch():
    from nanovllm.layers.cuda_linear import load_decode_extension
    x = torch.empty(5, 8, device="cuda", dtype=torch.float16)
    w = torch.empty(17, 8, device="cuda", dtype=torch.float16)
    out = torch.empty(5, 17, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="warp GEMV requires M<=4"):
        load_decode_extension().decode_out(x, w, out, 38)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_warp_gemv_adapter_graph(dtype):
    from nanovllm.layers.cuda_linear import CudaLinear, load_extension, load_decode_extension
    weight = torch.randn(71, 520, device="cuda", dtype=dtype)
    bias = torch.randn(71, device="cuda", dtype=dtype)
    selectors = {f"{m},71,520": dict(mode="decode", config=49) for m in range(1, 5)}
    adapter = CudaLinear(weight, "cuda_decode", load_extension(), selectors=selectors,
                         decode_ext=load_decode_extension())
    for m in range(1, 5):
        x = torch.randn(m, 520, device="cuda", dtype=dtype)
        adapter(x, bias)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = adapter(x, bias)
        x.mul_(.5)
        graph.replay()
        expected = (x.double() @ weight.double().T).to(dtype) + bias
        assert (output.double()-expected.double()).norm()/expected.double().norm() < (.002 if dtype == torch.float16 else .012)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_pipeline_candidates(dtype):
    from nanovllm.layers.cuda_linear import load_decode_extension
    ext = load_decode_extension()
    stream = torch.cuda.Stream()
    torch.manual_seed(701)
    with torch.cuda.stream(stream):
        for m, n, k in ((1, 17, 8), (8, 129, 520), (31, 71, 1024)):
            x = torch.randn(m, k, device="cuda", dtype=dtype)
            w = torch.randn(n, k, device="cuda", dtype=dtype)
            out = torch.empty(m, n, device="cuda", dtype=dtype)
            reference = x.double() @ w.double().T
            for config in range(24, 38):
                ext.decode_out(x, w, out, config)
                assert (out.double()-reference).norm()/reference.norm() < (.001 if dtype == torch.float16 else .006)
            for config in (24, 26, 28, 33, 37):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    ext.decode_out(x, w, out, config)
                x.mul_(.5)
                graph.replay()
                reference = x.double() @ w.double().T
                assert (out.double()-reference).norm()/reference.norm() < (.001 if dtype == torch.float16 else .006)
                normal = out.clone()
                ext.decode_no_swizzle_out(x, w, out, config)
                torch.testing.assert_close(out, normal, rtol=0, atol=0)
    stream.synchronize()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_pipeline_candidates(dtype):
    from nanovllm.layers.cuda_linear import load_extension
    ext = load_extension()
    for m, n, k in ((8, 136, 40), (73, 264, 144), (1, 8, 8)):
        x = torch.randn(m, k, device="cuda", dtype=dtype)
        w = torch.randn(k, n, device="cuda", dtype=dtype)
        out = torch.empty(m, n, device="cuda", dtype=dtype)
        expected = x.double() @ w.double()
        for cfg in range(24, 36):
            ext.gemm_out(x, w, out, cfg)
            assert (out.double()-expected).norm()/expected.norm() < (.001 if dtype == torch.float16 else .006)
        for cfg in (24, 28, 30, 32, 35):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ext.gemm_out(x, w, out, cfg)
            x.mul_(.5)
            graph.replay()
            expected = x.double() @ w.double()
            assert (out.double()-expected).norm()/expected.norm() < (.001 if dtype == torch.float16 else .006)


def test_fp8_pipeline_candidates():
    from nanovllm.layers.cuda_linear import load_extension
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("native FP8 targets SM120")
    ext = load_extension(True)
    x = torch.randn(37, 144, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(81, 144, device="cuda").to(torch.float8_e4m3fn)
    packed = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(37, device="cuda", dtype=torch.float32)
    out = torch.empty(37, 81, device="cuda", dtype=torch.bfloat16)
    for cfg in range(28, 34):
        ext.pack_rows_out(x, packed, scales)
        ext.gemm_rows_out(packed, w, out, scales, cfg, .02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ext.pack_rows_out(x, packed, scales)
            ext.gemm_rows_out(packed, w, out, scales, cfg, .02)
        x.mul_(.5)
        graph.replay()
        expected = (packed.double() @ w.double().T) * scales.double()[:, None] * .02
        assert (out.double()-expected).norm()/expected.norm() < .006


@pytest.mark.parametrize("backend,dtype,m,n,k,config", [
    ("cuda_decode", torch.bfloat16, 2, 6144, 1024, 49),
    ("cuda_decode", torch.bfloat16, 3, 6144, 1024, 40),
    ("cuda_decode", torch.float16, 2, 4096, 1024, 45),
    ("cuda_decode", torch.float16, 4, 6144, 1024, 48),
    ("cuda_decode", torch.float16, 8, 6144, 1024, 26),
    ("cuda_decode", torch.float16, 8, 1024, 3072, 7),
    ("cuda_fp8", torch.bfloat16, 128, 4096, 1024, 0),
])
def test_tuned_selector_integration(backend, dtype, m, n, k, config):
    from nanovllm.layers.cuda_linear import install_cuda_linears
    from nanovllm.layers.linear import ColumnParallelLinear
    if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 5060 Ti":
        pytest.skip("bundled selectors are device specific")
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.proj = ColumnParallelLinear(k, n, bias=False).to(device="cuda", dtype=dtype)
    root.model.layers = torch.nn.ModuleList([layer])
    with torch.no_grad():
        layer.proj.weight.normal_(std=.02)
    assert install_cuda_linears(root, backend) == 1
    adapter = layer.proj._cuda_linear
    assert adapter.selectors[f"{m},{n},{k}"]["config"] == config
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    layer.proj(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer.proj(x)
    x.mul_(.5)
    graph.replay()
    if backend == "cuda_fp8":
        packed = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scales = torch.empty(m, device="cuda", dtype=torch.float32)
        adapter.fp8_ext.pack_rows_out(x, packed, scales)
        expected = (packed.double() @ adapter.packed.double().T) * scales.double()[:, None] * adapter.scale
    else:
        expected = x.double() @ layer.proj.weight.double().T
    assert (output.double()-expected).norm()/expected.norm() < (.001 if dtype == torch.float16 else .006)
