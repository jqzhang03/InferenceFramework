import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_small_batch_and_graph(dtype):
    from nanovllm.layers.cuda_linear import CudaLinear, load_extension
    ext = load_extension()
    for m in (1, 4, 8, 33):
        x = torch.randn(m, 512, device="cuda", dtype=dtype)
        w = torch.randn(71, 512, device="cuda", dtype=dtype)
        layer = CudaLinear(w, "cuda", ext)
        expected = x.double() @ w.double().T
        output = layer(x)
        error = (output.double() - expected).norm() / expected.norm()
        assert error < (0.001 if dtype == torch.float16 else 0.006)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(x)
        x.mul_(0.5)
        graph.replay()
        torch.testing.assert_close(captured, layer(x), rtol=0, atol=0)


def test_fp8_row_quantization_and_graph():
    from nanovllm.layers.cuda_linear import load_extension
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("native FP8 targets SM120")
    ext = load_extension(True)
    x = torch.randn(37, 144, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    packed = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(37, device="cuda")
    w = torch.randn(81, 144, device="cuda").to(torch.float8_e4m3fn)
    out = torch.empty(37, 81, device="cuda", dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    ext.pack_rows_out(x, packed, scales)
    with torch.cuda.graph(graph):
        ext.pack_rows_out(x, packed, scales)
        ext.gemm_rows_out(packed, w, out, scales, 3, 0.02)
    x.mul_(2)
    graph.replay()
    expected = (packed.double() @ w.double().T) * scales.double()[:, None] * 0.02
    assert (out.double()-expected).norm()/expected.norm() < 0.004
    assert torch.count_nonzero(out[0]) == 0


def test_qwen06_default_backend_generation():
    from pathlib import Path

    from nanovllm import LLM, SamplingParams

    model = Path("/home/zjq/huggingface/Qwen3-0.6B")
    if not model.is_dir():
        pytest.skip("local Qwen3-0.6B checkpoint required")
    llm = LLM(str(model), max_num_seqs=8, max_num_batched_tokens=256,
              max_model_len=256, gpu_memory_utilization=0.55)
    try:
        assert llm.config.linear_backend == "cuda_decode"
        assert llm.model_runner.cuda_linear_count == 112
        outputs = llm.generate([[100, 200, 300, 400]] * 8,
                               SamplingParams(temperature=0.8, max_tokens=4, ignore_eos=True),
                               use_tqdm=False)
        assert len(outputs) == 8
        assert all(len(item["token_ids"]) == 4 for item in outputs)
        assert llm.model_runner.decode_cudagraph_replay_count > 0
    finally:
        llm.exit()
