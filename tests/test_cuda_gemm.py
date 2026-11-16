"""Standalone CUDA GEMM checks; no model checkpoint or inference engine needed."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_gemm_accuracy_boundaries_and_streams() -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires SM80+")
    from benchmarks.cuda_gemm_bench import correctness_checks, load_extension

    torch.manual_seed(42)
    result = correctness_checks(load_extension())
    assert result["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_gemm_profiling_controls_preserve_outputs() -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires SM80+")
    from benchmarks.cuda_gemm_bench import load_extension, relative_l2

    ext = load_extension()
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(17)
        a = torch.randn(129, 72, device="cuda", dtype=dtype)
        b = torch.randn(72, 136, device="cuda", dtype=dtype)
        out = torch.empty(129, 136, device="cuda", dtype=dtype)
        expected = a.double() @ b.double()
        for variant in (0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 13):
            ext.gemm_diagnostic_out(a, b, out, variant, 0)
            assert relative_l2(out, expected) < (8e-4 if dtype == torch.float16 else 6e-3)
        # Reserving unused shared memory changes residency, not arithmetic.
        baseline = torch.empty_like(out)
        ext.gemm_out(a, b, baseline, 10)
        ext.gemm_out_smem(a, b, out, 10, 16 * 1024)
        torch.testing.assert_close(out, baseline, rtol=0, atol=0)
        natural = ext.kernel_info(dtype == torch.bfloat16, 10, 1024, 1024, 0)
        padded = ext.kernel_info(dtype == torch.bfloat16, 10, 1024, 1024, 16 * 1024)
        assert padded["active_blocks_per_sm_limit"] <= natural["active_blocks_per_sm_limit"]
