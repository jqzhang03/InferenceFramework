"""Native FP8 fragment, tail, graph/stream and cuBLASLt parity checks on SM120."""
import pytest

torch=pytest.importorskip("torch")
pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,0),
                            reason="requires RTX Blackwell SM120 and CUDA toolchain")


def test_native_fp8_all_configurations() -> None:
    from benchmarks.cuda_fp8_gemm_bench import correctness_checks,load_extension

    assert correctness_checks(load_extension())["passed"]


def test_cublaslt_fp8_plan_graph_and_scale() -> None:
    from benchmarks.cuda_fp8_gemm_bench import accuracy,load_extension,pack

    ext=load_extension()
    torch.manual_seed(29)
    a,sa=pack(torch.randn(64,128,device="cuda"))
    w,sw=pack(torch.randn(64,128,device="cuda"))
    c=torch.empty(64,64,device="cuda",dtype=torch.bfloat16)
    d=torch.empty_like(c)
    plan=ext.Plan(a,w,d)
    alpha=sa*sw
    plan.run_out(a,w,d,0,alpha)
    ext.gemm_out(a,w,c,4,alpha)
    accuracy(a,w,d,alpha)
    torch.testing.assert_close(c,d,rtol=8e-3,atol=0.125)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run_out(a,w,d,0,alpha*2)
        ext.gemm_out(a,w,c,4,alpha*2)
    graph.replay()
    torch.cuda.synchronize()
    accuracy(a,w,c,alpha*2)
    accuracy(a,w,d,alpha*2)
    with pytest.raises(RuntimeError,match="plan mismatch"):
        plan.run_out(a[:32],w,d[:32],0,alpha)
