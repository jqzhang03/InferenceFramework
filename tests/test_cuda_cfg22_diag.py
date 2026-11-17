import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_cfg22_alt_a_preserves_results_and_graph(dtype):
    from nanovllm.layers.cuda_linear import load_extension
    ext=load_extension();torch.manual_seed(99)
    for m,n,k in ((8,4096,1024),(8,6144,1024),(1,128,40),(16,256,64)):
        a=torch.randn(m,k,device='cuda',dtype=dtype)
        b=torch.randn(k,n,device='cuda',dtype=dtype)
        original=torch.empty(m,n,device='cuda',dtype=dtype)
        alternate=torch.empty_like(original)
        ext.gemm_out(a,b,original,22)
        ext.gemm_diagnostic_out(a,b,alternate,14,0)
        torch.testing.assert_close(alternate,original,rtol=0,atol=0)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):ext.gemm_diagnostic_out(a,b,alternate,14,0)
        a.mul_(0.5);g.replay();ext.gemm_out(a,b,original,22)
        torch.testing.assert_close(alternate,original,rtol=0,atol=0)
