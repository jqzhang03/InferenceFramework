import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_decode_configs(dtype):
    from nanovllm.layers.cuda_linear import load_decode_extension
    ext=load_decode_extension()
    torch.manual_seed(93)
    for m,n,k in ((1,71,512),(4,81,144),(8,129,1024),(16,63,256),(31,33,520),(8,17,8)):
        x=torch.randn(m,k,device='cuda',dtype=dtype)
        w=torch.randn(n,k,device='cuda',dtype=dtype)
        out=torch.empty(m,n,device='cuda',dtype=dtype)
        expected=x.double()@w.double().T
        for config in range(24):
            ext.decode_out(x,w,out,config)
            error=(out.double()-expected).norm()/expected.norm()
            assert error < (0.001 if dtype==torch.float16 else 0.006),(m,n,k,config,error)
    for config in (1,7,18,22):
        ext.decode_out(x,w,out,config)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):ext.decode_out(x,w,out,config)
        x.mul_(0.5);graph.replay()
        expected=x.double()@w.double().T
        assert (out.double()-expected).norm()/expected.norm() < (0.001 if dtype==torch.float16 else 0.006)
    zero=torch.empty(8,0,device='cuda',dtype=dtype)
    weight=torch.empty(17,0,device='cuda',dtype=dtype)
    result=torch.empty(8,17,device='cuda',dtype=dtype)
    ext.decode_out(zero,weight,result,1)
    assert torch.count_nonzero(result)==0


def test_bundled_qwen_projection_dispatch():
    from nanovllm.layers.cuda_linear import install_cuda_linears
    from nanovllm.layers.linear import ColumnParallelLinear
    if torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5060 Ti':
        pytest.skip('bundled selector is specific to RTX 5060 Ti')
    root=torch.nn.Module();root.model=torch.nn.Module()
    layer=torch.nn.Module()
    layer.o_proj=ColumnParallelLinear(2048,1024,bias=False).to(device='cuda',dtype=torch.bfloat16)
    root.model.layers=torch.nn.ModuleList([layer])
    with torch.no_grad():layer.o_proj.weight.normal_(std=0.02)
    assert install_cuda_linears(root,'cuda_decode')==1
    assert layer.o_proj._cuda_linear.selectors['8,1024,2048']==dict(mode='decode',config=5)
    x=torch.randn(8,2048,device='cuda',dtype=torch.bfloat16)
    actual=layer.o_proj(x)
    expected=x.double()@layer.o_proj.weight.double().T
    assert (actual.double()-expected).norm()/expected.norm()<0.006
