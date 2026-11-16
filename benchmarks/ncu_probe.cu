// Minimal counter-profiling probe: no PyTorch, Python extension, or cuBLAS.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); return 2; } } while(0)

__global__ void ncu_probe(const float* a,float* b,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) b[i]=a[i]*2.0f+1.0f;
}

int main() {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props,0));
    int driver=0,runtime=0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::printf("device=%s sm=%d%d driver=%d runtime=%d\n",props.name,props.major,props.minor,driver,runtime);
    constexpr int n=65536;
    float *a=nullptr,*b=nullptr;
    CUDA_CHECK(cudaMalloc(&a,n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b,n*sizeof(float)));
    CUDA_CHECK(cudaMemset(a,0,n*sizeof(float)));
    ncu_probe<<<(n+255)/256,256>>>(a,b,n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(),b,n*sizeof(float),cudaMemcpyDeviceToHost));
    for(float value:host) if(value!=1.0f) { std::fprintf(stderr,"wrong output\n"); return 3; }
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    std::puts("probe validation: PASS");
    return 0;
}
