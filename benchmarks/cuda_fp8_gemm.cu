// Native SM120 FP8 GEMM: C[M,N] = alpha * A[M,K] * W[N,K]^T.
// Packed FP8 values stay in FP8 through shared memory and native Tensor Cores.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cmath>
#include <memory>

__device__ __forceinline__ unsigned shared_address(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void async16(void* dst,const void* src,bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 ::"r"(shared_address(dst)),"l"(src),"r"(valid?16:0):"memory");
}
template<int BK,int NT,int ROWS> __device__ __forceinline__
void copy_rows(unsigned char* dst,const unsigned char* src,int rows,int K,int start,int kt) {
    #pragma unroll
    for(int v=threadIdx.x;v<ROWS*BK/16;v+=NT) {
        int row=v/(BK/16),col=(v%(BK/16))*16;
        bool valid=start+row<rows && kt*BK+col<K;
        int swizzled=col ^ (((row/2)%(BK/16))*16);
        async16(dst+row*BK+swizzled,src+(valid?int64_t(start+row)*K+kt*BK+col:0),valid);
    }
}
template<int BM,int BN,int BK,int NT> __device__ __forceinline__
void load_tile(unsigned char* dst,const unsigned char* a,const unsigned char* w,
               int M,int N,int K,int m0,int n0,int kt) {
    copy_rows<BK,NT,BM>(dst,a,M,K,m0,kt);
    copy_rows<BK,NT,BN>(dst+BM*BK,w,N,K,n0,kt);
    asm volatile("cp.async.commit_group;":::"memory");
}
template<bool E5> __device__ __forceinline__
void fp8_mma(float (&d)[4],const unsigned (&a)[4],const unsigned (&b)[2]) {
    if constexpr(E5) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e5m2.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
}
template<bool E5,int BM,int BN,int BK,int WM,int WN,int ST,int GROUP_M=1>
__global__ void fp8_kernel(const unsigned char* __restrict__ a,const unsigned char* __restrict__ w,
                          __nv_bfloat16* __restrict__ out,int M,int N,int K,float alpha) {
    constexpr int NT=(BM/WM)*(BN/WN)*32,AS=WM/16,BS=WN/8,SS=(BM+BN)*BK;
    extern __shared__ __align__(16) unsigned char sm[];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,wm=warp/(BN/WN),wn=warp%(BN/WN);
    int tile_m=blockIdx.y,tile_n=blockIdx.x;
    if constexpr(GROUP_M>1) {
        int num_m=(M+BM-1)/BM,num_n=(N+BN-1)/BN;
        int group=blockIdx.x/(GROUP_M*num_n),first_m=group*GROUP_M;
        int group_m=min(GROUP_M,num_m-first_m);
        int within=blockIdx.x%(GROUP_M*num_n);
        tile_m=first_m+within%group_m;tile_n=within/group_m;
    }
    int m0=tile_m*BM,n0=tile_n*BN;
    float acc[AS][BS][4]={};
    #pragma unroll
    for(int st=0;st<ST-1;++st) load_tile<BM,BN,BK,NT>(sm+st*SS,a,w,M,N,K,m0,n0,st);
    for(int kt=0;kt<(K+BK-1)/BK;++kt) {
        int next=kt+ST-1;
        load_tile<BM,BN,BK,NT>(sm+(next%ST)*SS,a,w,M,N,K,m0,n0,next);
        asm volatile("cp.async.wait_group %0;"::"n"(ST-1):"memory");
        __syncthreads();
        auto sa=sm+(kt%ST)*SS,sb=sa+BM*BK;
        #pragma unroll
        for(int kh=0;kh<BK;kh+=32) {
            unsigned ar[AS][4],br[BS][2];
            #pragma unroll
            for(int i=0;i<AS;++i) {
                int r=wm*WM+i*16+lane%16;
                int col=(kh+(lane/16)*16)^(((r/2)%(BK/16))*16);
                unsigned addr=shared_address(sa+r*BK+col);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    :"=r"(ar[i][0]),"=r"(ar[i][1]),"=r"(ar[i][2]),"=r"(ar[i][3]):"r"(addr));
            }
            #pragma unroll
            for(int j=0;j<BS;++j) {
                int r=wn*WN+j*8+lane%8;
                int col=(kh+((lane/8)%2)*16)^(((r/2)%(BK/16))*16);
                unsigned addr=shared_address(sb+r*BK+col);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                    :"=r"(br[j][0]),"=r"(br[j][1]):"r"(addr));
            }
            #pragma unroll
            for(int i=0;i<AS;++i)
                #pragma unroll
                for(int j=0;j<BS;++j) fp8_mma<E5>(acc[i][j],ar[i],br[j]);
        }
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;":::"memory");
    #pragma unroll
    for(int i=0;i<AS;++i)
        #pragma unroll
        for(int j=0;j<BS;++j) {
            int r=m0+wm*WM+i*16+lane/4,col=n0+wn*WN+j*8+(lane%4)*2;
            #pragma unroll
            for(int pair=0;pair<2;++pair) {
                int rr=r+pair*8;
                if(rr<M && col<N) {
                    if(!(N%2) && !(reinterpret_cast<uintptr_t>(out)%4) && col+1<N) {
                        auto packed=__floats2bfloat162_rn(acc[i][j][pair*2]*alpha,acc[i][j][pair*2+1]*alpha);
                        *reinterpret_cast<__nv_bfloat162*>(out+int64_t(rr)*N+col)=packed;
                    } else {
                        out[int64_t(rr)*N+col]=__float2bfloat16_rn(acc[i][j][pair*2]*alpha);
                        if(col+1<N) out[int64_t(rr)*N+col+1]=__float2bfloat16_rn(acc[i][j][pair*2+1]*alpha);
                    }
                }
            }
        }
}

void validate_fp8(torch::Tensor a,torch::Tensor w,torch::Tensor out) {
    TORCH_CHECK(a.is_cuda() && w.is_cuda() && out.is_cuda(),"CUDA tensors required");
    TORCH_CHECK(a.device()==w.device() && a.device()==out.device(),"device mismatch");
    TORCH_CHECK(a.dim()==2 && w.dim()==2 && out.dim()==2,"2D tensors required");
    TORCH_CHECK(a.scalar_type()==at::kFloat8_e4m3fn || a.scalar_type()==at::kFloat8_e5m2,"E4M3/E5M2 required");
    TORCH_CHECK(a.scalar_type()==w.scalar_type() && out.scalar_type()==at::kBFloat16,"dtype mismatch; output must be BF16");
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && out.is_contiguous(),"contiguous A[M,K], W[N,K], C[M,N] required");
    TORCH_CHECK(a.size(1)==w.size(1) && out.size(0)==a.size(0) && out.size(1)==w.size(0),"shape mismatch");
    TORCH_CHECK(a.size(0)<=INT_MAX && w.size(0)<=INT_MAX && a.size(1)<=INT_MAX,"dimensions exceed int32");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr())%16==0 && reinterpret_cast<uintptr_t>(w.data_ptr())%16==0,"16-byte input alignment required");
    TORCH_CHECK(a.size(1)%16==0,"K must be divisible by 16");
    TORCH_CHECK(!out.is_alias_of(a) && !out.is_alias_of(w),"output aliases input");
}

#define FP8_CONFIGS(X) \
 X(0,64,64,64,32,32,2) X(1,64,64,64,32,32,3) X(2,64,64,64,32,32,4) \
 X(3,64,128,64,32,64,2) X(4,64,128,64,32,64,3) X(5,64,128,64,32,64,4) \
 X(6,128,128,64,32,64,2) X(7,128,128,64,32,64,3) X(8,128,128,64,32,64,4) \
 X(9,64,128,128,32,64,2) X(10,64,128,128,32,64,3) \
 X(11,128,128,128,32,64,2) X(12,128,256,64,64,64,2) X(13,128,256,64,64,64,3) \
 X(14,64,128,64,64,32,2) X(15,64,128,64,64,32,3) \
 X(16,32,64,64,16,32,3) X(17,64,64,128,32,32,3)
#define FP8_GROUPED_CONFIGS(X) \
 X(18,64,64,64,32,32,2) X(19,64,128,64,32,64,2) \
 X(20,128,128,64,32,64,2) X(21,64,128,128,32,64,2) X(22,128,128,128,32,64,2) \
 X(23,128,64,64,32,32,2) X(24,128,64,64,32,32,3) X(25,128,64,128,32,32,2) \
 X(26,128,64,64,64,32,2) X(27,32,64,64,16,32,3)

template<bool E5,int BM,int BN,int BK,int WM,int WN,int ST,int GROUP_M=1>
void launch_fp8(torch::Tensor a,torch::Tensor w,torch::Tensor out,float alpha) {
    constexpr int bytes=(BM+BN)*BK*ST,threads=(BM/WM)*(BN/WN)*32;
    auto fn=fp8_kernel<E5,BM,BN,BK,WM,WN,ST,GROUP_M>;
    if constexpr(bytes>48*1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    dim3 grid((w.size(0)+BN-1)/BN,(a.size(0)+BM-1)/BM);
    if constexpr(GROUP_M>1) grid=dim3(grid.x*grid.y);
    fn<<<grid,threads,bytes,at::cuda::getCurrentCUDAStream()>>>(
        static_cast<const unsigned char*>(a.data_ptr()),static_cast<const unsigned char*>(w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),a.size(0),w.size(0),a.size(1),alpha);
}
template<bool E5> void dispatch_fp8(torch::Tensor a,torch::Tensor w,torch::Tensor out,int config,float alpha) {
    switch(config) {
    #define CASE(ID,BM,BN,BK,WM,WN,ST) case ID: launch_fp8<E5,BM,BN,BK,WM,WN,ST>(a,w,out,alpha); break;
    FP8_CONFIGS(CASE)
    #undef CASE
    #define GROUP_CASE(ID,BM,BN,BK,WM,WN,ST) case ID: launch_fp8<E5,BM,BN,BK,WM,WN,ST,8>(a,w,out,alpha); break;
    FP8_GROUPED_CONFIGS(GROUP_CASE)
    #undef GROUP_CASE
    default: TORCH_CHECK(false,"unknown FP8 config");
    }
}
void gemm_fp8_out(torch::Tensor a,torch::Tensor w,torch::Tensor out,int64_t config,double alpha) {
    validate_fp8(a,w,out);
    TORCH_CHECK(config>=0 && config<28,"config must be in [0,28)");
    TORCH_CHECK(std::isfinite(alpha) && std::abs(alpha)<=FLT_MAX,"invalid alpha");
    const c10::cuda::CUDAGuard guard(a.device());
    auto props=at::cuda::getDeviceProperties(a.get_device());
    TORCH_CHECK(props->major==12 && props->minor==0,"native FP8 extension requires SM120");
    if(!out.numel()) return;
    if(!a.size(1)) {out.zero_();return;}
    if(a.scalar_type()==at::kFloat8_e5m2) dispatch_fp8<true>(a,w,out,config,alpha);
    else dispatch_fp8<false>(a,w,out,config,alpha);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lt_check(cublasStatus_t status,const char* name) {
    TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,name,": cuBLASLt status ",int(status));
}
struct Fp8Plan {
    cublasLtHandle_t handle=nullptr;
    cublasLtMatmulDesc_t op=nullptr;
    cublasLtMatrixLayout_t la=nullptr,lb=nullptr,lc=nullptr;
    cublasLtMatmulPreference_t pref=nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> algorithms;
    torch::Tensor workspace;
    int M,N,K,device; bool e5;
    ~Fp8Plan() {
        if(pref) cublasLtMatmulPreferenceDestroy(pref);
        if(la) cublasLtMatrixLayoutDestroy(la);
        if(lb) cublasLtMatrixLayoutDestroy(lb);
        if(lc) cublasLtMatrixLayoutDestroy(lc);
        if(op) cublasLtMatmulDescDestroy(op);
        if(handle) cublasLtDestroy(handle);
    }
    void init(torch::Tensor a,torch::Tensor w,torch::Tensor out) {
        validate_fp8(a,w,out);
        M=a.size(0);N=w.size(0);K=a.size(1);device=a.get_device();e5=a.scalar_type()==at::kFloat8_e5m2;
        TORCH_CHECK(M>0 && N>0 && K>0,"cuBLASLt plan requires positive dimensions");
        const c10::cuda::CUDAGuard guard(a.device());
        lt_check(cublasLtCreate(&handle),"create");
        lt_check(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F),"op");
        cublasOperation_t transA=CUBLAS_OP_T,transB=CUBLAS_OP_N;
        lt_check(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA)),"transA");
        lt_check(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB)),"transB");
        int8_t fast=0;
        lt_check(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fast,sizeof(fast)),"FP32 accumulation");
        auto type=e5?CUDA_R_8F_E5M2:CUDA_R_8F_E4M3;
        // Column-major transpose identity: C^T = W * A^T, TN FP8 layout.
        lt_check(cublasLtMatrixLayoutCreate(&la,type,K,N,K),"layout A");
        lt_check(cublasLtMatrixLayoutCreate(&lb,type,K,M,K),"layout B");
        lt_check(cublasLtMatrixLayoutCreate(&lc,CUDA_R_16BF,N,M,N),"layout C");
        lt_check(cublasLtMatmulPreferenceCreate(&pref),"preference");
        size_t max_workspace=32*1024*1024;
        lt_check(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&max_workspace,sizeof(max_workspace)),"workspace");
        workspace=torch::empty({int64_t(max_workspace)},a.options().dtype(at::kByte));
        std::vector<cublasLtMatmulHeuristicResult_t> found(32); int count=0;
        lt_check(cublasLtMatmulAlgoGetHeuristic(handle,op,la,lb,lc,lc,pref,found.size(),found.data(),&count),"heuristics");
        for(int i=0;i<count;++i) if(found[i].state==CUBLAS_STATUS_SUCCESS) algorithms.push_back(found[i]);
        TORCH_CHECK(!algorithms.empty(),"no cuBLASLt FP8 algorithm supports this shape/dtype");
    }
    int count() const {return algorithms.size();}
    pybind11::dict algorithm_info(int64_t index) const {
        TORCH_CHECK(index>=0 && index<int64_t(algorithms.size()),"invalid algorithm");
        pybind11::dict result;
        auto attribute=[&](const char* key,cublasLtMatmulAlgoConfigAttributes_t attr) {
            int value=0;size_t written=0;
            auto status=cublasLtMatmulAlgoConfigGetAttribute(&algorithms[index].algo,attr,&value,sizeof(value),&written);
            if(status==CUBLAS_STATUS_SUCCESS) result[key]=value;
        };
        attribute("id",CUBLASLT_ALGO_CONFIG_ID);
        attribute("tile",CUBLASLT_ALGO_CONFIG_TILE_ID);
        attribute("stages",CUBLASLT_ALGO_CONFIG_STAGES_ID);
        attribute("split_k",CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
        attribute("reduction",CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
        attribute("swizzle",CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
        attribute("custom",CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
        result["workspace_bytes"]=algorithms[index].workspaceSize;
        return result;
    }
    void run(torch::Tensor a,torch::Tensor w,torch::Tensor out,int64_t algorithm,double scale) {
        validate_fp8(a,w,out);
        TORCH_CHECK(a.get_device()==device && a.size(0)==M && w.size(0)==N && a.size(1)==K && (a.scalar_type()==at::kFloat8_e5m2)==e5,"plan mismatch");
        TORCH_CHECK(algorithm>=0 && algorithm<int64_t(algorithms.size()),"invalid algorithm");
        TORCH_CHECK(std::isfinite(scale) && std::abs(scale)<=FLT_MAX,"invalid alpha");
        const c10::cuda::CUDAGuard guard(a.device());
        float alpha=scale,beta=0;
        lt_check(cublasLtMatmul(handle,op,&alpha,w.data_ptr(),la,a.data_ptr(),lb,&beta,
            out.data_ptr(),lc,out.data_ptr(),lc,&algorithms[algorithm].algo,workspace.data_ptr(),workspace.numel(),
            at::cuda::getCurrentCUDAStream()),"matmul");
    }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("gemm_out",&gemm_fp8_out);
    pybind11::class_<Fp8Plan,std::shared_ptr<Fp8Plan>>(m,"Plan")
        .def(pybind11::init([](torch::Tensor a,torch::Tensor w,torch::Tensor out) {
            auto p=std::make_shared<Fp8Plan>();p->init(a,w,out);return p;
        }))
        .def("count",&Fp8Plan::count).def("algorithm_info",&Fp8Plan::algorithm_info).def("run_out",&Fp8Plan::run);
}
