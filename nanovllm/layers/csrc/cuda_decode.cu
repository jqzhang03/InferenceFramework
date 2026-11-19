// Small-M row-major Y = X W^T: weight reuse, FP32 accumulation, current stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ unsigned addr(const void* p) {return unsigned(__cvta_generic_to_shared(p));}
__device__ void cp16(void* dst,const void* src,bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(addr(dst)),"l"(src),"r"(valid?16:0):"memory");
}
template<bool BF> __device__ float unpack(unsigned short v) {
    if constexpr(BF) return __bfloat162float(__ushort_as_bfloat16(v));
    else return __half2float(__ushort_as_half(v));
}
template<bool BF> __device__ unsigned short pack(float v) {
    if constexpr(BF) return __bfloat16_as_ushort(__float2bfloat16_rn(v));
    else return __half_as_ushort(__float2half_rn(v));
}
template<bool BF> __device__ void mma(float (&d)[4],unsigned (&a)[4],unsigned (&b)[2]) {
    if constexpr(BF) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    else asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

// Weight tile is the MMA M axis; requests are its N axis, filling m16n8 at M=8.
template<int BN,int BM,int BK,int NT,bool SWIZZLE=true> __device__ void stage(unsigned short* s,
    const unsigned short* w,const unsigned short* x,int N,int M,int K,int n0,int m0,int kt,int kend) {
    #pragma unroll
    for(int v=threadIdx.x;v<(BN+BM)*BK/8;v+=NT) {
        int r=v/(BK/8),c=(v%(BK/8))*8;
        int sw=SWIZZLE?(c^((r%(BK/8))*8)):c;
        bool weight=r<BN;int row=weight?n0+r:m0+r-BN;
        bool valid=row<(weight?N:M) && kt*BK+c<kend && kt*BK+c<K;
        const unsigned short* src=weight?w:x;
        cp16(s+r*BK+sw,src+(valid?int64_t(row)*K+kt*BK+c:0),valid);
    }
    asm volatile("cp.async.commit_group;":::"memory");
}
template<bool BF,int BN,int BM,int BK,int WARPS,int ST,int SPLIT,bool SWIZZLE=true>
__global__ void decode_mma_kernel(const unsigned short* x,const unsigned short* w,
    unsigned short* out,float* partial,int M,int N,int K) {
    constexpr int NT=WARPS*32,WN=BN/WARPS,AS=WN/16,BS=BM/8,SS=(BN+BM)*BK;
    extern __shared__ __align__(16) unsigned short sm[];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,n0=blockIdx.x*BN,m0=blockIdx.y*BM;
    int tiles=(K+BK-1)/BK,per=(tiles+SPLIT-1)/SPLIT;
    int begin=blockIdx.z*per,finish=min(tiles,begin+per),kend=min(K,finish*BK);
    float accum[AS][BS][4]={};
    #pragma unroll
    for(int st=0;st<ST-1;++st) stage<BN,BM,BK,NT,SWIZZLE>(sm+st*SS,w,x,N,M,K,n0,m0,begin+st,kend);
    for(int kt=begin;kt<finish;++kt) {
        int next=kt+ST-1;
        stage<BN,BM,BK,NT,SWIZZLE>(sm+((next-begin)%ST)*SS,w,x,N,M,K,n0,m0,next,kend);
        asm volatile("cp.async.wait_group %0;"::"n"(ST-1):"memory");__syncthreads();
        auto sa=sm+((kt-begin)%ST)*SS;auto sb=sa+BN*BK;
        #pragma unroll
        for(int kh=0;kh<BK;kh+=16) {
            unsigned ar[AS][4],br[BS][2];
            #pragma unroll
            for(int i=0;i<AS;++i) {
                int row=warp*WN+i*16+lane%16;
                int col=kh+(lane/16)*8;if constexpr(SWIZZLE)col^=(row%(BK/8))*8;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    :"=r"(ar[i][0]),"=r"(ar[i][1]),"=r"(ar[i][2]),"=r"(ar[i][3]):"r"(addr(sa+row*BK+col)));
            }
            #pragma unroll
            for(int j=0;j<BS;++j) {
                int row=j*8+lane%8;
                int col=kh+((lane/8)%2)*8;if constexpr(SWIZZLE)col^=((BN+row)%(BK/8))*8;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                    :"=r"(br[j][0]),"=r"(br[j][1]):"r"(addr(sb+row*BK+col)));
            }
            #pragma unroll
            for(int i=0;i<AS;++i)
                #pragma unroll
                for(int j=0;j<BS;++j) mma<BF>(accum[i][j],ar[i],br[j]);
        }
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;":::"memory");
    #pragma unroll
    for(int i=0;i<AS;++i)
        #pragma unroll
        for(int j=0;j<BS;++j)
            #pragma unroll
            for(int p=0;p<4;++p) {
                int n=n0+warp*WN+i*16+lane/4+(p/2)*8,m=m0+j*8+(lane%4)*2+p%2;
                if(n<N && m<M) {
                    int64_t idx=int64_t(m)*N+n;
                    if constexpr(SPLIT==1) out[idx]=pack<BF>(accum[i][j][p]);
                    else partial[int64_t(blockIdx.z)*M*N+idx]=accum[i][j][p];
                }
            }
}
template<bool BF,int SPLIT> __global__ void finish_kernel(const float* partial,unsigned short* out,int64_t size) {
    int64_t idx=int64_t(blockIdx.x)*256+threadIdx.x;if(idx>=size)return;
    float sum=0;
    #pragma unroll
    for(int s=0;s<SPLIT;++s)sum+=partial[int64_t(s)*size+idx];
    out[idx]=pack<BF>(sum);
}

// Packed 16-byte shared loads: each 128-byte warp transaction addresses distinct
// banks; duplicate lane-group reads are broadcasts. One padded vector per row.
template<bool BF,int ROWS,int LANES,int BK=256>
__global__ void decode_simt_kernel(const unsigned short* x,const unsigned short* w,
    unsigned short* out,int M,int N,int K) {
    constexpr int NT=128,STRIDE=BK/8+1,GROUPS=NT/LANES;
    __shared__ __align__(16) uint4 sx[ROWS*STRIDE];
    int lane=threadIdx.x%LANES,n=blockIdx.x*GROUPS+threadIdx.x/LANES,m0=blockIdx.y*ROWS;
    float accum[ROWS]={};
    for(int k0=0;k0<K;k0+=BK) {
        #pragma unroll
        for(int v=threadIdx.x;v<ROWS*(BK/8);v+=NT) {
            int r=v/(BK/8),c=v%(BK/8);bool valid=m0+r<M && k0+c*8<K;
            cp16(sx+r*STRIDE+c,x+(valid?int64_t(m0+r)*K+k0+c*8:0),valid);
        }
        asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");__syncthreads();
        #pragma unroll
        for(int kk=lane*8;kk<BK;kk+=LANES*8) {
            uint4 wv={0,0,0,0};if(n<N && k0+kk<K)wv=*reinterpret_cast<const uint4*>(w+int64_t(n)*K+k0+kk);
            auto wp=reinterpret_cast<unsigned short*>(&wv);
            #pragma unroll
            for(int r=0;r<ROWS;++r) {
                uint4 xv=sx[r*STRIDE+kk/8];auto xp=reinterpret_cast<unsigned short*>(&xv);
                #pragma unroll
                for(int j=0;j<8;++j)accum[r]=fmaf(unpack<BF>(wp[j]),unpack<BF>(xp[j]),accum[r]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<ROWS;++r) {
        #pragma unroll
        for(int d=LANES/2;d;d/=2)accum[r]+=__shfl_down_sync(0xffffffff,accum[r],d,LANES);
        if(lane==0 && n<N && m0+r<M)out[int64_t(m0+r)*N+n]=pack<BF>(accum[r]);
    }
}

// Every lane group owns its outputs across all K and all request rows. X and W
// stay in registers: no shared-memory exchange, CTA barrier or partial buffer.
// All 128 threads execute every shuffle, including predicated boundary outputs.
template<bool BF,int ROWS,int LANES,int OUTPUTS,int CHAINS>
__global__ void decode_warp_gemv_kernel(const unsigned short* x,const unsigned short* w,
    unsigned short* out,int N,int K) {
    constexpr int NT=128,GROUPS=NT/LANES;
    int lane=threadIdx.x%LANES;
    int n0=(blockIdx.x*GROUPS+threadIdx.x/LANES)*OUTPUTS;
    float accum[ROWS][OUTPUTS][CHAINS]={};
    for(int k=lane*8;k<K;k+=LANES*8) {
        uint4 weights[OUTPUTS];
        #pragma unroll
        for(int o=0;o<OUTPUTS;++o) {
            weights[o]={0,0,0,0};
            if(n0+o<N)weights[o]=*reinterpret_cast<const uint4*>(w+int64_t(n0+o)*K+k);
        }
        #pragma unroll
        for(int r=0;r<ROWS;++r) {
            uint4 activation=*reinterpret_cast<const uint4*>(x+int64_t(r)*K+k);
            auto xp=reinterpret_cast<const unsigned short*>(&activation);
            #pragma unroll
            for(int o=0;o<OUTPUTS;++o) {
                auto wp=reinterpret_cast<const unsigned short*>(&weights[o]);
                #pragma unroll
                for(int j=0;j<8;++j)
                    accum[r][o][j%CHAINS]=fmaf(unpack<BF>(xp[j]),unpack<BF>(wp[j]),accum[r][o][j%CHAINS]);
            }
        }
    }
    #pragma unroll
    for(int r=0;r<ROWS;++r)
        #pragma unroll
        for(int o=0;o<OUTPUTS;++o) {
            float sum=accum[r][o][0];
            #pragma unroll
            for(int c=1;c<CHAINS;++c)sum+=accum[r][o][c];
            #pragma unroll
            for(int d=LANES/2;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d,LANES);
            if(lane==0 && n0+o<N)out[int64_t(r)*N+n0+o]=pack<BF>(sum);
        }
}

#define WARP_GEMV_CONFIGS(X) \
 X(38,32,1,1) X(39,32,1,2) X(40,32,2,1) X(41,32,2,2) \
 X(42,16,1,1) X(43,16,1,2) X(44,16,2,1) X(45,16,2,2) \
 X(46,8,1,1) X(47,8,1,2) X(48,8,2,1) X(49,8,2,2)

template<bool BF,int LANES,int OUTPUTS,int CHAINS>
void warp_gemv(torch::Tensor x,torch::Tensor w,torch::Tensor out) {
    dim3 grid((w.size(0)+(128/LANES)*OUTPUTS-1)/((128/LANES)*OUTPUTS));
    auto stream=at::cuda::getCurrentCUDAStream();
    #define ROW_CASE(R) case R:decode_warp_gemv_kernel<BF,R,LANES,OUTPUTS,CHAINS><<<grid,128,0,stream>>>( \
        (const unsigned short*)x.data_ptr(),(const unsigned short*)w.data_ptr(),(unsigned short*)out.data_ptr(),w.size(0),x.size(1));break;
    switch(x.size(0)) {ROW_CASE(1) ROW_CASE(2) ROW_CASE(3) ROW_CASE(4)}
    #undef ROW_CASE
}

#define MMA_CONFIGS(X) \
 X(0,64,8,32,4,2,1) X(1,64,8,64,4,2,1) X(2,64,8,64,4,3,1) \
 X(3,128,8,64,4,2,1) X(4,32,8,64,2,2,1) X(5,64,8,128,4,2,1) \
 X(6,64,8,64,4,2,2) X(7,64,8,64,4,2,4) X(8,64,8,64,4,2,8) \
 X(9,128,8,64,4,2,4) X(10,32,8,64,2,2,4) X(11,64,16,64,4,2,1) \
 X(12,64,16,64,4,2,4) X(13,64,8,128,4,2,4) \
 X(24,32,8,64,1,2,1) X(25,32,8,128,1,2,1) \
 X(26,64,8,64,2,2,1) X(27,64,8,128,2,2,1) \
 X(28,64,8,64,2,3,1) X(29,64,8,128,2,3,1) \
 X(30,32,8,128,1,3,1) X(31,32,8,64,1,2,2) \
 X(32,64,8,64,2,2,2) X(33,64,8,128,2,2,2) \
 X(34,32,8,64,1,2,4) X(35,64,8,64,2,2,4) \
 X(36,64,8,64,2,4,1) X(37,64,16,64,2,2,1)
template<bool BF,int BN,int BM,int BK,int WARPS,int ST,int SPLIT,bool SWIZZLE=true>
void launch(torch::Tensor x,torch::Tensor w,torch::Tensor out) {
    int M=x.size(0),N=w.size(0),K=x.size(1);auto stream=at::cuda::getCurrentCUDAStream();
    torch::Tensor partial;
    if constexpr(SPLIT>1)partial=torch::empty({SPLIT,M,N},x.options().dtype(at::kFloat));
    auto fn=decode_mma_kernel<BF,BN,BM,BK,WARPS,ST,SPLIT,SWIZZLE>;
    constexpr int bytes=(BN+BM)*BK*ST*2;
    if constexpr(bytes>49152) C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    fn<<<dim3((N+BN-1)/BN,(M+BM-1)/BM,SPLIT),WARPS*32,bytes,stream>>>(
        (const unsigned short*)x.data_ptr(),(const unsigned short*)w.data_ptr(),(unsigned short*)out.data_ptr(),
        SPLIT>1?partial.data_ptr<float>():nullptr,M,N,K);
    if constexpr(SPLIT>1)finish_kernel<BF,SPLIT><<<(int64_t(M)*N+255)/256,256,0,stream>>>(partial.data_ptr<float>(),(unsigned short*)out.data_ptr(),int64_t(M)*N);
}
template<bool BF,int ROWS,int LANES> void simt(torch::Tensor x,torch::Tensor w,torch::Tensor out) {
    decode_simt_kernel<BF,ROWS,LANES><<<dim3((w.size(0)+128/LANES-1)/(128/LANES),(x.size(0)+ROWS-1)/ROWS),128,0,at::cuda::getCurrentCUDAStream()>>>(
        (const unsigned short*)x.data_ptr(),(const unsigned short*)w.data_ptr(),(unsigned short*)out.data_ptr(),x.size(0),w.size(0),x.size(1));
}
template<bool BF> void dispatch(torch::Tensor x,torch::Tensor w,torch::Tensor out,int cfg) {
    switch(cfg) {
        #define CASE(ID,BN,BM,BK,WARPS,ST,SPLIT) case ID:launch<BF,BN,BM,BK,WARPS,ST,SPLIT>(x,w,out);break;
        MMA_CONFIGS(CASE)
        #undef CASE
        #define WARP_CASE(ID,LANES,OUTPUTS,CHAINS) case ID:warp_gemv<BF,LANES,OUTPUTS,CHAINS>(x,w,out);break;
        WARP_GEMV_CONFIGS(WARP_CASE)
        #undef WARP_CASE
        case 14:simt<BF,1,32>(x,w,out);break;case 15:simt<BF,1,16>(x,w,out);break;case 16:simt<BF,1,8>(x,w,out);break;
        case 17:simt<BF,4,32>(x,w,out);break;case 18:simt<BF,4,16>(x,w,out);break;case 19:simt<BF,4,8>(x,w,out);break;
        case 20:simt<BF,8,32>(x,w,out);break;case 21:simt<BF,8,16>(x,w,out);break;case 22:simt<BF,8,8>(x,w,out);break;
        // Diagnostic only: same split-K/down tile, no shared-memory swizzle.
        case 23:launch<BF,64,8,64,4,2,8,false>(x,w,out);break;
        default:TORCH_CHECK(false,"invalid decode config");
    }
}
void decode_impl(torch::Tensor x,torch::Tensor w,torch::Tensor out,int64_t cfg,bool no_swizzle) {
    TORCH_CHECK(x.is_cuda() && w.device()==x.device() && out.device()==x.device(),"CUDA device mismatch");
    TORCH_CHECK(x.dim()==2 && w.dim()==2 && out.dim()==2 && x.is_contiguous() && w.is_contiguous() && out.is_contiguous(),"contiguous matrices required");
    TORCH_CHECK(x.scalar_type()==w.scalar_type() && x.scalar_type()==out.scalar_type() && (x.scalar_type()==at::kHalf || x.scalar_type()==at::kBFloat16),"FP16/BF16 required");
    TORCH_CHECK(x.size(1)==w.size(1) && out.size(0)==x.size(0) && out.size(1)==w.size(0) && x.size(1)%8==0,"invalid shape");
    TORCH_CHECK(x.size(0)<=32 && cfg>=0 && cfg<50,"small-M requires M<=32 and config 0..49");
    TORCH_CHECK(cfg<38 || x.size(0)<=4,"warp GEMV requires M<=4");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0 && reinterpret_cast<uintptr_t>(w.data_ptr())%16==0,"16-byte alignment required");
    TORCH_CHECK(!out.is_alias_of(x) && !out.is_alias_of(w),"output aliases inputs");
    const c10::cuda::CUDAGuard guard(x.device());
    if(!out.numel())return;
    if(!x.size(1)){out.zero_();return;}
    if(no_swizzle) {
        // Controlled layout ablation: identical geometry and instruction mapping.
        #define NOSW(ID,BN,BM,BK,WARPS,ST,SPLIT) case ID: \
            if(x.scalar_type()==at::kBFloat16)launch<true,BN,BM,BK,WARPS,ST,SPLIT,false>(x,w,out); \
            else launch<false,BN,BM,BK,WARPS,ST,SPLIT,false>(x,w,out);break;
        switch(cfg) {MMA_CONFIGS(NOSW) default:TORCH_CHECK(false,"MMA config required for layout diagnostic");}
        #undef NOSW
    } else if(x.scalar_type()==at::kBFloat16)dispatch<true>(x,w,out,cfg);else dispatch<false>(x,w,out,cfg);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void decode_out(torch::Tensor x,torch::Tensor w,torch::Tensor out,int64_t cfg) {decode_impl(x,w,out,cfg,false);}
void decode_no_swizzle_out(torch::Tensor x,torch::Tensor w,torch::Tensor out,int64_t cfg) {decode_impl(x,w,out,cfg,true);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("decode_out",&decode_out);
    m.def("decode_no_swizzle_out",&decode_no_swizzle_out);
}
