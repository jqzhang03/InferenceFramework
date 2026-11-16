// Standalone row-major C = A @ B. No cuBLAS calls in the custom kernels.
// SM80+ mma.sync, ldmatrix, XOR-swizzled shared memory and cp.async pipeline.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(smem_addr(dst)), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

template<bool BF> __device__ __forceinline__ void mma(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    if constexpr (BF) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}

template<bool BF> __device__ __forceinline__ unsigned short narrow(float v) {
    if constexpr (BF) return __bfloat16_as_ushort(__float2bfloat16_rn(v));
    else return __half_as_ushort(__float2half_rn(v));
}
template<bool BF> __device__ __forceinline__ float widen(unsigned short v) {
    if constexpr (BF) return __bfloat162float(__ushort_as_bfloat16(v));
    else return __half2float(__ushort_as_half(v));
}

template<int BM, int BN, int BK, int THREADS, bool ALT_A = false>
__device__ __forceinline__ void load_stage(
    unsigned short* s, const unsigned short* a, const unsigned short* b,
    int M, int N, int K, int m0, int n0, int kt) {
    const int tid = threadIdx.x;
    // XOR permutes 16-byte chunks within a row; both copy and ldmatrix use it.
    #pragma unroll
    for (int v = tid; v < BM * BK / 8; v += THREADS) {
        int r = v / (BK / 8), c = (v % (BK / 8)) * 8;
        bool ok = m0 + r < M && kt * BK + c < K;
        int swizzle_row = ALT_A && BK == 32 ? r / 2 : r;
        copy16(s + r * BK + (c ^ ((swizzle_row % (BK / 8)) * 8)),
               a + (ok ? (int64_t(m0 + r) * K + kt * BK + c) : 0), ok);
    }
    #pragma unroll
    for (int v = tid; v < BK * BN / 8; v += THREADS) {
        int r = v / (BN / 8), c = (v % (BN / 8)) * 8;
        bool ok = kt * BK + r < K && n0 + c < N;
        copy16(s + BM * BK + r * BN + (c ^ ((r % (BN / 8)) * 8)),
               b + (ok ? (int64_t(kt * BK + r) * N + n0 + c) : 0), ok);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template<bool BF, int BM, int BN, int BK, int WM, int WN, int STAGES, bool ALT_A = false>
__global__ void gemm_kernel(const unsigned short* __restrict__ a,
    const unsigned short* __restrict__ b, unsigned short* __restrict__ c,
    int M, int N, int K) {
    constexpr int NWN = BN / WN, NT = (BM / WM) * NWN * 32;
    constexpr int AS = WM / 16, BS = WN / 8, SS = (BM + BN) * BK;
    extern __shared__ __align__(16) unsigned short storage[];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int wm = warp / NWN, wn = warp % NWN;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    float acc[AS][BS][4] = {};
    #pragma unroll
    for (int st = 0; st < STAGES - 1; ++st)
        load_stage<BM, BN, BK, NT, ALT_A>(storage + st * SS, a, b, M, N, K, m0, n0, st);

    for (int kt = 0; kt < (K + BK - 1) / BK; ++kt) {
        int next = kt + STAGES - 1;
        load_stage<BM, BN, BK, NT, ALT_A>(storage + (next % STAGES) * SS,
                                    a, b, M, N, K, m0, n0, next);
        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 1) : "memory");
        __syncthreads();
        auto sa = storage + (kt % STAGES) * SS;
        auto sb = sa + BM * BK;
        #pragma unroll
        for (int kh = 0; kh < BK; kh += 16) {
            unsigned ar[AS][4], br[BS][2];
            #pragma unroll
            for (int i = 0; i < AS; ++i) {
                int r = wm * WM + i * 16 + lane % 16;
                int swizzle_row = ALT_A && BK == 32 ? r / 2 : r;
                int col = (kh + (lane / 16) * 8) ^ ((swizzle_row % (BK / 8)) * 8);
                unsigned addr = smem_addr(sa + r * BK + col);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(ar[i][0]), "=r"(ar[i][1]), "=r"(ar[i][2]), "=r"(ar[i][3]) : "r"(addr));
            }
            #pragma unroll
            for (int j = 0; j < BS; ++j) {
                int r = kh + lane % 16;
                int col = (wn * WN + j * 8) ^ ((r % (BN / 8)) * 8);
                unsigned addr = smem_addr(sb + r * BN + col);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                    : "=r"(br[j][0]), "=r"(br[j][1]) : "r"(addr));
            }
            #pragma unroll
            for (int i = 0; i < AS; ++i)
                #pragma unroll
                for (int j = 0; j < BS; ++j) mma<BF>(acc[i][j], ar[i], br[j]);
        }
        __syncthreads();  // All readers finish before the ring slot is reused.
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    #pragma unroll
    for (int i = 0; i < AS; ++i)
        #pragma unroll
        for (int j = 0; j < BS; ++j) {
            int r = m0 + wm * WM + i * 16 + lane / 4;
            int col = n0 + wn * WN + j * 8 + (lane % 4) * 2;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                int rr = r + (p / 2) * 8, cc = col + p % 2;
                if (rr < M && cc < N) c[int64_t(rr) * N + cc] = narrow<BF>(acc[i][j][p]);
            }
        }
}

// Correct scalar fallback for unaligned K/N; deliberately not a fast GEMM path.
template<bool BF> __global__ void fallback_kernel(const unsigned short* a,
    const unsigned short* b, unsigned short* c, int M, int N, int K) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(M) * N) return;
    int m = i / N, n = i % N;
    float sum = 0;
    for (int k = 0; k < K; ++k) sum = fmaf(widen<BF>(a[int64_t(m)*K+k]),
                                         widen<BF>(b[int64_t(k)*N+n]), sum);
    c[i] = narrow<BF>(sum);
}

void validate(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(), "device mismatch");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "2D matrices required");
    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
                "FP16/BF16 inputs required");
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type(), "dtype mismatch");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(a.storage_offset() % 8 == 0 && b.storage_offset() % 8 == 0,
                "input pointers must be 16-byte aligned");
    TORCH_CHECK(a.size(1) == b.size(0) && c.size(0) == a.size(0) && c.size(1) == b.size(1), "shape mismatch");
    TORCH_CHECK(!c.is_alias_of(a) && !c.is_alias_of(b), "output must not alias inputs");
    TORCH_CHECK(a.size(0) <= INT_MAX && a.size(1) <= INT_MAX && b.size(1) <= INT_MAX, "dimensions exceed int32");
}

template<bool BF, int BM, int BN, int BK, int WM, int WN, int ST, bool ALT_A = false>
void launch(const unsigned short* a, const unsigned short* b, unsigned short* c,
            int M, int N, int K, cudaStream_t stream, int extra_smem = 0) {
    constexpr int bytes = (BM + BN) * BK * ST * 2;
    auto fn = gemm_kernel<BF, BM, BN, BK, WM, WN, ST, ALT_A>;
    if constexpr (bytes > 48 * 1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes + extra_smem));
    else if (bytes + extra_smem > 48 * 1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes + extra_smem));
    fn<<<dim3((N+BN-1)/BN, (M+BM-1)/BM), (BM/WM)*(BN/WN)*32, bytes + extra_smem, stream>>>(a,b,c,M,N,K);
}

template<bool BF> void dispatch(torch::Tensor a, torch::Tensor b, torch::Tensor c, int config, int extra_smem = 0) {
    int M=a.size(0), K=a.size(1), N=b.size(1);
    if (!M || !N) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto ap = reinterpret_cast<const unsigned short*>(a.data_ptr());
    auto bp = reinterpret_cast<const unsigned short*>(b.data_ptr());
    auto cp = reinterpret_cast<unsigned short*>(c.data_ptr());
    if (N % 8 || K % 8 || !K) {
        fallback_kernel<BF><<<(int64_t(M)*N+255)/256,256,0,stream>>>(ap,bp,cp,M,N,K);
        return;
    }
    #define CASE(ID, BM, BN, BK, WM, WN, ST) \
        case ID: launch<BF,BM,BN,BK,WM,WN,ST>(ap,bp,cp,M,N,K,stream,extra_smem); break
    switch(config) {
        CASE(0,64,128,32,32,64,3);
        CASE(1,64,128,64,32,64,3);
        CASE(2,128,128,32,32,64,3);
        CASE(3,128,128,64,32,64,3);
        CASE(4,64,64,32,32,32,3);
        CASE(5,64,128,32,32,64,4);
        CASE(6,128,128,32,32,64,4);
        CASE(7,128,256,32,64,64,3);
        CASE(8,64,256,32,32,64,3);
        CASE(9,128,64,32,32,32,3);
        CASE(10,64,128,32,32,64,2);
        CASE(11,64,64,32,32,32,2);
        CASE(12,128,64,32,32,32,2);
        CASE(13,128,128,32,32,64,2);
        CASE(14,128,64,32,32,32,4);
        CASE(15,64,128,32,16,64,3);
        CASE(16,64,128,32,64,32,3);
        CASE(17,64,128,32,64,32,2);
        CASE(18,64,64,32,64,32,3);
        CASE(19,64,64,32,64,32,4);
        CASE(20,64,128,32,16,64,4);
        default: TORCH_CHECK(false,"unknown config");
    }
    #undef CASE
}

void gemm_out(torch::Tensor a, torch::Tensor b, torch::Tensor c, int64_t config) {
    validate(a,b,c);
    TORCH_CHECK(config >= 0 && config < 21, "config must be in [0, 21)");
    const c10::cuda::CUDAGuard guard(a.device());
    TORCH_CHECK(at::cuda::getDeviceProperties(a.get_device())->major >= 8, "SM80+ required");
    if(a.scalar_type()==at::kHalf) dispatch<false>(a,b,c,config);
    else dispatch<true>(a,b,c,config);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Diagnostic only: reserve unused shared memory to limit residency without
// changing the GEMM instruction stream or the pipeline layout.
void gemm_out_smem(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                   int64_t config, int64_t extra_smem) {
    validate(a,b,c);
    TORCH_CHECK(config >= 0 && config < 21, "config must be in [0, 21)");
    TORCH_CHECK(extra_smem >= 0 && extra_smem <= 64 * 1024, "invalid shared memory padding");
    const c10::cuda::CUDAGuard guard(a.device());
    if(a.scalar_type()==at::kHalf) dispatch<false>(a,b,c,config,extra_smem);
    else dispatch<true>(a,b,c,config,extra_smem);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<bool BF, int BM, int BN, int BK, int WM, int WN, int ST, bool ALT_A = false>
pybind11::dict info(int M, int N, int extra) {
    auto fn = gemm_kernel<BF,BM,BN,BK,WM,WN,ST,ALT_A>;
    constexpr int threads = BM/WM * (BN/WN) * 32;
    constexpr int base = (BM+BN)*BK*ST*2;
    if(base+extra > 48*1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,base+extra));
    cudaFuncAttributes attrs;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs,fn));
    int active, device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    auto props=at::cuda::getDeviceProperties(device);
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,threads,base+extra));
    pybind11::dict d;
    d["registers_per_thread"]=attrs.numRegs;
    d["local_bytes_per_thread"]=attrs.localSizeBytes;
    d["static_shared_bytes"]=attrs.sharedSizeBytes;
    d["dynamic_shared_bytes"]=base+extra;
    d["threads_per_block"]=threads;
    d["active_blocks_per_sm_limit"]=active;
    d["sm_count"]=props->multiProcessorCount;
    d["max_threads_per_sm"]=props->maxThreadsPerMultiProcessor;
    d["registers_per_sm"]=props->regsPerMultiprocessor;
    d["shared_bytes_per_sm"]=props->sharedMemPerMultiprocessor;
    int blocks=((M+BM-1)/BM)*((N+BN-1)/BN);
    int capacity=active*props->multiProcessorCount;
    d["grid_blocks"]=blocks;
    d["theoretical_occupancy"]=double(active*threads)/props->maxThreadsPerMultiProcessor;
    d["waves"]=double(blocks)/capacity;
    d["last_wave_fill_model"]=double(blocks)/(((blocks+capacity-1)/capacity)*capacity);
    return d;
}

template<bool BF> pybind11::dict dispatch_info(int config,int M,int N,int extra) {
    #define INFO(ID,BM,BN,BK,WM,WN,ST) case ID: return info<BF,BM,BN,BK,WM,WN,ST>(M,N,extra)
    switch(config) {
        INFO(0,64,128,32,32,64,3); INFO(1,64,128,64,32,64,3);
        INFO(2,128,128,32,32,64,3); INFO(3,128,128,64,32,64,3);
        INFO(4,64,64,32,32,32,3); INFO(5,64,128,32,32,64,4);
        INFO(6,128,128,32,32,64,4); INFO(7,128,256,32,64,64,3);
        INFO(8,64,256,32,32,64,3); INFO(9,128,64,32,32,32,3);
        INFO(10,64,128,32,32,64,2); INFO(11,64,64,32,32,32,2);
        INFO(12,128,64,32,32,32,2); INFO(13,128,128,32,32,64,2);
        INFO(14,128,64,32,32,32,4); INFO(15,64,128,32,16,64,3);
        INFO(16,64,128,32,64,32,3); INFO(17,64,128,32,64,32,2);
        INFO(18,64,64,32,64,32,3); INFO(19,64,64,32,64,32,4);
        INFO(20,64,128,32,16,64,4);
        default: TORCH_CHECK(false,"unknown config");
    }
    #undef INFO
}

pybind11::dict kernel_info(bool bf16,int64_t config,int64_t M,int64_t N,int64_t extra) {
    TORCH_CHECK(M>0 && N>0 && M<=65536 && N<=65536,"invalid diagnostic M/N");
    TORCH_CHECK(extra>=0 && extra<=64*1024,"invalid shared memory padding");
    return bf16 ? dispatch_info<true>(config,M,N,extra) : dispatch_info<false>(config,M,N,extra);
}

// Diagnostic 64x64 stage-count sweep. Normal config IDs stay unchanged.
template<bool BF> void diagnostic_dispatch(torch::Tensor a,torch::Tensor b,
    torch::Tensor c,int variant,int extra) {
    int M=a.size(0),K=a.size(1),N=b.size(1);
    auto ap=reinterpret_cast<const unsigned short*>(a.data_ptr());
    auto bp=reinterpret_cast<const unsigned short*>(b.data_ptr());
    auto cp=reinterpret_cast<unsigned short*>(c.data_ptr());
    auto stream=at::cuda::getCurrentCUDAStream();
    #define DIAG(ID,ST) case ID: launch<BF,64,64,32,32,32,ST>(ap,bp,cp,M,N,K,stream,extra); break
    switch(variant) {
        DIAG(0,2); DIAG(1,3); DIAG(2,4); DIAG(3,5); DIAG(4,6); DIAG(5,8);
        case 6: launch<BF,64,64,32,32,32,2,true>(ap,bp,cp,M,N,K,stream,extra); break;
        case 7: launch<BF,64,64,32,32,32,3,true>(ap,bp,cp,M,N,K,stream,extra); break;
        case 10: launch<BF,64,64,32,32,32,6,true>(ap,bp,cp,M,N,K,stream,extra); break;
        case 12: launch<BF,64,128,32,32,64,2,true>(ap,bp,cp,M,N,K,stream,extra); break;
        case 13: launch<BF,64,128,32,64,32,2,true>(ap,bp,cp,M,N,K,stream,extra); break;
        default: TORCH_CHECK(false,"unknown diagnostic variant");
    }
    #undef DIAG
}
void gemm_diagnostic_out(torch::Tensor a,torch::Tensor b,torch::Tensor c,int64_t variant,int64_t extra) {
    validate(a,b,c);
    TORCH_CHECK(a.size(0)>0 && b.size(1)>0 && a.size(1)>0 &&
                a.size(1)%8==0 && b.size(1)%8==0,"diagnostic requires positive aligned dimensions");
    TORCH_CHECK(extra>=0 && extra<=48*1024,"invalid diagnostic padding");
    const c10::cuda::CUDAGuard guard(a.device());
    if(a.scalar_type()==at::kHalf) diagnostic_dispatch<false>(a,b,c,variant,extra);
    else diagnostic_dispatch<true>(a,b,c,variant,extra);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<bool BF> pybind11::dict diagnostic_info_dispatch(int variant,int M,int N,int extra) {
    #define DIAG_INFO(ID,ST) case ID: return info<BF,64,64,32,32,32,ST>(M,N,extra)
    switch(variant) {
        DIAG_INFO(0,2); DIAG_INFO(1,3); DIAG_INFO(2,4); DIAG_INFO(3,5); DIAG_INFO(4,6); DIAG_INFO(5,8);
        case 6: return info<BF,64,64,32,32,32,2,true>(M,N,extra);
        case 7: return info<BF,64,64,32,32,32,3,true>(M,N,extra);
        case 10: return info<BF,64,64,32,32,32,6,true>(M,N,extra);
        case 12: return info<BF,64,128,32,32,64,2,true>(M,N,extra);
        case 13: return info<BF,64,128,32,64,32,2,true>(M,N,extra);
        default: TORCH_CHECK(false,"unknown diagnostic variant");
    }
    #undef DIAG_INFO
}
pybind11::dict diagnostic_info(bool bf16,int64_t variant,int64_t M,int64_t N,int64_t extra) {
    TORCH_CHECK(M>0 && N>0 && M<=65536 && N<=65536,"invalid diagnostic shape");
    TORCH_CHECK(extra>=0 && extra<=48*1024,"invalid diagnostic padding");
    return bf16 ? diagnostic_info_dispatch<true>(variant,M,N,extra)
                : diagnostic_info_dispatch<false>(variant,M,N,extra);
}

__global__ void empty_kernel() {}
void launch_empty() {
    empty_kernel<<<1,32,0,at::cuda::getCurrentCUDAStream()>>>();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cublas_out(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    validate(a,b,c);
    const c10::cuda::CUDAGuard guard(a.device());
    int M=a.size(0), K=a.size(1), N=b.size(1);
    if (!M || !N) return;
    if (!K) { c.zero_(); return; }
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasMath_t previous;
    TORCH_CHECK(cublasGetMathMode(handle,&previous)==CUBLAS_STATUS_SUCCESS,"get math mode failed");
    TORCH_CHECK(cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)==CUBLAS_STATUS_SUCCESS,"set math mode failed");
    float alpha=1, beta=0;
    auto type=a.scalar_type()==at::kHalf ? CUDA_R_16F : CUDA_R_16BF;
    auto status=cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,
        b.data_ptr(),type,N,a.data_ptr(),type,K,&beta,c.data_ptr(),type,N,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    auto restored=cublasSetMathMode(handle,previous);
    TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"cublasGemmEx failed: ",int(status));
    TORCH_CHECK(restored==CUBLAS_STATUS_SUCCESS,"restore math mode failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_out", &gemm_out, "Custom FP16/BF16 GEMM into a preallocated output");
    m.def("cublas_out", &cublas_out, "cuBLAS GemmEx, FP32 accumulation, same buffers");
    m.def("gemm_out_smem", &gemm_out_smem, "Diagnostic: extra unused shared memory to cap residency");
    m.def("kernel_info", &kernel_info, "Static attributes and CUDA occupancy API residency limit");
    m.def("launch_empty", &launch_empty, "Diagnostic: graph node / launch floor");
    m.def("gemm_diagnostic_out", &gemm_diagnostic_out, "Diagnostic: fixed 64x64 tile stage sweep");
    m.def("diagnostic_info", &diagnostic_info, "Diagnostic kernel resources");
}
