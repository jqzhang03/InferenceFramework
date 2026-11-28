// Absorbed DeepSeek-V2-Lite attention over a paged [512 latent | 64 RoPE] cache.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <mma.h>
#include <algorithm>

namespace {
namespace wmma = nvcuda::wmma;

constexpr int heads = 16;
constexpr int latent = 512;
constexpr int width = latent + 64;
constexpr int tile_tokens = 16;
constexpr int warps = 8;
constexpr int threads = 32 * warps;
constexpr int value_tiles = latent / (warps * 8);
constexpr int min_split_tokens = 32;
constexpr int vectors_per_tile = tile_tokens * width / 8;
constexpr int vectors_per_thread = (vectors_per_tile + threads - 1) / threads;
constexpr int async_bytes = vectors_per_tile * sizeof(uint4);

enum class CacheType { fp16, bf16, fp8 };
enum class Pipeline { original, padded, registers, async };

__device__ uint4 convert_bf16(const uint4& raw, float scale) {
    uint4 packed;
    auto* pairs = reinterpret_cast<half2*>(&packed);
    const auto* in = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 values = __bfloat1622float2(in[i]);
        pairs[i] = __floats2half2_rn(values.x * scale, values.y * scale);
    }
    return packed;
}

template <CacheType TYPE>
__device__ uint4 load_cache(const void* cache, int64_t index, float scale) {
    uint4 packed;
    if constexpr (TYPE == CacheType::fp16) {
        packed = *reinterpret_cast<const uint4*>(static_cast<const half*>(cache) + index);
        if (scale == 1.0f) return packed;
        auto* pairs = reinterpret_cast<half2*>(&packed);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 values = __half22float2(pairs[i]);
            pairs[i] = __floats2half2_rn(values.x * scale, values.y * scale);
        }
    } else {
        auto* pairs = reinterpret_cast<half2*>(&packed);
        if constexpr (TYPE == CacheType::bf16) {
            const uint4 raw = *reinterpret_cast<const uint4*>(
                static_cast<const __nv_bfloat16*>(cache) + index);
            return convert_bf16(raw, scale);
        } else {
            const auto raw = *reinterpret_cast<const uint64_t*>(
                static_cast<const uint8_t*>(cache) + index);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                __nv_fp8_e4m3 low, high;
                low.__x = raw >> (16 * i);
                high.__x = raw >> (16 * i + 8);
                pairs[i] = __floats2half2_rn(static_cast<float>(low) * scale,
                                             static_cast<float>(high) * scale);
            }
        }
    }
    return packed;
}

__device__ int64_t tile_base(const int* pages, int token, int page_size) {
    return int64_t(pages[token / page_size]) * page_size + token % page_size;
}

__device__ void prefetch_registers(uint4 (&raw)[vectors_per_thread], const uint4* cache,
                                   const int* pages, int token, int length, int page_size) {
    const int64_t base = tile_base(pages, token, page_size) * (width / 8);
    #pragma unroll
    for (int chunk = 0; chunk < vectors_per_thread; ++chunk) {
        const int v = threadIdx.x + chunk * threads;
        uint4 data{};
        if (v < vectors_per_tile && token + v / (width / 8) < length) {
            asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                         : "l"(cache + base + v) : "memory");
        }
        raw[chunk] = data;
    }
}

__device__ void prefetch_async(uint4* raw, const uint4* cache, const int* pages,
                              int token, int length, int page_size) {
    const int64_t base = tile_base(pages, token, page_size) * (width / 8);
    for (int v = threadIdx.x; v < vectors_per_tile; v += threads) {
        const int bytes = token + v / (width / 8) < length ? 16 : 0;
        const unsigned address = __cvta_generic_to_shared(raw + v);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                     :: "r"(address), "l"(cache + base + (bytes ? v : 0)), "r"(bytes) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ int active_splits(int length, int limit) {
    return min(limit, (max(length, 0) + min_split_tokens - 1) / min_split_tokens);
}

// PTX C/D fragment: lane / 4 owns head h and h+8, each at two adjacent columns.
__device__ void value_mma(float (&acc)[4], const unsigned (&a)[4],
                           unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <CacheType TYPE, Pipeline PIPE>
__global__ void mla_decode_cuda_kernel(
        const half* query, const void* cache, const int* pages, const int* lengths,
        half* out, float* partials, float* lse, int page_size, int page_columns,
        int split_limit, float softmax_scale, float c_scale, float r_scale) {
    const int request = blockIdx.x, split = blockIdx.y;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int length = lengths[request];
    const int splits = active_splits(length, split_limit);
    const int64_t request_out = int64_t(request) * heads * latent;
    if (!splits && split == 0) {
        for (int i = threadIdx.x; i < heads * latent; i += threads)
            out[request_out + i] = __float2half_rn(0.0f);
    }
    if (split >= splits) return;

    constexpr int query_stride = PIPE == Pipeline::original ? width : width + 8;
    constexpr int cache_stride = PIPE == Pipeline::original ? width + 16 : width + 8;
    extern __shared__ __align__(16) uint4 async_tile[];
    __shared__ __align__(32) half q[heads * query_stride];
    __shared__ __align__(32) half kv[tile_tokens * cache_stride];
    __shared__ __align__(32) float scores[warps][heads * tile_tokens];
    __shared__ __align__(32) half probs[heads * tile_tokens];
    __shared__ float running_max[heads], normalizer[heads], correction[heads];

    for (int i = threadIdx.x * 8; i < heads * width; i += threads * 8) {
        const auto* source = query + int64_t(request) * heads * width + i;
        *reinterpret_cast<uint4*>(q + (i / width) * query_stride + i % width) =
            *reinterpret_cast<const uint4*>(source);
    }
    if (threadIdx.x < heads) {
        running_max[threadIdx.x] = -CUDART_INF_F;
        normalizer[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    const int tiles = (length + tile_tokens - 1) / tile_tokens;
    const int begin = tiles * split / splits, end = tiles * (split + 1) / splits;
    const int head = lane / 4, pair_start = (lane % 4) * 2;
    const int* request_pages = pages + int64_t(request) * page_columns;
    const auto* raw_cache = static_cast<const uint4*>(cache);
    uint4 prefetched[vectors_per_thread];
    if constexpr (PIPE == Pipeline::registers) {
        prefetch_registers(prefetched, raw_cache, request_pages, begin * tile_tokens, length, page_size);
    } else if constexpr (PIPE == Pipeline::async) {
        prefetch_async(async_tile, raw_cache, request_pages, begin * tile_tokens, length, page_size);
    }
    float accum[value_tiles][4] = {};
    for (int tile = begin; tile < end; ++tile) {
        const int first_token = tile * tile_tokens;
        if constexpr (PIPE == Pipeline::registers || PIPE == Pipeline::async) {
            if constexpr (PIPE == Pipeline::async) {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            // Each thread converts the same vectors it prefetched. The CTA
            // barrier below publishes the converted FP16 tile to all warps.
            #pragma unroll
            for (int chunk = 0; chunk < vectors_per_thread; ++chunk) {
                const int i = (threadIdx.x + chunk * threads) * 8;
                if (i < tile_tokens * width) {
                    const int row = i / width, col = i % width;
                    const uint4 raw = PIPE == Pipeline::async ? async_tile[i / 8] : prefetched[chunk];
                    *reinterpret_cast<uint4*>(kv + row * cache_stride + col) =
                        convert_bf16(raw, col < latent ? c_scale : r_scale);
                }
            }
        } else {
            const int64_t first_slot = tile_base(request_pages, first_token, page_size);
            for (int i = threadIdx.x * 8; i < tile_tokens * width; i += threads * 8) {
                const int row = i / width, col = i % width;
                uint4 value = {};
                if (first_token + row < length) {
                    value = load_cache<TYPE>(cache, (first_slot + row) * width + col,
                                             col < latent ? c_scale : r_scale);
                }
                *reinterpret_cast<uint4*>(kv + row * cache_stride + col) = value;
            }
        }
        __syncthreads();
        if (tile + 1 < end) {
            if constexpr (PIPE == Pipeline::registers) {
                prefetch_registers(prefetched, raw_cache, request_pages,
                                   first_token + tile_tokens, length, page_size);
            } else if constexpr (PIPE == Pipeline::async) {
                prefetch_async(async_tile, raw_cache, request_pages,
                               first_token + tile_tokens, length, page_size);
            }
        }

        // Eight warps divide the QK reduction over the 576-element head dimension.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> score;
        wmma::fill_fragment(score, 0.0f);
        for (int col = warp * 16; col < width; col += warps * 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
            wmma::load_matrix_sync(qa, q + col, query_stride);
            wmma::load_matrix_sync(kb, kv + col, cache_stride);
            wmma::mma_sync(score, qa, kb, score);
        }
        wmma::store_matrix_sync(scores[warp], score, tile_tokens, wmma::mem_row_major);
        __syncthreads();

        const int h = threadIdx.x / tile_tokens, t = threadIdx.x % tile_tokens;
        float logit = 0.0f;
        #pragma unroll
        for (int w = 0; w < warps; ++w)
            logit += scores[w][threadIdx.x];
        logit = first_token + t < length ? logit * softmax_scale : -CUDART_INF_F;
        float tile_max = logit;
        #pragma unroll
        for (int shift = 8; shift > 0; shift /= 2)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, shift, tile_tokens));
        const float old_max = running_max[h];
        const float new_max = fmaxf(tile_max, old_max);
        const float alpha = __expf(old_max - new_max);
        const float p = __expf(logit - new_max);
        float sum = p;
        #pragma unroll
        for (int shift = 8; shift > 0; shift /= 2)
            sum += __shfl_xor_sync(0xffffffff, sum, shift, tile_tokens);
        probs[threadIdx.x] = __float2half_rn(p);
        __syncwarp();
        if (t == 0) {
            running_max[h] = new_max;
            normalizer[h] = normalizer[h] * alpha + sum;
            correction[h] = alpha;
        }
        __syncthreads();

        unsigned a[4];
        const unsigned p_address = __cvta_generic_to_shared(
            probs + (lane % 16) * tile_tokens + (lane / 16) * 8);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(p_address));
        #pragma unroll
        for (int v = 0; v < value_tiles; ++v) {
            const int col = (v * warps + warp) * 8;
            const unsigned kv_address = __cvta_generic_to_shared(
                kv + (lane % 16) * cache_stride + col);
            unsigned b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                         : "=r"(b0), "=r"(b1) : "r"(kv_address));
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                accum[v][i] *= correction[head + (i / 2) * 8];
            value_mma(accum[v], a, b0, b1);
        }
        __syncthreads();
    }

    const int64_t partial_index = int64_t(request) * split_limit + split;
    if (splits > 1 && threadIdx.x < heads)
        lse[partial_index * heads + threadIdx.x] =
            running_max[threadIdx.x] + logf(normalizer[threadIdx.x]);
    const float inverse[2] = {1.0f / normalizer[head], 1.0f / normalizer[head + 8]};
    #pragma unroll
    for (int v = 0; v < value_tiles; ++v) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int h = head + (i / 2) * 8;
            const int col = (v * warps + warp) * 8 + pair_start + i % 2;
            const float value = accum[v][i] * inverse[i / 2];
            const int index = h * latent + col;
            if (splits == 1) out[request_out + index] = __float2half_rn(value);
            else partials[partial_index * heads * latent + index] = value;
        }
    }
}

__global__ void mla_decode_combine(const float* partials, const float* lse, const int* lengths,
                                  half* out, int split_limit) {
    const int request = blockIdx.x, head = blockIdx.y;
    const int splits = active_splits(lengths[request], split_limit);
    if (splits <= 1) return;
    const int64_t base = int64_t(request) * split_limit * heads + head;
    __shared__ float weight[16];
    if (threadIdx.x < 32) {
        const int part = threadIdx.x;
        const float stat = part < splits ? lse[base + part * heads] : -CUDART_INF_F;
        float maximum = stat;
        for (int offset = 16; offset > 0; offset /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        const float e = __expf(stat - maximum);
        float sum = e;
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        if (part < splits) weight[part] = e / sum;
    }
    __syncthreads();
    for (int col = threadIdx.x; col < latent; col += blockDim.x) {
        float value = 0.0f;
        for (int part = 0; part < splits; ++part)
            value = fmaf(weight[part], partials[(base + part * heads) * latent + col], value);
        out[(int64_t(request) * heads + head) * latent + col] = __float2half_rn(value);
    }
}

void prepare_async(int device) {
    const c10::cuda::CUDAGuard guard(device);
    C10_CUDA_CHECK(cudaFuncSetAttribute(mla_decode_cuda_kernel<CacheType::bf16, Pipeline::async>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, async_bytes));
}

at::Tensor decode(const at::Tensor& query, const at::Tensor& cache, const at::Tensor& pages,
                  const at::Tensor& lengths, double softmax_scale, double c_scale, double r_scale,
                  const std::string& variant) {
    for (const auto* tensor : {&query, &cache, &pages, &lengths})
        TORCH_CHECK(tensor->is_cuda() && tensor->device() == query.device() &&
                    tensor->is_contiguous(), "CUDA MLA expects contiguous tensors on one CUDA device");
    TORCH_CHECK(query.scalar_type() == at::kHalf && query.dim() == 3 &&
                query.size(1) == heads && query.size(2) == width &&
                cache.dim() == 3 && cache.size(2) == width && cache.size(1) > 0 &&
                cache.size(1) % tile_tokens == 0 &&
                pages.dim() == 2 && lengths.dim() == 1 && pages.size(0) == query.size(0) &&
                lengths.size(0) == query.size(0) &&
                pages.scalar_type() == at::kInt && lengths.scalar_type() == at::kInt,
                "invalid CUDA MLA query, cache, or paging metadata");
    const auto type = cache.scalar_type();
    TORCH_CHECK(type == at::kHalf || type == at::kBFloat16 || type == at::kFloat8_e4m3fn,
                "CUDA MLA cache must be FP16, BF16 or E4M3FN");
    TORCH_CHECK(variant == "auto" || variant == "original" || variant == "padded" ||
                variant == "registers" || variant == "async", "unknown CUDA MLA variant");
    TORCH_CHECK(type == at::kBFloat16 || (variant != "registers" && variant != "async"),
                "CUDA MLA prefetch variants require BF16 KV");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(query.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(cache.data_ptr()) % 16 == 0,
                "CUDA MLA query and cache must be 16-byte aligned");
    const c10::cuda::CUDAGuard guard(query.device());
    const int batch = query.size(0);
    auto out = at::empty({batch, heads, latent}, query.options());
    if (!batch) return out;
    const int page_size = cache.size(1), page_columns = pages.size(1);
    const int capacity_splits = std::min<int64_t>(
        16, (pages.size(1) * cache.size(1) + min_split_tokens - 1) / min_split_tokens);
    const int batch_limit = batch <= 8 ? 16 : batch <= 16 ? 8 : batch <= 64 ? 4 : batch <= 128 ? 2 : 1;
    const int splits = std::max(1, std::min(batch_limit, capacity_splits));
    auto scratch = query.options().dtype(at::kFloat);
    auto partial = splits > 1 ? at::empty({batch, splits, heads, latent}, scratch) : at::Tensor();
    auto lse = splits > 1 ? at::empty({batch, splits, heads}, scratch) : at::Tensor();
    auto* partial_ptr = splits > 1 ? partial.data_ptr<float>() : nullptr;
    auto* lse_ptr = splits > 1 ? lse.data_ptr<float>() : nullptr;
    auto* output = reinterpret_cast<half*>(out.data_ptr());
    const auto* q = reinterpret_cast<const half*>(query.data_ptr());
    const auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
    const dim3 grid(batch, splits);
    const auto launch = [&](auto kernel, int dynamic_shared = 0) {
        kernel<<<grid, threads, dynamic_shared, stream>>>(
            q, cache.data_ptr(), pages.data_ptr<int>(), lengths.data_ptr<int>(),
            output, partial_ptr, lse_ptr, page_size, page_columns, splits,
            float(softmax_scale), float(c_scale), float(r_scale));
    };
    if (type == at::kBFloat16) {
        const auto* properties = at::cuda::getDeviceProperties(query.get_device());
        // The register pipeline was validated on SM120.
        const bool prefetch = variant == "registers" ||
            (variant == "auto" && properties->major == 12 && properties->minor == 0);
        if (variant == "async")
            launch(mla_decode_cuda_kernel<CacheType::bf16, Pipeline::async>, async_bytes);
        else if (prefetch)
            launch(mla_decode_cuda_kernel<CacheType::bf16, Pipeline::registers>);
        else if (variant == "original")
            launch(mla_decode_cuda_kernel<CacheType::bf16, Pipeline::original>);
        else
            launch(mla_decode_cuda_kernel<CacheType::bf16, Pipeline::padded>);
    } else if (type == at::kHalf) {
        if (variant == "original") launch(mla_decode_cuda_kernel<CacheType::fp16, Pipeline::original>);
        else launch(mla_decode_cuda_kernel<CacheType::fp16, Pipeline::padded>);
    } else {
        if (variant == "original") launch(mla_decode_cuda_kernel<CacheType::fp8, Pipeline::original>);
        else launch(mla_decode_cuda_kernel<CacheType::fp8, Pipeline::padded>);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (splits > 1) {
        mla_decode_combine<<<dim3(batch, heads), 128, 0, stream>>>(
            partial_ptr, lse_ptr, lengths.data_ptr<int>(), output, splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    namespace py = pybind11;
    module.def("decode", &decode, "CUDA paged latent attention for DeepSeek-V2-Lite",
               py::arg("query"), py::arg("cache"), py::arg("pages"), py::arg("lengths"),
               py::arg("softmax_scale"), py::arg("c_scale"), py::arg("r_scale"),
               py::arg("variant") = "auto");
    module.def("prepare_async", &prepare_async, "Opt in to the async variant's shared-memory size");
}
