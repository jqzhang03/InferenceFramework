// W4A16: X[M,K] times offset-encoded INT4 W[N,K/2] transposed.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cooperative_groups.h>
#include <limits>
#include <unordered_map>

namespace {

template <bool BF16>
__device__ float to_float(unsigned short value) {
    if constexpr (BF16) return __bfloat162float(__ushort_as_bfloat16(value));
    else return __half2float(__ushort_as_half(value));
}

template <bool BF16>
__device__ unsigned short from_float(float value) {
    if constexpr (BF16) return __bfloat16_as_ushort(__float2bfloat16_rn(value));
    else return __half_as_ushort(__float2half_rn(value));
}

template <bool BF16>
__device__ uint4 dequantize(unsigned packed, float scale) {
    uint4 weights;
    auto* elements = reinterpret_cast<unsigned short*>(&weights);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int signed_value = int(packed & 15) - 8;
        elements[i] = from_float<BF16>(signed_value * scale);
        packed >>= 4;
    }
    return weights;
}

// One eight-lane group owns one output column and reuses its weights over
// the entire small batch. The final warp shuffles include boundary groups.
struct PackedMatrix {
    const unsigned char* weights;
    const unsigned short* scales;
    int group_size;
};

template <bool BF16>
__device__ uint4 load_weights(PackedMatrix matrix, int col, int k, int kk) {
    const unsigned packed = *reinterpret_cast<const unsigned*>(
        matrix.weights + int64_t(col) * (k / 2) + kk / 2);
    const int scale_index = kk / matrix.group_size;
    const float scale = to_float<BF16>(matrix.scales[int64_t(col) * (k / matrix.group_size) + scale_index]);
    return dequantize<BF16>(packed, scale);
}

template <bool BF16>
__device__ unsigned short silu_product(float gate, float up) {
    // Preserve the conversion at each projection and after ATen's SiLU.
    gate = to_float<BF16>(from_float<BF16>(gate));
    up = to_float<BF16>(from_float<BF16>(up));
    float silu = gate / (1.0f + expf(-gate));
    return from_float<BF16>(to_float<BF16>(from_float<BF16>(silu)) * up);
}

template <bool BF16, int ROWS, bool FUSED>
__device__ void gemv_columns(const unsigned short* x, PackedMatrix weight, PackedMatrix up,
                             unsigned short* out, int n, int k, int block, int rows) {
    constexpr int lanes = 8;
    const int lane = threadIdx.x % lanes;
    const int col = block * (blockDim.x / lanes) + threadIdx.x / lanes;
    float even[ROWS] = {}, odd[ROWS] = {};
    float up_even[ROWS] = {}, up_odd[ROWS] = {};
    if (col < n) {
        for (int kk = lane * 8; kk < k; kk += lanes * 8) {
            uint4 w = load_weights<BF16>(weight, col, k, kk);
            uint4 u;
            if constexpr (FUSED) u = load_weights<BF16>(up, col, k, kk);
            const auto* weights = reinterpret_cast<const unsigned short*>(&w);
            const auto* up_weights = reinterpret_cast<const unsigned short*>(&u);
            #pragma unroll
            for (int row = 0; row < ROWS; ++row) {
                uint4 activation = *reinterpret_cast<const uint4*>(x + int64_t(row) * k + kk);
                const auto* a = reinterpret_cast<const unsigned short*>(&activation);
                #pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    const float a_even = to_float<BF16>(a[i]), a_odd = to_float<BF16>(a[i + 1]);
                    even[row] = fmaf(a_even, to_float<BF16>(weights[i]), even[row]);
                    odd[row] = fmaf(a_odd, to_float<BF16>(weights[i + 1]), odd[row]);
                    if constexpr (FUSED) {
                        up_even[row] = fmaf(a_even, to_float<BF16>(up_weights[i]), up_even[row]);
                        up_odd[row] = fmaf(a_odd, to_float<BF16>(up_weights[i + 1]), up_odd[row]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int row = 0; row < ROWS; ++row) {
        float sum = even[row] + odd[row];
        float up_sum = up_even[row] + up_odd[row];
        #pragma unroll
        for (int delta = lanes / 2; delta > 0; delta /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, delta, lanes);
            if constexpr (FUSED) up_sum += __shfl_down_sync(0xffffffff, up_sum, delta, lanes);
        }
        if (row < rows && lane == 0 && col < n) {
            const auto value = FUSED ? silu_product<BF16>(sum, up_sum) : from_float<BF16>(sum);
            out[int64_t(row) * n + col] = value;
        }
    }
}

template <bool BF16, int ROWS>
__global__ void w4a16_gemv_kernel(const unsigned short* x, const unsigned char* weights,
                                const unsigned short* scales, unsigned short* out,
                                int n, int k, int group_size, int rows) {
    gemv_columns<BF16, ROWS, false>(x, {weights, scales, group_size}, {}, out, n, k, blockIdx.x, rows);
}

template <bool BF16, int ROWS>
__global__ void w4a16_gate_up_kernel(const unsigned short* x, PackedMatrix gate, PackedMatrix up,
                                    unsigned short* hidden, int intermediate, int width, int rows) {
    gemv_columns<BF16, ROWS, true>(x, gate, up, hidden, intermediate, width, blockIdx.x, rows);
}

// Each CTA handles the same column tiles as the two-launch path. The
// cooperative grid barrier publishes the intermediate before any CTA reads it.
template <bool BF16, int ROWS>
__global__ void w4a16_expert_kernel(const unsigned short* x, PackedMatrix gate, PackedMatrix up,
                                   PackedMatrix down, unsigned short* hidden, unsigned short* out,
                                   int intermediate, int width, int gate_tiles, int down_tiles,
                                   int rows) {
    for (int tile = blockIdx.x; tile < gate_tiles; tile += gridDim.x)
        gemv_columns<BF16, ROWS, true>(x, gate, up, hidden, intermediate, width, tile, rows);
    cooperative_groups::this_grid().sync();
    for (int tile = blockIdx.x; tile < down_tiles; tile += gridDim.x)
        gemv_columns<BF16, ROWS, false>(hidden, down, {}, out, width, intermediate, tile, rows);
}

// Route-major compact dispatch: the router sorts slots by expert id and writes
// device-side offsets/counts. One 2D grid covers every expert, so the host never
// reads a route back. Rows beyond counts[e] are masked in the store; the padded
// tail of x/hidden keeps their loads in bounds.
template <bool BF16, int ROWS>
__global__ void w4a16_compact_gate_up_kernel(
        const unsigned short* x,
        const int64_t* gate_w_ptrs, const int64_t* gate_s_ptrs,
        const int64_t* up_w_ptrs, const int64_t* up_s_ptrs,
        const int* offsets, const int* counts,
        unsigned short* hidden, int intermediate, int width, int group_size,
        int tiles) {
    const int expert = blockIdx.x;
    const int start = static_cast<int>(blockIdx.y) * ROWS;
    const int total = counts[expert];
    if (start >= total) return;
    const int rows = min(ROWS, total - start);
    const int64_t offset = offsets[expert] + start;
    PackedMatrix gate{
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(gate_w_ptrs[expert])),
        reinterpret_cast<const unsigned short*>(static_cast<uintptr_t>(gate_s_ptrs[expert])),
        group_size};
    PackedMatrix up{
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(up_w_ptrs[expert])),
        reinterpret_cast<const unsigned short*>(static_cast<uintptr_t>(up_s_ptrs[expert])),
        group_size};
    for (int tile = blockIdx.z; tile < tiles; tile += gridDim.z)
        gemv_columns<BF16, ROWS, true>(
            x + offset * width, gate, up, hidden + offset * intermediate,
            intermediate, width, tile, rows);
}

template <bool BF16, int ROWS>
__global__ void w4a16_compact_down_kernel(
        const unsigned short* hidden,
        const int64_t* down_w_ptrs, const int64_t* down_s_ptrs,
        const int* offsets, const int* counts,
        unsigned short* out, int width, int intermediate, int group_size,
        int tiles) {
    const int expert = blockIdx.x;
    const int start = static_cast<int>(blockIdx.y) * ROWS;
    const int total = counts[expert];
    if (start >= total) return;
    const int rows = min(ROWS, total - start);
    const int64_t offset = offsets[expert] + start;
    PackedMatrix down{
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(down_w_ptrs[expert])),
        reinterpret_cast<const unsigned short*>(static_cast<uintptr_t>(down_s_ptrs[expert])),
        group_size};
    for (int tile = blockIdx.z; tile < tiles; tile += gridDim.z)
        gemv_columns<BF16, ROWS, false>(
            hidden + offset * intermediate, down, {}, out + offset * width,
            width, intermediate, tile, rows);
}

template <bool BF16>
__device__ void mma(float (&acc)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    if constexpr (BF16) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}

constexpr int block_k = 64;

__device__ int swizzled(int row, int k) {
    return row * block_k + (k ^ ((row % 8) * 8));
}

// Two warps own 16 output channels each. Compute W @ X^T so an eight-row
// request tile fills the narrow MMA axis, then store X @ W^T in row-major.
template <bool BF16, int BLOCK_M>
__global__ void w4a16_mma_kernel(const unsigned short* x, const unsigned char* weights,
                               const unsigned short* scales, unsigned short* out,
                               int m, int n, int k, int group_size, int n_tiles) {
    constexpr int block_n = 32, threads = 64;
    __shared__ __align__(16) unsigned short tile[(block_n + BLOCK_M) * block_k];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int n0 = (blockIdx.x % n_tiles) * block_n;
    const int m0 = (blockIdx.x / n_tiles) * BLOCK_M;
    const int groups = k / group_size;
    float accum[BLOCK_M / 8][4] = {};

    for (int k0 = 0; k0 < k; k0 += block_k) {
        #pragma unroll
        for (int index = threadIdx.x; index < block_n * (block_k / 8); index += threads) {
            const int row = index / (block_k / 8), col = (index % (block_k / 8)) * 8;
            const int channel = n0 + row, kk = k0 + col;
            uint4 values = {};
            if (channel < n && kk < k) {
                unsigned packed = *reinterpret_cast<const unsigned*>(
                    weights + int64_t(channel) * (k / 2) + kk / 2);
                float scale = to_float<BF16>(scales[int64_t(channel) * groups + kk / group_size]);
                values = dequantize<BF16>(packed, scale);
            }
            *reinterpret_cast<uint4*>(tile + swizzled(row, col)) = values;
        }
        #pragma unroll
        for (int index = threadIdx.x; index < BLOCK_M * (block_k / 8); index += threads) {
            const int row = index / (block_k / 8), col = (index % (block_k / 8)) * 8;
            const int request = m0 + row, kk = k0 + col;
            uint4 values = {};
            if (request < m && kk < k) {
                values = *reinterpret_cast<const uint4*>(x + int64_t(request) * k + kk);
            }
            *reinterpret_cast<uint4*>(tile + swizzled(block_n + row, col)) = values;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < block_k; kk += 16) {
            unsigned a[4];
            unsigned address = __cvta_generic_to_shared(
                tile + swizzled(warp * 16 + lane % 16, kk + (lane / 16) * 8));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(address));
            #pragma unroll
            for (int r = 0; r < BLOCK_M / 8; ++r) {
                unsigned b[2];
                address = __cvta_generic_to_shared(
                    tile + swizzled(block_n + r * 8 + lane % 8, kk + ((lane / 8) % 2) * 8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(b[0]), "=r"(b[1]) : "r"(address));
                mma<BF16>(accum[r], a, b);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < BLOCK_M / 8; ++r) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = n0 + warp * 16 + lane / 4 + (i / 2) * 8;
            const int row = m0 + r * 8 + (lane % 4) * 2 + i % 2;
            if (row < m && col < n) out[int64_t(row) * n + col] = from_float<BF16>(accum[r][i]);
        }
    }
}

template <bool BF16, int BLOCK_M>
void launch_mma(const unsigned short* x, const unsigned char* w, const unsigned short* s,
                unsigned short* out, int m, int n, int k, int group, cudaStream_t stream) {
    const int n_tiles = (n - 1) / 32 + 1;
    const int64_t ctas = int64_t(n_tiles) * ((m - 1) / BLOCK_M + 1);
    TORCH_CHECK(ctas <= std::numeric_limits<int>::max(), "W4A16 grid is too large");
    w4a16_mma_kernel<BF16, BLOCK_M><<<ctas, 64, 0, stream>>>(x, w, s, out, m, n, k, group, n_tiles);
}

template <bool BF16>
void launch(const unsigned short* x, const unsigned char* w, const unsigned short* s,
            unsigned short* out, int m, int n, int k, int group, cudaStream_t stream) {
    if (m <= 4) {
        const int grid = (n - 1) / 16 + 1;
        switch (m) {
            case 1: w4a16_gemv_kernel<BF16, 1><<<grid, 128, 0, stream>>>(x, w, s, out, n, k, group, 1); break;
            case 2: w4a16_gemv_kernel<BF16, 2><<<grid, 128, 0, stream>>>(x, w, s, out, n, k, group, 2); break;
            case 3: w4a16_gemv_kernel<BF16, 3><<<grid, 128, 0, stream>>>(x, w, s, out, n, k, group, 3); break;
            case 4: w4a16_gemv_kernel<BF16, 4><<<grid, 128, 0, stream>>>(x, w, s, out, n, k, group, 4); break;
        }
    } else if (m <= 8) {
        launch_mma<BF16, 8>(x, w, s, out, m, n, k, group, stream);
    } else if (m <= 16) {
        launch_mma<BF16, 16>(x, w, s, out, m, n, k, group, stream);
    } else {
        launch_mma<BF16, 32>(x, w, s, out, m, n, k, group, stream);
    }
}

PackedMatrix checked_matrix(const at::Tensor& x, const at::Tensor& packed,
                           const at::Tensor& scales, const at::Tensor& out) {
    for (const auto* tensor : {&x, &packed, &scales, &out}) {
        TORCH_CHECK(tensor->is_cuda() && tensor->device() == x.device() &&
                    tensor->dim() == 2 && tensor->is_contiguous(),
                    "W4A16 expects contiguous rank-two tensors on one CUDA device");
    }
    const auto type = x.scalar_type();
    TORCH_CHECK((type == at::kHalf || type == at::kBFloat16) &&
                scales.scalar_type() == type && out.scalar_type() == type &&
                packed.scalar_type() == at::kChar, "invalid W4A16 tensor dtypes");
    const auto m = x.size(0), k = x.size(1), n = packed.size(0), groups = scales.size(1);
    constexpr auto max_size = std::numeric_limits<int>::max() - block_k;
    TORCH_CHECK(m <= max_size && n <= max_size && k <= max_size &&
                k > 0 && k % 8 == 0 && packed.size(1) == k / 2 &&
                scales.size(0) == n && out.size(0) == m && out.size(1) == n &&
                groups > 0 && k % groups == 0, "invalid W4A16 tensor shapes");
    const int group = k / groups;
    TORCH_CHECK(group >= 8 && 128 % group == 0, "invalid W4A16 K-group size");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(packed.data_ptr()) % 4 == 0,
                "W4A16 requires aligned activations and packed weights");
    return {reinterpret_cast<const unsigned char*>(packed.data_ptr()),
            reinterpret_cast<const unsigned short*>(scales.data_ptr()), group};
}

void gemm_out(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& scales,
              at::Tensor& out) {
    const PackedMatrix matrix = checked_matrix(x, packed, scales, out);
    const int m = x.size(0), k = x.size(1), n = packed.size(0);
    if (m == 0 || n == 0) return;
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    const auto* a = reinterpret_cast<const unsigned short*>(x.data_ptr());
    auto* c = reinterpret_cast<unsigned short*>(out.data_ptr());
    if (x.scalar_type() == at::kBFloat16)
        launch<true>(a, matrix.weights, matrix.scales, c, m, n, k, matrix.group_size, stream);
    else
        launch<false>(a, matrix.weights, matrix.scales, c, m, n, k, matrix.group_size, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool BF16, int ROWS>
bool launch_expert(const unsigned short* x, PackedMatrix gate, PackedMatrix up,
                   PackedMatrix down, unsigned short* hidden, unsigned short* out,
                   int intermediate, int width, bool full, int device, cudaStream_t stream) {
    constexpr int threads = 128;
    int gate_tiles = (intermediate - 1) / 16 + 1;
    int down_tiles = (width - 1) / 16 + 1;
    if (!full) {
        w4a16_gate_up_kernel<BF16, ROWS><<<gate_tiles, threads, 0, stream>>>(
            x, gate, up, hidden, intermediate, width, ROWS);
        w4a16_gemv_kernel<BF16, ROWS><<<down_tiles, threads, 0, stream>>>(
            hidden, down.weights, down.scales, out, width, intermediate, down.group_size, ROWS);
        return true;
    }
    const auto* properties = at::cuda::getDeviceProperties(device);
    if (!properties->cooperativeLaunch) return false;
    static thread_local std::unordered_map<int, int> grid_limits;
    auto [limit, inserted] = grid_limits.try_emplace(device, 0);
    if (inserted) {
        int blocks_per_sm = 0;
        const auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, w4a16_expert_kernel<BF16, ROWS>, threads, 0);
        if (status != cudaSuccess) grid_limits.erase(limit);
        C10_CUDA_CHECK(status);
        limit->second = blocks_per_sm * properties->multiProcessorCount;
    }
    const int blocks = std::min(std::max(gate_tiles, down_tiles), limit->second);
    if (!blocks) return false;
    int rows = ROWS;
    void* args[] = {&x, &gate, &up, &down, &hidden, &out,
                    &intermediate, &width, &gate_tiles, &down_tiles, &rows};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(w4a16_expert_kernel<BF16, ROWS>), blocks, threads, args, 0, stream));
    return true;
}

template <bool BF16>
bool dispatch_expert(const at::Tensor& x, PackedMatrix gate, PackedMatrix up, PackedMatrix down,
                     at::Tensor& hidden, at::Tensor& out, bool full) {
    const int device = x.get_device();
    const auto stream = at::cuda::getCurrentCUDAStream(device);
    const auto* a = reinterpret_cast<const unsigned short*>(x.data_ptr());
    auto* h = reinterpret_cast<unsigned short*>(hidden.data_ptr());
    auto* y = reinterpret_cast<unsigned short*>(out.data_ptr());
    const int intermediate = hidden.size(1), width = x.size(1);
    switch (x.size(0)) {
        case 1: return launch_expert<BF16, 1>(a, gate, up, down, h, y, intermediate, width, full, device, stream);
        case 2: return launch_expert<BF16, 2>(a, gate, up, down, h, y, intermediate, width, full, device, stream);
        case 3: return launch_expert<BF16, 3>(a, gate, up, down, h, y, intermediate, width, full, device, stream);
        case 4: return launch_expert<BF16, 4>(a, gate, up, down, h, y, intermediate, width, full, device, stream);
        default: return false;
    }
}

template <bool BF16, int ROWS>
void launch_compact(const unsigned short* x,
                    const int64_t* gate_w_ptrs, const int64_t* gate_s_ptrs,
                    const int64_t* up_w_ptrs, const int64_t* up_s_ptrs,
                    const int64_t* down_w_ptrs, const int64_t* down_s_ptrs,
                    const int* offsets, const int* counts, unsigned short* hidden,
                    unsigned short* out, int experts, int max_rows, int width,
                    int intermediate, int group_size, cudaStream_t stream) {
    const int gate_tiles = (intermediate - 1) / 16 + 1;
    const int down_tiles = (width - 1) / 16 + 1;
    const int chunks = (max_rows + ROWS - 1) / ROWS;
    // Decode experts are tiny; cap the per-expert CTA fan-out so empty experts
    // do not launch the full 88/128-tile grid. Each CTA strides over the rest.
    const int gate_ctas = std::min(gate_tiles, 16);
    const int down_ctas = std::min(down_tiles, 16);
    const dim3 gate_grid(experts, chunks, gate_ctas);
    const dim3 down_grid(experts, chunks, down_ctas);
    w4a16_compact_gate_up_kernel<BF16, ROWS><<<gate_grid, 128, 0, stream>>>(
        x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs,
        offsets, counts, hidden, intermediate, width, group_size, gate_tiles);
    w4a16_compact_down_kernel<BF16, ROWS><<<down_grid, 128, 0, stream>>>(
        hidden, down_w_ptrs, down_s_ptrs, offsets, counts,
        out, width, intermediate, group_size, down_tiles);
}

template <bool BF16>
void launch_compact_rows(int rows_cap, const unsigned short* x,
                         const int64_t* gate_w_ptrs, const int64_t* gate_s_ptrs,
                         const int64_t* up_w_ptrs, const int64_t* up_s_ptrs,
                         const int64_t* down_w_ptrs, const int64_t* down_s_ptrs,
                         const int* offsets, const int* counts, unsigned short* hidden,
                         unsigned short* out, int experts, int max_rows, int width,
                         int intermediate, int group_size, cudaStream_t stream) {
    switch (rows_cap) {
        case 1: launch_compact<BF16, 1>(x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs,
                                        down_w_ptrs, down_s_ptrs, offsets, counts, hidden,
                                        out, experts, max_rows, width, intermediate,
                                        group_size, stream); break;
        case 2: launch_compact<BF16, 2>(x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs,
                                        down_w_ptrs, down_s_ptrs, offsets, counts, hidden,
                                        out, experts, max_rows, width, intermediate,
                                        group_size, stream); break;
        case 4: launch_compact<BF16, 4>(x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs,
                                        down_w_ptrs, down_s_ptrs, offsets, counts, hidden,
                                        out, experts, max_rows, width, intermediate,
                                        group_size, stream); break;
        case 8: launch_compact<BF16, 8>(x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs,
                                        down_w_ptrs, down_s_ptrs, offsets, counts, hidden,
                                        out, experts, max_rows, width, intermediate,
                                        group_size, stream); break;
    }
}

bool expert_out_compact(const at::Tensor& x,
                        const at::Tensor& gate_w_ptrs, const at::Tensor& gate_s_ptrs,
                        const at::Tensor& up_w_ptrs, const at::Tensor& up_s_ptrs,
                        const at::Tensor& down_w_ptrs, const at::Tensor& down_s_ptrs,
                        const at::Tensor& offsets, const at::Tensor& counts,
                        at::Tensor& hidden, at::Tensor& out, int64_t max_rows,
                        int64_t logical_rows, int64_t group_size) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(),
                "compact W4A16 expects contiguous rank-two CUDA activations");
    TORCH_CHECK(hidden.is_cuda() && hidden.dim() == 2 && hidden.is_contiguous() &&
                out.is_cuda() && out.dim() == 2 && out.is_contiguous(),
                "compact W4A16 expects contiguous rank-two CUDA outputs");
    const auto type = x.scalar_type();
    TORCH_CHECK((type == at::kHalf || type == at::kBFloat16) &&
                hidden.scalar_type() == type && out.scalar_type() == type,
                "invalid compact W4A16 activation dtype");
    TORCH_CHECK(offsets.is_cuda() && counts.is_cuda() &&
                offsets.scalar_type() == at::kInt && counts.scalar_type() == at::kInt,
                "compact W4A16 expects int32 CUDA offsets/counts");
    TORCH_CHECK(offsets.dim() == 1 && counts.dim() == 1 &&
                offsets.numel() == counts.numel() && offsets.is_contiguous() &&
                counts.is_contiguous(), "invalid compact W4A16 offsets/counts");
    for (const auto* tensor : {&gate_w_ptrs, &gate_s_ptrs, &up_w_ptrs,
                               &up_s_ptrs, &down_w_ptrs, &down_s_ptrs}) {
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kLong &&
                    tensor->dim() == 1 && tensor->numel() == counts.numel() &&
                    tensor->is_contiguous(),
                    "compact W4A16 expects one int64 pointer per expert");
    }
    const int experts = counts.numel();
    if (experts == 0) return true;
    const int rows = static_cast<int>(logical_rows);
    const int width = x.size(1);
    const int intermediate = hidden.size(1);
    TORCH_CHECK(rows >= 0 && rows <= x.size(0), "invalid compact W4A16 logical row count");
    TORCH_CHECK(out.size(1) == width && hidden.size(0) >= rows && out.size(0) >= rows,
                "invalid compact W4A16 output shapes");
    TORCH_CHECK(width > 0 && intermediate > 0 && width % 8 == 0 && intermediate % 8 == 0,
                "compact W4A16 expects dimensions divisible by 8");
    const int rows_cap = max_rows <= 1 ? 1 : max_rows <= 2 ? 2 : max_rows <= 4 ? 4
                      : max_rows <= 8 ? 8 : 0;
    TORCH_CHECK(max_rows > 0 && rows_cap > 0, "compact W4A16 supports at most 8 rows per expert");
    TORCH_CHECK(x.size(0) >= rows + rows_cap &&
                hidden.size(0) >= rows + rows_cap && out.size(0) >= rows + rows_cap,
                "compact W4A16 buffer padding is too small");
    const int group = static_cast<int>(group_size);
    TORCH_CHECK(group >= 8 && 128 % group == 0 &&
                width % group == 0 && intermediate % group == 0,
                "invalid compact W4A16 INT4 group size");
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    const auto* x_ptr = reinterpret_cast<const unsigned short*>(x.data_ptr());
    auto* hidden_ptr = reinterpret_cast<unsigned short*>(hidden.data_ptr());
    auto* out_ptr = reinterpret_cast<unsigned short*>(out.data_ptr());
    const auto* gate_w = gate_w_ptrs.data_ptr<int64_t>();
    const auto* gate_s = gate_s_ptrs.data_ptr<int64_t>();
    const auto* up_w = up_w_ptrs.data_ptr<int64_t>();
    const auto* up_s = up_s_ptrs.data_ptr<int64_t>();
    const auto* down_w = down_w_ptrs.data_ptr<int64_t>();
    const auto* down_s = down_s_ptrs.data_ptr<int64_t>();
    const auto* offsets_ptr = offsets.data_ptr<int>();
    const auto* counts_ptr = counts.data_ptr<int>();
    if (x.scalar_type() == at::kBFloat16)
        launch_compact_rows<true>(rows_cap, x_ptr, gate_w, gate_s, up_w, up_s,
                                  down_w, down_s, offsets_ptr, counts_ptr, hidden_ptr,
                                  out_ptr, experts, static_cast<int>(max_rows), width,
                                  intermediate, group, stream);
    else
        launch_compact_rows<false>(rows_cap, x_ptr, gate_w, gate_s, up_w, up_s,
                                   down_w, down_s, offsets_ptr, counts_ptr, hidden_ptr,
                                   out_ptr, experts, static_cast<int>(max_rows), width,
                                   intermediate, group, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

bool expert_out(const at::Tensor& x, const at::Tensor& gate_weight, const at::Tensor& gate_scale,
                const at::Tensor& up_weight, const at::Tensor& up_scale,
                const at::Tensor& down_weight, const at::Tensor& down_scale,
                at::Tensor& hidden, at::Tensor& out, bool full) {
    const auto gate = checked_matrix(x, gate_weight, gate_scale, hidden);
    const auto up = checked_matrix(x, up_weight, up_scale, hidden);
    const auto down = checked_matrix(hidden, down_weight, down_scale, out);
    TORCH_CHECK(out.size(1) == x.size(1), "W4A16 expert must preserve the hidden width");
    const c10::cuda::CUDAGuard guard(x.device());
    const bool launched = x.scalar_type() == at::kBFloat16
        ? dispatch_expert<true>(x, gate, up, down, hidden, out, full)
        : dispatch_expert<false>(x, gate, up, down, hidden, out, full);
    if (launched) C10_CUDA_KERNEL_LAUNCH_CHECK();
    return launched;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("gemm_out", &gemm_out, "Fused packed-INT4 x FP16/BF16 GEMM");
    module.def("expert_out", &expert_out, "Fused packed-INT4 expert FFN for M1-4");
    module.def("expert_out_compact", &expert_out_compact,
               "Route-major packed-INT4 expert FFN with device offsets");
}
