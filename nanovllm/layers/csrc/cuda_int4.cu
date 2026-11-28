// W4A16: X[M,K] times offset-encoded INT4 W[N,K/2] transposed.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
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

// One eight-lane group owns one output column and reuses its weights over
// the entire small batch. The final warp shuffles include boundary groups.
struct PackedMatrix {
    const unsigned char* weights;
    const unsigned short* scales;
    int group_size;
    float tensor_scale = 1.0f;
};

template <int FP4_GROUP>
__device__ float decode_nibble(unsigned code) {
    if constexpr (FP4_GROUP == 0) return int(code) - 8;
    // E2M1: exponent zero encodes 0 and 0.5, then binary normals.
    const unsigned exp = (code >> 1) & 3;
    const float magnitude = exp == 0 ? (code & 1) * 0.5f
        : __uint_as_float(((exp + 126) << 23) | ((code & 1) << 22));
    return (code & 8) ? -magnitude : magnitude;
}

template <bool BF16, int FP4_GROUP = 0>
__device__ uint4 dequantize(unsigned packed, float scale) {
    uint4 weights;
    auto* elements = reinterpret_cast<unsigned short*>(&weights);
    // Pair the eight scalar int4->float->bf16 conversions into four packed
    // cvt.rn.bf16x2.f32 / cvt.rn.f16x2.f32 instructions. The low half still
    // receives the lower nibble first, so every element and its rounding are
    // identical to the scalar path.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float low = decode_nibble<FP4_GROUP>(packed & 15);
        const float high = decode_nibble<FP4_GROUP>((packed >> 4) & 15);
        packed >>= 8;
        const float f0 = low * scale;
        const float f1 = high * scale;
        if constexpr (BF16) {
            *reinterpret_cast<__nv_bfloat162*>(elements + 2 * i) =
                __floats2bfloat162_rn(f0, f1);
        } else {
            *reinterpret_cast<__half2*>(elements + 2 * i) =
                __floats2half2_rn(f0, f1);
        }
    }
    return weights;
}

// Raw packed word and scale, split out so gemv_columns can prefetch the next
// K chunk before consuming the current dequantised tile.
template <bool BF16>
__device__ unsigned load_packed(PackedMatrix matrix, int col, int k, int kk) {
    return *reinterpret_cast<const unsigned*>(
        matrix.weights + int64_t(col) * (k / 2) + kk / 2);
}

template <bool BF16, int FP4_GROUP = 0>
__device__ float load_scale(PackedMatrix matrix, int col, int k, int kk) {
    if constexpr (FP4_GROUP == 0) {
        return to_float<BF16>(
            matrix.scales[int64_t(col) * (k / matrix.group_size) + kk / matrix.group_size]);
    } else {
        const auto* scales = reinterpret_cast<const uint8_t*>(matrix.scales);
        const uint8_t scale = scales[int64_t(col) * (k / FP4_GROUP) + kk / FP4_GROUP];
        if constexpr (FP4_GROUP == 16) {
            __nv_fp8_e4m3 e4m3;
            e4m3.__x = scale;
            return static_cast<float>(e4m3) * matrix.tensor_scale;
        } else {
            // Code 0 is 2^-127 (subnormal in FP32), and 255 is NaN.
            const unsigned fraction = (scale == 0 || scale == 255) ? 0x00400000 : 0;
            return __uint_as_float((unsigned(scale) << 23) | fraction);
        }
    }
}

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

template <bool BF16, int ROWS, bool FUSED, int FP4_GROUP = 0>
__device__ void gemv_columns(const unsigned short* x, PackedMatrix weight, PackedMatrix up,
                             unsigned short* out, int n, int k, int block, int rows) {
    constexpr int lanes = 8;
    const int lane = threadIdx.x % lanes;
    const int col = block * (blockDim.x / lanes) + threadIdx.x / lanes;
    float even[ROWS] = {}, odd[ROWS] = {};
    float up_even[ROWS] = {}, up_odd[ROWS] = {};
    if (col < n) {
        constexpr int stride = lanes * 8;
        int kk = lane * 8;
        unsigned packed = 0, up_packed = 0;
        float scale = 0.0f, up_scale = 0.0f;
        if (kk < k) {
            packed = load_packed<BF16>(weight, col, k, kk);
            scale = load_scale<BF16, FP4_GROUP>(weight, col, k, kk);
            if constexpr (FUSED) {
                up_packed = load_packed<BF16>(up, col, k, kk);
                up_scale = load_scale<BF16, FP4_GROUP>(up, col, k, kk);
            }
        }
        for (; kk < k; kk += stride) {
            // One K-chunk lookahead: the next packed word/scale is in flight
            // while the current tile is dequantised and applied to all rows.
            const int next = kk + stride;
            unsigned next_packed = 0, next_up_packed = 0;
            float next_scale = 0.0f, next_up_scale = 0.0f;
            if (next < k) {
                next_packed = load_packed<BF16>(weight, col, k, next);
                next_scale = load_scale<BF16, FP4_GROUP>(weight, col, k, next);
                if constexpr (FUSED) {
                    next_up_packed = load_packed<BF16>(up, col, k, next);
                    next_up_scale = load_scale<BF16, FP4_GROUP>(up, col, k, next);
                }
            }
            uint4 w = dequantize<BF16, FP4_GROUP>(packed, scale);
            uint4 u;
            if constexpr (FUSED) u = dequantize<BF16, FP4_GROUP>(up_packed, up_scale);
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
            packed = next_packed;
            scale = next_scale;
            up_packed = next_up_packed;
            up_scale = next_up_scale;
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
    // 5-8 rows follow the reference MMA dispatch; keep this kernel to <=4 so
    // both paths disjointly cover the decode range.
    if (rows > 4) return;
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
    if (rows > 4) return;
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
constexpr int block_n = 32;

__device__ int swizzled(int row, int k) {
    return row * block_k + (k ^ ((row % 8) * 8));
}

// Compute W @ X^T so an eight-row request tile fills the narrow MMA axis, then
// store X @ W^T in row-major. Kept as a device function so the per-expert
// reference kernel and the route-major compact kernel execute exactly the same
// arithmetic.
//
// THREADS warps are grouped into `col_groups` channel pairs (16 channels each)
// and `warps / col_groups` row groups. The output tile is block_n channels x
// BLOCK_M rows either way; a wider CTA amortises the weight dequantisation over
// more rows, and a narrower one keeps more CTAs resident.
template <bool BF16, int BLOCK_M, int THREADS, int FP4_GROUP = 0>
__device__ void mma_tile(const unsigned short* x, const unsigned char* weights,
                         const unsigned short* scales, unsigned short* out,
                         int m, int n, int k, int group_size, int m0, int n0,
                         unsigned short* tile, float tensor_scale = 1.0f) {
    constexpr int warps = THREADS / 32;
    constexpr int col_groups = block_n / 16;
    constexpr int row_groups = warps / col_groups;
    constexpr int rows_per_warp = BLOCK_M / row_groups;
    constexpr int row_tiles = rows_per_warp / 8;
    static_assert(THREADS % 32 == 0 && warps % col_groups == 0,
                  "mma_tile needs whole warp groups");
    static_assert(row_groups >= 1 && rows_per_warp % 8 == 0 && row_tiles >= 1,
                  "mma_tile needs at least one eight-row tile per warp");
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int col_group = warp % col_groups;
    const int row_offset = (warp / col_groups) * rows_per_warp;
    const PackedMatrix matrix{weights, scales, group_size, tensor_scale};
    float accum[row_tiles][4] = {};

    for (int k0 = 0; k0 < k; k0 += block_k) {
        #pragma unroll
        for (int index = threadIdx.x; index < block_n * (block_k / 8); index += THREADS) {
            const int row = index / (block_k / 8), col = (index % (block_k / 8)) * 8;
            const int channel = n0 + row, kk = k0 + col;
            uint4 values = {};
            if (channel < n && kk < k) {
                const unsigned packed = load_packed<BF16>(matrix, channel, k, kk);
                const float scale = load_scale<BF16, FP4_GROUP>(matrix, channel, k, kk);
                values = dequantize<BF16, FP4_GROUP>(packed, scale);
            }
            *reinterpret_cast<uint4*>(tile + swizzled(row, col)) = values;
        }
        #pragma unroll
        for (int index = threadIdx.x; index < BLOCK_M * (block_k / 8); index += THREADS) {
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
                tile + swizzled(col_group * 16 + lane % 16, kk + (lane / 16) * 8));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(address));
            #pragma unroll
            for (int r = 0; r < row_tiles; ++r) {
                unsigned b[2];
                address = __cvta_generic_to_shared(
                    tile + swizzled(block_n + row_offset + r * 8 + lane % 8,
                                    kk + ((lane / 8) % 2) * 8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(b[0]), "=r"(b[1]) : "r"(address));
                mma<BF16>(accum[r], a, b);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < row_tiles; ++r) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = n0 + col_group * 16 + lane / 4 + (i / 2) * 8;
            const int row = m0 + row_offset + r * 8 + (lane % 4) * 2 + i % 2;
            if (row < m && col < n) out[int64_t(row) * n + col] = from_float<BF16>(accum[r][i]);
        }
    }
}

// Gate/up share the same 32-column output tile and the same activation rows.
// Four warps split the two projections (warps 0-1 gate, 2-3 up), load the
// activation tile once, and each warp keeps one accumulator so the arithmetic
// of every output stays identical to two mma_tile calls.
//
// FUSE_SILU keeps the same MMA schedule but moves the activation into the
// epilogue: the gate warps stage their rounded values in shared memory, then
// the up warps -- which own the same 16 output channels of the same tile --
// apply silu(gate)*up and write the single hidden tile. This drops the two
// global intermediate stores and the separate elementwise kernels without
// changing the rounding, because silu_product rounds the up accumulator exactly
// as the split path's store did and the staged gate value is already BF16.
template <bool BF16, int BLOCK_M, bool FUSE_SILU>
__device__ void mma_gate_up_tile(const unsigned short* x,
                                 const unsigned char* gate_w,
                                 const unsigned short* gate_s,
                                 const unsigned char* up_w,
                                 const unsigned short* up_s,
                                 unsigned short* gate_out,
                                 unsigned short* up_out,
                                 int m, int n, int k, int group_size,
                                 int m0, int n0, unsigned short* tile,
                                 unsigned short* silu_scratch) {
    constexpr int threads = 128;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const bool gate_warp = warp < 2;
    const int proj_warp = warp % 2;
    const int groups = k / group_size;
    float acc[BLOCK_M / 8][4] = {};
    unsigned short* gate_tile = tile;
    unsigned short* up_tile = tile + block_n * block_k;
    unsigned short* x_tile = tile + 2 * block_n * block_k;

    for (int k0 = 0; k0 < k; k0 += block_k) {
        #pragma unroll
        for (int index = threadIdx.x; index < block_n * (block_k / 8);
             index += threads) {
            const int row = index / (block_k / 8), col = (index % (block_k / 8)) * 8;
            const int channel = n0 + row, kk = k0 + col;
            uint4 gate_values = {}, up_values = {};
            if (channel < n && kk < k) {
                unsigned gate_packed = *reinterpret_cast<const unsigned*>(
                    gate_w + int64_t(channel) * (k / 2) + kk / 2);
                unsigned up_packed = *reinterpret_cast<const unsigned*>(
                    up_w + int64_t(channel) * (k / 2) + kk / 2);
                gate_values = dequantize<BF16>(
                    gate_packed,
                    to_float<BF16>(gate_s[int64_t(channel) * groups + kk / group_size]));
                up_values = dequantize<BF16>(
                    up_packed,
                    to_float<BF16>(up_s[int64_t(channel) * groups + kk / group_size]));
            }
            *reinterpret_cast<uint4*>(gate_tile + swizzled(row, col)) = gate_values;
            *reinterpret_cast<uint4*>(up_tile + swizzled(row, col)) = up_values;
        }
        #pragma unroll
        for (int index = threadIdx.x; index < BLOCK_M * (block_k / 8);
             index += threads) {
            const int row = index / (block_k / 8), col = (index % (block_k / 8)) * 8;
            const int request = m0 + row, kk = k0 + col;
            uint4 values = {};
            if (request < m && kk < k) {
                values = *reinterpret_cast<const uint4*>(x + int64_t(request) * k + kk);
            }
            *reinterpret_cast<uint4*>(x_tile + swizzled(row, col)) = values;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < block_k; kk += 16) {
            unsigned a[4];
            const unsigned short* a_tile = gate_warp ? gate_tile : up_tile;
            unsigned address = __cvta_generic_to_shared(
                a_tile + swizzled(proj_warp * 16 + lane % 16,
                                  kk + (lane / 16) * 8));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                         "{%0,%1,%2,%3}, [%4];"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                         : "r"(address));
            #pragma unroll
            for (int r = 0; r < BLOCK_M / 8; ++r) {
                unsigned b[2];
                address = __cvta_generic_to_shared(
                    x_tile + swizzled(r * 8 + lane % 8,
                                      kk + ((lane / 8) % 2) * 8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                             "{%0,%1}, [%2];"
                             : "=r"(b[0]), "=r"(b[1]) : "r"(address));
                mma<BF16>(acc[r], a, b);
            }
        }
        __syncthreads();
    }

    if constexpr (FUSE_SILU) {
        // Both warp pairs cover the identical (row, channel) set, so the gate
        // half publishes through a shared scratch tile. The staging store uses
        // the same BF16 rounding as the split path's global store.
        if (gate_warp) {
            #pragma unroll
            for (int r = 0; r < BLOCK_M / 8; ++r) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int col = proj_warp * 16 + lane / 4 + (i / 2) * 8;
                    const int row = r * 8 + (lane % 4) * 2 + i % 2;
                    silu_scratch[row * block_n + col] = from_float<BF16>(acc[r][i]);
                }
            }
        }
        __syncthreads();
        if (!gate_warp) {
            #pragma unroll
            for (int r = 0; r < BLOCK_M / 8; ++r) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int col = proj_warp * 16 + lane / 4 + (i / 2) * 8;
                    const int row = r * 8 + (lane % 4) * 2 + i % 2;
                    const int channel = n0 + col;
                    const int request = m0 + row;
                    if (request < m && channel < n) {
                        gate_out[int64_t(request) * n + channel] =
                            silu_product<BF16>(
                                to_float<BF16>(silu_scratch[row * block_n + col]),
                                acc[r][i]);
                    }
                }
            }
        }
        return;
    }
    unsigned short* out = gate_warp ? gate_out : up_out;
    #pragma unroll
    for (int r = 0; r < BLOCK_M / 8; ++r) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = n0 + proj_warp * 16 + lane / 4 + (i / 2) * 8;
            const int row = m0 + r * 8 + (lane % 4) * 2 + i % 2;
            if (row < m && col < n) {
                out[int64_t(row) * n + col] = from_float<BF16>(acc[r][i]);
            }
        }
    }
}

// Fused gate/up tile for the 32-row template. Each warp owns both projections
// for eight rows x 32 channels, so ``silu(gate)*up`` is computed from registers
// and no cross-warp staging, shared roundtrip, or extra barrier is needed. The
// MMA K order per output element is unchanged, so the fused values are bitwise
// the split path's. Registers and shared bytes match the split kernel: four
// accumulators per warp and the same (2*block_n + BLOCK_M) x block_k tile.
template <bool BF16>
__device__ void mma_gate_up_silu_wide_tile(const unsigned short* x,
                                           const unsigned char* gate_w,
                                           const unsigned short* gate_s,
                                           const unsigned char* up_w,
                                           const unsigned short* up_s,
                                           unsigned short* hidden,
                                           int m, int n, int k, int group_size,
                                           int m0, int n0, unsigned short* tile) {
    constexpr int threads = 128;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int groups = k / group_size;
    // Four warps take one eight-row group each and the full 32-channel tile of
    // both projections, so every output thread holds gate and up together.
    const int row_group = warp;
    float gate_acc[2][4] = {}, up_acc[2][4] = {};
    unsigned short* gate_tile = tile;
    unsigned short* up_tile = tile + block_n * block_k;
    unsigned short* x_tile = tile + 2 * block_n * block_k;

    for (int k0 = 0; k0 < k; k0 += block_k) {
        #pragma unroll
        for (int index = threadIdx.x; index < block_n * (block_k / 8);
             index += threads) {
            const int row = index / (block_k / 8), col = (index % (block_k / 8)) * 8;
            const int channel = n0 + row, kk = k0 + col;
            uint4 gate_values = {}, up_values = {};
            if (channel < n && kk < k) {
                unsigned gate_packed = *reinterpret_cast<const unsigned*>(
                    gate_w + int64_t(channel) * (k / 2) + kk / 2);
                unsigned up_packed = *reinterpret_cast<const unsigned*>(
                    up_w + int64_t(channel) * (k / 2) + kk / 2);
                gate_values = dequantize<BF16>(
                    gate_packed,
                    to_float<BF16>(gate_s[int64_t(channel) * groups + kk / group_size]));
                up_values = dequantize<BF16>(
                    up_packed,
                    to_float<BF16>(up_s[int64_t(channel) * groups + kk / group_size]));
            }
            *reinterpret_cast<uint4*>(gate_tile + swizzled(row, col)) = gate_values;
            *reinterpret_cast<uint4*>(up_tile + swizzled(row, col)) = up_values;
        }
        #pragma unroll
        for (int index = threadIdx.x; index < block_n * (block_k / 8);
             index += threads) {
            const int row = index / (block_k / 8), col = (index % (block_k / 8)) * 8;
            const int request = m0 + row, kk = k0 + col;
            uint4 values = {};
            if (request < m && kk < k) {
                values = *reinterpret_cast<const uint4*>(x + int64_t(request) * k + kk);
            }
            *reinterpret_cast<uint4*>(x_tile + swizzled(row, col)) = values;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < block_k; kk += 16) {
            unsigned gate_a[2][4], up_a[2][4];
            #pragma unroll
            for (int ct = 0; ct < 2; ++ct) {
                unsigned address = __cvta_generic_to_shared(
                    gate_tile + swizzled(ct * 16 + lane % 16,
                                         kk + (lane / 16) * 8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                             "{%0,%1,%2,%3}, [%4];"
                             : "=r"(gate_a[ct][0]), "=r"(gate_a[ct][1]),
                               "=r"(gate_a[ct][2]), "=r"(gate_a[ct][3])
                             : "r"(address));
                address = __cvta_generic_to_shared(
                    up_tile + swizzled(ct * 16 + lane % 16,
                                       kk + (lane / 16) * 8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                             "{%0,%1,%2,%3}, [%4];"
                             : "=r"(up_a[ct][0]), "=r"(up_a[ct][1]),
                               "=r"(up_a[ct][2]), "=r"(up_a[ct][3])
                             : "r"(address));
            }
            unsigned b[2];
            unsigned address = __cvta_generic_to_shared(
                x_tile + swizzled(row_group * 8 + lane % 8,
                                  kk + ((lane / 8) % 2) * 8));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                         "{%0,%1}, [%2];"
                         : "=r"(b[0]), "=r"(b[1]) : "r"(address));
            #pragma unroll
            for (int ct = 0; ct < 2; ++ct) {
                mma<BF16>(gate_acc[ct], gate_a[ct], b);
                mma<BF16>(up_acc[ct], up_a[ct], b);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ct = 0; ct < 2; ++ct) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int channel = n0 + ct * 16 + lane / 4 + (i / 2) * 8;
            const int request = m0 + row_group * 8 + (lane % 4) * 2 + i % 2;
            if (request < m && channel < n) {
                hidden[int64_t(request) * n + channel] =
                    silu_product<BF16>(gate_acc[ct][i], up_acc[ct][i]);
            }
        }
    }
}

template <bool BF16, int BLOCK_M, int THREADS = 64>
__global__ void w4a16_mma_kernel(const unsigned short* x, const unsigned char* weights,
                               const unsigned short* scales, unsigned short* out,
                               int m, int n, int k, int group_size, int n_tiles) {
    __shared__ __align__(16) unsigned short tile[(block_n + BLOCK_M) * block_k];
    const int n0 = (blockIdx.x % n_tiles) * block_n;
    const int m0 = (blockIdx.x / n_tiles) * BLOCK_M;
    mma_tile<BF16, BLOCK_M, THREADS>(x, weights, scales, out, m, n, k, group_size, m0, n0, tile);
}

// Route-major MMA projection for experts whose segment has 5-8 rows. Experts
// with at most four rows stay on the GEMV compact kernel; the two paths cover
// disjoint counts and together reproduce the per-expert reference dispatch.
template <bool BF16, int BLOCK_M>
__global__ void w4a16_compact_mma_kernel(
        const unsigned short* x, const int64_t* weight_ptrs, const int64_t* scale_ptrs,
        const int* offsets, const int* counts, unsigned short* out,
        int n, int k, int group_size, int n_tiles, int min_count, int max_count) {
    __shared__ __align__(16) unsigned short tile[(block_n + BLOCK_M) * block_k];    const int expert = blockIdx.x;
    const int total = counts[expert];
    if (total < min_count || total > max_count) return;
    const auto* weights = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(weight_ptrs[expert]));
    const auto* scales = reinterpret_cast<const unsigned short*>(
        static_cast<uintptr_t>(scale_ptrs[expert]));
    for (int start = static_cast<int>(blockIdx.y) * BLOCK_M; start < total;
         start += gridDim.y * BLOCK_M) {
        const int rows = min(BLOCK_M, total - start);
        const int64_t offset = offsets[expert] + start;
        for (int tile_index = blockIdx.z; tile_index < n_tiles;
             tile_index += gridDim.z) {
            const int n0 = tile_index * block_n;
            mma_tile<BF16, BLOCK_M, 64>(x + offset * k, weights, scales,
                                        out + offset * n, rows, n, k, group_size,
                                        0, n0, tile);
        }
    }
}

template <bool BF16, int BLOCK_M>
__global__ void w4a16_compact_mma_gate_up_kernel(
        const unsigned short* x,
        const int64_t* gate_w_ptrs, const int64_t* gate_s_ptrs,
        const int64_t* up_w_ptrs, const int64_t* up_s_ptrs,
        const int* offsets, const int* counts, unsigned short* gate_out,
        unsigned short* up_out, int n, int k, int group_size, int n_tiles,
        int min_count, int max_count) {
    __shared__ __align__(16)
        unsigned short tile[(2 * block_n + BLOCK_M) * block_k];
    const int expert = blockIdx.x;
    const int total = counts[expert];
    if (total < min_count || total > max_count) return;
    const auto* gate_w = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(gate_w_ptrs[expert]));
    const auto* gate_s = reinterpret_cast<const unsigned short*>(
        static_cast<uintptr_t>(gate_s_ptrs[expert]));
    const auto* up_w = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(up_w_ptrs[expert]));
    const auto* up_s = reinterpret_cast<const unsigned short*>(
        static_cast<uintptr_t>(up_s_ptrs[expert]));
    for (int start = static_cast<int>(blockIdx.y) * BLOCK_M; start < total;
         start += gridDim.y * BLOCK_M) {
        const int rows = min(BLOCK_M, total - start);
        const int64_t offset = offsets[expert] + start;
        for (int tile_index = blockIdx.z; tile_index < n_tiles;
             tile_index += gridDim.z) {
            const int n0 = tile_index * block_n;
            mma_gate_up_tile<BF16, BLOCK_M, false>(
                x + offset * k, gate_w, gate_s, up_w, up_s,
                gate_out + offset * n, up_out + offset * n, rows, n, k,
                group_size, 0, n0, tile, nullptr);
        }
    }
}

// Single-output variant of the gate/up kernel: hidden = silu(gate)*up. The
// route-major grid, the per-expert chunk walk, and the MMA K order are
// unchanged; only the epilogue differs. The 32-row template keeps the same
// four-accumulator register budget as the split kernel by giving each warp both
// projections for its own eight rows; 8/16-row templates keep one projection
// per warp pair and stage the gate half through shared memory.
template <bool BF16, int BLOCK_M>
__global__ void w4a16_compact_mma_gate_up_silu_kernel(
        const unsigned short* x,
        const int64_t* gate_w_ptrs, const int64_t* gate_s_ptrs,
        const int64_t* up_w_ptrs, const int64_t* up_s_ptrs,
        const int* offsets, const int* counts, unsigned short* hidden,
        int n, int k, int group_size, int n_tiles, int min_count, int max_count) {
    __shared__ __align__(16)
        unsigned short tile[(2 * block_n + BLOCK_M) * block_k];
    __shared__ __align__(16)
        unsigned short gate_scratch[BLOCK_M == 32 ? 1 : block_n * BLOCK_M];
    const int expert = blockIdx.x;
    const int total = counts[expert];
    if (total < min_count || total > max_count) return;
    const auto* gate_w = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(gate_w_ptrs[expert]));
    const auto* gate_s = reinterpret_cast<const unsigned short*>(
        static_cast<uintptr_t>(gate_s_ptrs[expert]));
    const auto* up_w = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(up_w_ptrs[expert]));
    const auto* up_s = reinterpret_cast<const unsigned short*>(
        static_cast<uintptr_t>(up_s_ptrs[expert]));
    for (int start = static_cast<int>(blockIdx.y) * BLOCK_M; start < total;
         start += gridDim.y * BLOCK_M) {
        const int rows = min(BLOCK_M, total - start);
        const int64_t offset = offsets[expert] + start;
        for (int tile_index = blockIdx.z; tile_index < n_tiles;
             tile_index += gridDim.z) {
            const int n0 = tile_index * block_n;
            if constexpr (BLOCK_M == 32) {
                mma_gate_up_silu_wide_tile<BF16>(
                    x + offset * k, gate_w, gate_s, up_w, up_s,
                    hidden + offset * n, rows, n, k, group_size, 0, n0, tile);
            } else {
                mma_gate_up_tile<BF16, BLOCK_M, true>(
                    x + offset * k, gate_w, gate_s, up_w, up_s,
                    hidden + offset * n, nullptr, rows, n, k, group_size, 0,
                    n0, tile, gate_scratch);
            }
        }
    }
}

// 5-16 row band: one launch covers both the 5-8 and 9-16 row ranges and picks
// the 8/16-row template from counts[expert]. The old dispatch issued two grids
// per projection, and the 8-row grid carried an empty second chunk for every
// expert below nine rows. BLOCK_M=16 is the largest template in this band, so
// its shared tile bounds both branches.
template <bool BF16>
__global__ void w4a16_compact_mma_band_kernel(
        const unsigned short* x, const int64_t* weight_ptrs, const int64_t* scale_ptrs,
        const int* offsets, const int* counts, unsigned short* out,
        int n, int k, int group_size, int n_tiles, int min_count, int max_count) {
    __shared__ __align__(16) unsigned short tile[(block_n + 16) * block_k];
    const int expert = blockIdx.x;
    const int total = counts[expert];
    if (total < min_count || total > max_count) return;
    const auto* weights = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(weight_ptrs[expert]));
    const auto* scales = reinterpret_cast<const unsigned short*>(
        static_cast<uintptr_t>(scale_ptrs[expert]));
    const int64_t offset = offsets[expert];
    for (int tile_index = blockIdx.z; tile_index < n_tiles; tile_index += gridDim.z) {
        const int n0 = tile_index * block_n;
        if (total <= 8) {
            mma_tile<BF16, 8, 64>(x + offset * k, weights, scales, out + offset * n,
                                  total, n, k, group_size, 0, n0, tile);
        } else {
            mma_tile<BF16, 16, 64>(x + offset * k, weights, scales, out + offset * n,
                                   total, n, k, group_size, 0, n0, tile);
        }
    }
}

template <bool BF16, bool FUSE_SILU>
__global__ void __launch_bounds__(128, 8) w4a16_compact_mma_gate_up_band_kernel(
        const unsigned short* x,
        const int64_t* gate_w_ptrs, const int64_t* gate_s_ptrs,
        const int64_t* up_w_ptrs, const int64_t* up_s_ptrs,
        const int* offsets, const int* counts, unsigned short* gate_out,
        unsigned short* up_out, int n, int k, int group_size, int n_tiles,
        int min_count, int max_count) {
    __shared__ __align__(16)
        unsigned short tile[(2 * block_n + 16) * block_k];
    __shared__ __align__(16) unsigned short gate_scratch[block_n * 16];
    const int expert = blockIdx.x;
    const int total = counts[expert];
    if (total < min_count || total > max_count) return;
    const auto* gate_w = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(gate_w_ptrs[expert]));
    const auto* gate_s = reinterpret_cast<const unsigned short*>(
        static_cast<uintptr_t>(gate_s_ptrs[expert]));
    const auto* up_w = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(up_w_ptrs[expert]));
    const auto* up_s = reinterpret_cast<const unsigned short*>(
        static_cast<uintptr_t>(up_s_ptrs[expert]));
    const int64_t offset = offsets[expert];
    for (int tile_index = blockIdx.z; tile_index < n_tiles; tile_index += gridDim.z) {
        const int n0 = tile_index * block_n;
        if (total <= 8) {
            mma_gate_up_tile<BF16, 8, FUSE_SILU>(
                x + offset * k, gate_w, gate_s, up_w, up_s,
                gate_out + offset * n, up_out == nullptr ? nullptr : up_out + offset * n,
                total, n, k, group_size, 0, n0, tile, gate_scratch);
        } else {
            mma_gate_up_tile<BF16, 16, FUSE_SILU>(
                x + offset * k, gate_w, gate_s, up_w, up_s,
                gate_out + offset * n, up_out == nullptr ? nullptr : up_out + offset * n,
                total, n, k, group_size, 0, n0, tile, gate_scratch);
        }
    }
}

template <bool BF16, int BLOCK_M, int THREADS>
void launch_mma(const unsigned short* x, const unsigned char* w, const unsigned short* s,
                unsigned short* out, int m, int n, int k, int group, cudaStream_t stream) {
    const int n_tiles = (n - 1) / 32 + 1;
    const int64_t ctas = int64_t(n_tiles) * ((m - 1) / BLOCK_M + 1);
    TORCH_CHECK(ctas <= std::numeric_limits<int>::max(), "W4A16 grid is too large");
    w4a16_mma_kernel<BF16, BLOCK_M, THREADS>
        <<<ctas, THREADS, 0, stream>>>(x, w, s, out, m, n, k, group, n_tiles);
}

// Tile/warp variants for the dense INT4 projection. Variant 0 is the historical
// two-warp, 32-row tile; the others widen the row tile so one weight
// dequantisation feeds more rows (the kernel re-dequantises weights per CTA).
template <bool BF16>
void launch_mma_variant(int variant, const unsigned short* x, const unsigned char* w,
                        const unsigned short* s, unsigned short* out, int m, int n,
                        int k, int group, cudaStream_t stream) {
    switch (variant) {
        case 0: launch_mma<BF16, 32, 64>(x, w, s, out, m, n, k, group, stream); break;
        case 1: launch_mma<BF16, 64, 128>(x, w, s, out, m, n, k, group, stream); break;
        case 2: launch_mma<BF16, 128, 128>(x, w, s, out, m, n, k, group, stream); break;
        case 3: launch_mma<BF16, 64, 64>(x, w, s, out, m, n, k, group, stream); break;
        case 4: launch_mma<BF16, 128, 256>(x, w, s, out, m, n, k, group, stream); break;
        case 5: launch_mma<BF16, 32, 128>(x, w, s, out, m, n, k, group, stream); break;
        case 6: launch_mma<BF16, 256, 256>(x, w, s, out, m, n, k, group, stream); break;
        case 7: launch_mma<BF16, 16, 128>(x, w, s, out, m, n, k, group, stream); break;
        default: TORCH_CHECK(false, "unknown W4A16 MMA variant");
    }
}

// Tile/warp schedule. The kernel re-dequantises the weight tile once per CTA,
// so the row tile is chosen to amortise that over as many rows as possible
// while keeping enough CTAs/warps resident. Measured on RTX 5060 Ti (layered
// 27-layer pass, see benchmarks/int4_dense_tuning.py): 16 rows x 4 warps wins
// for m<=64 (decode and small mixed batches), 32 x 4 warps for 65..160, and
// 128 rows x 8 warps above that (prefill), where the dequantisation saving
// dominates the lower CTA count.
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
    } else if (m <= 64) {
        launch_mma<BF16, 16, 128>(x, w, s, out, m, n, k, group, stream);
    } else if (m <= 160) {
        launch_mma<BF16, 32, 128>(x, w, s, out, m, n, k, group, stream);
    } else {
        launch_mma<BF16, 128, 256>(x, w, s, out, m, n, k, group, stream);
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

template <bool BF16, int GROUP>
__device__ void fp4_gemv(const unsigned short* x, PackedMatrix matrix, unsigned short* out,
                         int rows, int n, int k, int tile) {
    switch (rows) {
        case 1: gemv_columns<BF16, 1, false, GROUP>(x, matrix, {}, out, n, k, tile, 1); break;
        case 2: gemv_columns<BF16, 2, false, GROUP>(x, matrix, {}, out, n, k, tile, 2); break;
        case 3: gemv_columns<BF16, 3, false, GROUP>(x, matrix, {}, out, n, k, tile, 3); break;
        case 4: gemv_columns<BF16, 4, false, GROUP>(x, matrix, {}, out, n, k, tile, 4); break;
    }
}

template <bool BF16, int GROUP>
__global__ void fp4_gemv_kernel(const unsigned short* x, PackedMatrix matrix, const float* scale,
                                unsigned short* out, int m, int n, int k) {
    if constexpr (GROUP == 16) matrix.tensor_scale = *scale;
    fp4_gemv<BF16, GROUP>(x, matrix, out, m, n, k, blockIdx.x);
}

template <bool BF16, int GROUP, int BLOCK_M, int THREADS>
__global__ void fp4_mma_kernel(const unsigned short* x, PackedMatrix matrix, const float* scale,
                               unsigned short* out, int m, int n, int k, int n_tiles) {
    __shared__ __align__(16) unsigned short tile[(block_n + BLOCK_M) * block_k];
    if constexpr (GROUP == 16) matrix.tensor_scale = *scale;
    mma_tile<BF16, BLOCK_M, THREADS, GROUP>(
        x, matrix.weights, matrix.scales, out, m, n, k, GROUP,
        (blockIdx.x / n_tiles) * BLOCK_M, (blockIdx.x % n_tiles) * block_n,
        tile, matrix.tensor_scale);
}

template <bool BF16, int GROUP>
__global__ void fp4_grouped_kernel(const unsigned short* x, const int64_t* table,
        const int* offsets, const int* counts, unsigned short* out, int n, int k) {
    constexpr int block_m = 16;
    __shared__ __align__(16) unsigned short tile[(block_n + block_m) * block_k];
    const int expert = blockIdx.x;
    const int rows = counts[expert];
    if (!rows) return;
    const int64_t offset = offsets[expert];
    const int experts = gridDim.x;
    PackedMatrix matrix{reinterpret_cast<const uint8_t*>(table[expert]),
                        reinterpret_cast<const unsigned short*>(table[experts + expert]), GROUP};
    if constexpr (GROUP == 16) {
        matrix.tensor_scale = *reinterpret_cast<const float*>(table[2 * experts + expert]);
    }
    if (rows <= 4) {
        if (blockIdx.y != 0) return;
        for (int col = blockIdx.z * 16; col < n; col += gridDim.z * 16) {
            fp4_gemv<BF16, GROUP>(x + offset * k, matrix, out + offset * n,
                                  rows, n, k, col / 16);
        }
        return;
    }
    for (int start = blockIdx.y * block_m; start < rows; start += gridDim.y * block_m) {
        for (int col = blockIdx.z * block_n; col < n; col += gridDim.z * block_n) {
            mma_tile<BF16, block_m, 128, GROUP>(
                x + offset * k, matrix.weights, matrix.scales, out + offset * n,
                rows, n, k, GROUP, start, col, tile, matrix.tensor_scale);
        }
    }
}

template <bool BF16, int GROUP>
void launch_fp4(const unsigned short* x, PackedMatrix matrix, const float* scale,
                unsigned short* out, int m, int n, int k, cudaStream_t stream) {
    if (m <= 4) {
        fp4_gemv_kernel<BF16, GROUP><<<(n - 1) / 16 + 1, 128, 0, stream>>>(
            x, matrix, scale, out, m, n, k);
        return;
    }
    const int tiles = (n - 1) / block_n + 1;
    const int64_t grid = int64_t(tiles) * ((m - 1) / (m <= 64 ? 16 : 32) + 1);
    TORCH_CHECK(grid <= std::numeric_limits<int>::max(), "FP4 W4A16 grid is too large");
    if (m <= 64) {
        fp4_mma_kernel<BF16, GROUP, 16, 128><<<grid, 128, 0, stream>>>(
            x, matrix, scale, out, m, n, k, tiles);
    } else {
        fp4_mma_kernel<BF16, GROUP, 32, 128><<<grid, 128, 0, stream>>>(
            x, matrix, scale, out, m, n, k, tiles);
    }
}

void check_fp4_io(const at::Tensor& x, const at::Tensor& out, int64_t group) {
    TORCH_CHECK(group == 16 || group == 32, "FP4 W4A16 group must be 16 or 32");
    TORCH_CHECK(x.is_cuda() && out.device() == x.device() &&
                x.dim() == 2 && out.dim() == 2 && x.is_contiguous() && out.is_contiguous() &&
                (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
                out.scalar_type() == x.scalar_type() && x.size(0) == out.size(0) &&
                x.size(1) > 0 && x.size(1) % group == 0 &&
                std::max({x.size(0), x.size(1), out.size(1)}) <=
                    std::numeric_limits<int>::max() - block_k &&
                reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "invalid FP4 W4A16 I/O");
}

void fp4_gemm_out(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& scales,
                  const at::Tensor& scale, at::Tensor& out, int64_t group) {
    check_fp4_io(x, out, group);
    const int m = x.size(0), n = out.size(1), k = x.size(1);
    TORCH_CHECK(packed.device() == x.device() && scales.device() == x.device() &&
                packed.scalar_type() == at::kByte &&
                scales.scalar_type() == (group == 16 ? at::kFloat8_e4m3fn : at::kByte) &&
                packed.is_contiguous() && scales.is_contiguous() &&
                packed.sizes() == at::IntArrayRef({n, k / 2}) &&
                scales.sizes() == at::IntArrayRef({n, k / group}) &&
                reinterpret_cast<uintptr_t>(packed.data_ptr()) % 4 == 0,
                "invalid packed FP4 weight or block scales");
    if (group == 16) {
        TORCH_CHECK(scale.device() == x.device() && scale.scalar_type() == at::kFloat &&
                    scale.numel() == 1, "NVFP4 requires a FP32 tensor scale");
    }
    if (!m || !n) return;
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    const auto* a = reinterpret_cast<const unsigned short*>(x.data_ptr());
    auto* c = reinterpret_cast<unsigned short*>(out.data_ptr());
    const PackedMatrix matrix{packed.data_ptr<uint8_t>(),
                              reinterpret_cast<const unsigned short*>(scales.data_ptr()), int(group)};
    const auto* global = group == 16 ? scale.data_ptr<float>() : nullptr;
    if (x.scalar_type() == at::kBFloat16) {
        if (group == 16) launch_fp4<true, 16>(a, matrix, global, c, m, n, k, stream);
        else launch_fp4<true, 32>(a, matrix, global, c, m, n, k, stream);
    } else {
        if (group == 16) launch_fp4<false, 16>(a, matrix, global, c, m, n, k, stream);
        else launch_fp4<false, 32>(a, matrix, global, c, m, n, k, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool BF16, int GROUP>
void launch_fp4_grouped(const at::Tensor& x, const at::Tensor& table, const at::Tensor& offsets,
                        const at::Tensor& counts, at::Tensor& out, int max_rows, cudaStream_t stream) {
    const int n = out.size(1);
    const int fanout = max_rows <= 4 ? 8 : max_rows <= 64 ? 32 : 16;
    const dim3 grid(counts.numel(), std::min(4, (max_rows - 1) / 16 + 1),
                    std::min(fanout, (n - 1) / 16 + 1));
    fp4_grouped_kernel<BF16, GROUP><<<grid, 128, 0, stream>>>(
        reinterpret_cast<const unsigned short*>(x.data_ptr()), table.data_ptr<int64_t>(),
        offsets.data_ptr<int>(), counts.data_ptr<int>(),
        reinterpret_cast<unsigned short*>(out.data_ptr()), n, x.size(1));
}

// The pointer table and segment descriptors are produced by the MoE dispatcher.
void fp4_grouped_out(const at::Tensor& x, const at::Tensor& table, const at::Tensor& offsets,
                     const at::Tensor& counts, at::Tensor& out, int64_t group, int64_t max_rows) {
    check_fp4_io(x, out, group);
    for (const auto* t : {&table, &offsets, &counts}) {
        TORCH_CHECK(t->device() == x.device() && t->is_contiguous(), "invalid FP4 routing device");
    }
    TORCH_CHECK(counts.dim() == 1 && offsets.sizes() == counts.sizes() &&
                offsets.scalar_type() == at::kInt && counts.scalar_type() == at::kInt &&
                table.scalar_type() == at::kLong &&
                table.sizes() == at::IntArrayRef({group == 16 ? 3 : 2, counts.numel()}) &&
                max_rows > 0 && max_rows <= x.size(0), "invalid FP4 routing shapes");
    if (!counts.numel() || !out.size(1)) return;
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    if (x.scalar_type() == at::kBFloat16) {
        if (group == 16) launch_fp4_grouped<true, 16>(x, table, offsets, counts, out, max_rows, stream);
        else launch_fp4_grouped<true, 32>(x, table, offsets, counts, out, max_rows, stream);
    } else {
        if (group == 16) launch_fp4_grouped<false, 16>(x, table, offsets, counts, out, max_rows, stream);
        else launch_fp4_grouped<false, 32>(x, table, offsets, counts, out, max_rows, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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

// Diagnostic entry point: same layouts as gemm_out but a fixed MMA tile/warp
// variant, so benchmarks can screen tile shapes without rebuilding the engine.
void gemm_out_variant(const at::Tensor& x, const at::Tensor& packed,
                      const at::Tensor& scales, at::Tensor& out, int64_t variant) {
    const PackedMatrix matrix = checked_matrix(x, packed, scales, out);
    const int m = x.size(0), k = x.size(1), n = packed.size(0);
    if (m == 0 || n == 0) return;
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    const auto* a = reinterpret_cast<const unsigned short*>(x.data_ptr());
    auto* c = reinterpret_cast<unsigned short*>(out.data_ptr());
    if (x.scalar_type() == at::kBFloat16)
        launch_mma_variant<true>(static_cast<int>(variant), a, matrix.weights,
                                 matrix.scales, c, m, n, k, matrix.group_size, stream);
    else
        launch_mma_variant<false>(static_cast<int>(variant), a, matrix.weights,
                                  matrix.scales, c, m, n, k, matrix.group_size, stream);
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
                    int intermediate, int group_size, int cta_fanout,
                    int gate_fanout, cudaStream_t stream) {
    const int gate_tiles = (intermediate - 1) / 16 + 1;
    const int down_tiles = (width - 1) / 16 + 1;
    const int chunks = (max_rows + ROWS - 1) / ROWS;
    // Gate/up reuses its activation across four lane groups and prefers the
    // narrower grid; down is the larger K/column sweep and benefits from the
    // full fan-out. Empty experts still exit at the count guard.
    const int gate_ctas = std::min(gate_tiles, gate_fanout);
    const int down_ctas = std::min(down_tiles, cta_fanout);
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
                         int intermediate, int group_size, int cta_fanout,
                         int gate_fanout, cudaStream_t stream) {
    switch (rows_cap) {
        case 1: launch_compact<BF16, 1>(x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs,
                                        down_w_ptrs, down_s_ptrs, offsets, counts, hidden,
                                        out, experts, max_rows, width, intermediate,
                                        group_size, cta_fanout, gate_fanout, stream); break;
        case 2: launch_compact<BF16, 2>(x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs,
                                        down_w_ptrs, down_s_ptrs, offsets, counts, hidden,
                                        out, experts, max_rows, width, intermediate,
                                        group_size, cta_fanout, gate_fanout, stream); break;
        case 4: launch_compact<BF16, 4>(x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs,
                                        down_w_ptrs, down_s_ptrs, offsets, counts, hidden,
                                        out, experts, max_rows, width, intermediate,
                                        group_size, cta_fanout, gate_fanout, stream); break;
        case 8: launch_compact<BF16, 8>(x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs,
                                        down_w_ptrs, down_s_ptrs, offsets, counts, hidden,
                                        out, experts, max_rows, width, intermediate,
                                        group_size, cta_fanout, gate_fanout, stream); break;
    }
}

template <bool BF16, int BLOCK_M>
void launch_compact_mma_block(const unsigned short* x, const int64_t* weight_ptrs,
                              const int64_t* scale_ptrs, const int* offsets,
                              const int* counts, unsigned short* out, int experts,
                              int n, int k, int group_size, int cta_fanout,
                              int min_count, int max_count, cudaStream_t stream) {
    // Cap the grid's row-chunk dimension: each CTA walks the remaining
    // chunks with a stride. Prefill experts can need 50+ chunks; launching one
    // CTA per chunk makes the majority of them immediately exit.
    const int chunks = std::min((max_count + BLOCK_M - 1) / BLOCK_M, 4);
    const int n_tiles = (n + block_n - 1) / block_n;
    const int ctas = std::min(n_tiles, cta_fanout);
    const dim3 grid(experts, chunks, ctas);
    w4a16_compact_mma_kernel<BF16, BLOCK_M><<<grid, 64, 0, stream>>>(
        x, weight_ptrs, scale_ptrs, offsets, counts, out, n, k, group_size, n_tiles,
        min_count, max_count);
}

template <bool BF16>
void launch_compact_mma(const unsigned short* x, const int64_t* weight_ptrs,
                        const int64_t* scale_ptrs, const int* offsets, const int* counts,
                        unsigned short* out, int experts, int block_m, int n, int k,
                        int group_size, int cta_fanout, int min_count, int max_count,
                        cudaStream_t stream) {
    switch (block_m) {
        case 8: launch_compact_mma_block<BF16, 8>(
            x, weight_ptrs, scale_ptrs, offsets, counts, out, experts, n, k,
            group_size, cta_fanout, min_count, max_count, stream); break;
        case 16: launch_compact_mma_block<BF16, 16>(
            x, weight_ptrs, scale_ptrs, offsets, counts, out, experts, n, k,
            group_size, cta_fanout, min_count, max_count, stream); break;
        case 32: launch_compact_mma_block<BF16, 32>(
            x, weight_ptrs, scale_ptrs, offsets, counts, out, experts, n, k,
            group_size, cta_fanout, min_count, max_count, stream); break;
    }
}

template <bool BF16, int BLOCK_M>
void launch_compact_mma_gate_up_block(
        const unsigned short* x, const int64_t* gate_w_ptrs,
        const int64_t* gate_s_ptrs, const int64_t* up_w_ptrs,
        const int64_t* up_s_ptrs, const int* offsets, const int* counts,
        unsigned short* gate_out, unsigned short* up_out, int experts, int n,
        int k, int group_size, int cta_fanout, int min_count, int max_count,
        cudaStream_t stream) {
    const int chunks = std::min((max_count + BLOCK_M - 1) / BLOCK_M, 4);
    const int n_tiles = (n + block_n - 1) / block_n;
    const int ctas = std::min(n_tiles, cta_fanout);
    const dim3 grid(experts, chunks, ctas);
    w4a16_compact_mma_gate_up_kernel<BF16, BLOCK_M><<<grid, 128, 0, stream>>>(
        x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs, offsets, counts,
        gate_out, up_out, n, k, group_size, n_tiles, min_count, max_count);
}

template <bool BF16>
void launch_compact_mma_gate_up(
        const unsigned short* x, const int64_t* gate_w_ptrs,
        const int64_t* gate_s_ptrs, const int64_t* up_w_ptrs,
        const int64_t* up_s_ptrs, const int* offsets, const int* counts,
        unsigned short* gate_out, unsigned short* up_out, int experts,
        int block_m, int n, int k, int group_size, int cta_fanout,
        int min_count, int max_count, cudaStream_t stream) {
    switch (block_m) {
        case 8: launch_compact_mma_gate_up_block<BF16, 8>(
            x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs, offsets, counts,
            gate_out, up_out, experts, n, k, group_size, cta_fanout, min_count,
            max_count, stream); break;
        case 16: launch_compact_mma_gate_up_block<BF16, 16>(
            x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs, offsets, counts,
            gate_out, up_out, experts, n, k, group_size, cta_fanout, min_count,
            max_count, stream); break;
        case 32: launch_compact_mma_gate_up_block<BF16, 32>(
            x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs, offsets, counts,
            gate_out, up_out, experts, n, k, group_size, cta_fanout, min_count,
            max_count, stream); break;
    }
}

template <bool BF16, int BLOCK_M>
void launch_compact_mma_gate_up_silu_block(
        const unsigned short* x, const int64_t* gate_w_ptrs,
        const int64_t* gate_s_ptrs, const int64_t* up_w_ptrs,
        const int64_t* up_s_ptrs, const int* offsets, const int* counts,
        unsigned short* hidden, int experts, int n, int k, int group_size,
        int cta_fanout, int min_count, int max_count, cudaStream_t stream) {
    // Four row chunks per expert/channel tile is the measured optimum: one/two
    // chunks serialized too much work per CTA and lost more occupancy than the
    // repeated weight/activation reads saved. Kept as the baseline schedule.
    const int chunks = std::min((max_count + BLOCK_M - 1) / BLOCK_M, 4);
    const int n_tiles = (n + block_n - 1) / block_n;
    const int ctas = std::min(n_tiles, cta_fanout);
    const dim3 grid(experts, chunks, ctas);
    w4a16_compact_mma_gate_up_silu_kernel<BF16, BLOCK_M>
        <<<grid, 128, 0, stream>>>(
            x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs, offsets, counts,
            hidden, n, k, group_size, n_tiles, min_count, max_count);
}

template <bool BF16>
void launch_compact_mma_gate_up_silu(
        const unsigned short* x, const int64_t* gate_w_ptrs,
        const int64_t* gate_s_ptrs, const int64_t* up_w_ptrs,
        const int64_t* up_s_ptrs, const int* offsets, const int* counts,
        unsigned short* hidden, int experts, int block_m, int n, int k,
        int group_size, int cta_fanout, int min_count, int max_count,
        cudaStream_t stream) {
    switch (block_m) {
        case 8: launch_compact_mma_gate_up_silu_block<BF16, 8>(
            x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs, offsets, counts,
            hidden, experts, n, k, group_size, cta_fanout, min_count,
            max_count, stream); break;
        case 16: launch_compact_mma_gate_up_silu_block<BF16, 16>(
            x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs, offsets, counts,
            hidden, experts, n, k, group_size, cta_fanout, min_count,
            max_count, stream); break;
        case 32: launch_compact_mma_gate_up_silu_block<BF16, 32>(
            x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs, offsets, counts,
            hidden, experts, n, k, group_size, cta_fanout, min_count,
            max_count, stream); break;
    }
}

template <bool BF16>
void launch_compact_mma_band(const unsigned short* x, const int64_t* weight_ptrs,
                             const int64_t* scale_ptrs, const int* offsets,
                             const int* counts, unsigned short* out, int experts,
                             int n, int k, int group_size, int cta_fanout,
                             int min_count, int max_count, cudaStream_t stream) {
    const int n_tiles = (n + block_n - 1) / block_n;
    const int ctas = std::min(n_tiles, cta_fanout);
    const dim3 grid(experts, 1, ctas);
    w4a16_compact_mma_band_kernel<BF16><<<grid, 64, 0, stream>>>(
        x, weight_ptrs, scale_ptrs, offsets, counts, out, n, k, group_size,
        n_tiles, min_count, max_count);
}

template <bool BF16>
void launch_compact_mma_gate_up_band(
        const unsigned short* x, const int64_t* gate_w_ptrs,
        const int64_t* gate_s_ptrs, const int64_t* up_w_ptrs,
        const int64_t* up_s_ptrs, const int* offsets, const int* counts,
        unsigned short* gate_out, unsigned short* up_out, int experts, int n,
        int k, int group_size, int cta_fanout, int min_count, int max_count,
        cudaStream_t stream) {
    const int n_tiles = (n + block_n - 1) / block_n;
    const int ctas = std::min(n_tiles, cta_fanout);
    const dim3 grid(experts, 1, ctas);
    w4a16_compact_mma_gate_up_band_kernel<BF16, false><<<grid, 128, 0, stream>>>(
        x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs, offsets, counts,
        gate_out, up_out, n, k, group_size, n_tiles, min_count, max_count);
}

template <bool BF16>
void launch_compact_mma_gate_up_silu_band(
        const unsigned short* x, const int64_t* gate_w_ptrs,
        const int64_t* gate_s_ptrs, const int64_t* up_w_ptrs,
        const int64_t* up_s_ptrs, const int* offsets, const int* counts,
        unsigned short* hidden, int experts, int n, int k, int group_size,
        int cta_fanout, int min_count, int max_count, cudaStream_t stream) {
    const int n_tiles = (n + block_n - 1) / block_n;
    const int ctas = std::min(n_tiles, cta_fanout);
    const dim3 grid(experts, 1, ctas);
    w4a16_compact_mma_gate_up_band_kernel<BF16, true><<<grid, 128, 0, stream>>>(
        x, gate_w_ptrs, gate_s_ptrs, up_w_ptrs, up_s_ptrs, offsets, counts,
        hidden, nullptr, n, k, group_size, n_tiles, min_count, max_count);
}

bool gemm_out_compact(const at::Tensor& x, const at::Tensor& weight_ptrs,
                      const at::Tensor& scale_ptrs, const at::Tensor& offsets,
                      const at::Tensor& counts, at::Tensor& out,
                      int64_t max_rows, int64_t logical_rows, int64_t group_size,
                      int64_t cta_fanout, int64_t block_m, int64_t min_count,
                      int64_t max_count) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() &&
                out.is_cuda() && out.dim() == 2 && out.is_contiguous(),
                "compact W4A16 MMA expects contiguous rank-two CUDA tensors");
    const auto type = x.scalar_type();
    TORCH_CHECK((type == at::kHalf || type == at::kBFloat16) &&
                out.scalar_type() == type, "invalid compact W4A16 MMA dtype");
    TORCH_CHECK(weight_ptrs.is_cuda() && scale_ptrs.is_cuda() &&
                weight_ptrs.scalar_type() == at::kLong &&
                scale_ptrs.scalar_type() == at::kLong &&
                weight_ptrs.dim() == 1 && scale_ptrs.dim() == 1 &&
                weight_ptrs.numel() == counts.numel() &&
                scale_ptrs.numel() == counts.numel() &&
                weight_ptrs.is_contiguous() && scale_ptrs.is_contiguous(),
                "compact W4A16 MMA expects int64 pointer tables");
    TORCH_CHECK(offsets.is_cuda() && counts.is_cuda() &&
                offsets.scalar_type() == at::kInt && counts.scalar_type() == at::kInt &&
                offsets.dim() == 1 && counts.dim() == 1 &&
                offsets.numel() == counts.numel() && offsets.is_contiguous() &&
                counts.is_contiguous(), "invalid compact W4A16 MMA offsets/counts");
    const int experts = counts.numel();
    if (experts == 0) return true;
    const int rows = static_cast<int>(logical_rows);
    const int k = x.size(1);
    const int n = out.size(1);
    const int block = static_cast<int>(block_m);
    const int padding = block == 0 ? std::min(16, static_cast<int>(max_rows)) : block;
    const int lo = static_cast<int>(min_count);
    const int hi = static_cast<int>(max_count);
    TORCH_CHECK(max_rows > 0 && max_rows <= 16384 &&
                (block == 0 || block == 8 || block == 16 || block == 32) &&
                lo >= 1 && lo <= hi && hi <= max_rows &&
                rows >= 0 && rows <= x.size(0) &&
                out.size(0) >= rows + padding && x.size(0) >= rows + padding,
                "invalid compact W4A16 MMA row range");
    const int group = static_cast<int>(group_size);
    TORCH_CHECK(group >= 8 && 128 % group == 0 && k % group == 0,
                "invalid compact W4A16 MMA group size");
    TORCH_CHECK(cta_fanout >= 1, "compact W4A16 MMA CTA fan-out must be positive");
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    const auto* x_ptr = reinterpret_cast<const unsigned short*>(x.data_ptr());
    auto* out_ptr = reinterpret_cast<unsigned short*>(out.data_ptr());
    const auto* weight = weight_ptrs.data_ptr<int64_t>();
    const auto* scale = scale_ptrs.data_ptr<int64_t>();
    const auto* offsets_ptr = offsets.data_ptr<int>();
    const auto* counts_ptr = counts.data_ptr<int>();
    if (block == 0) {
        if (type == at::kBFloat16)
            launch_compact_mma_band<true>(x_ptr, weight, scale, offsets_ptr, counts_ptr,
                                          out_ptr, experts, n, k, group,
                                          static_cast<int>(cta_fanout), lo, hi, stream);
        else
            launch_compact_mma_band<false>(x_ptr, weight, scale, offsets_ptr, counts_ptr,
                                           out_ptr, experts, n, k, group,
                                           static_cast<int>(cta_fanout), lo, hi, stream);
    } else if (type == at::kBFloat16)
        launch_compact_mma<true>(x_ptr, weight, scale, offsets_ptr, counts_ptr, out_ptr,
                                 experts, block, n, k, group,
                                 static_cast<int>(cta_fanout), lo, hi, stream);
    else
        launch_compact_mma<false>(x_ptr, weight, scale, offsets_ptr, counts_ptr, out_ptr,
                                  experts, block, n, k, group,
                                  static_cast<int>(cta_fanout), lo, hi, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

bool gemm_gate_up_out_compact(
        const at::Tensor& x,
        const at::Tensor& gate_w_ptrs, const at::Tensor& gate_s_ptrs,
        const at::Tensor& up_w_ptrs, const at::Tensor& up_s_ptrs,
        const at::Tensor& offsets, const at::Tensor& counts,
        at::Tensor& gate_out, at::Tensor& up_out, int64_t max_rows,
        int64_t logical_rows, int64_t group_size, int64_t cta_fanout,
        int64_t block_m, int64_t min_count, int64_t max_count) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() &&
                gate_out.is_cuda() && gate_out.dim() == 2 &&
                gate_out.is_contiguous() && up_out.is_cuda() &&
                up_out.dim() == 2 && up_out.is_contiguous(),
                "compact W4A16 gate/up expects contiguous rank-two CUDA tensors");
    const auto type = x.scalar_type();
    TORCH_CHECK((type == at::kHalf || type == at::kBFloat16) &&
                gate_out.scalar_type() == type && up_out.scalar_type() == type,
                "invalid compact W4A16 gate/up dtype");
    TORCH_CHECK(gate_out.sizes() == up_out.sizes(),
                "compact W4A16 gate/up outputs must have the same shape");
    for (const auto* tensor : {&gate_w_ptrs, &gate_s_ptrs, &up_w_ptrs,
                               &up_s_ptrs}) {
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kLong &&
                    tensor->dim() == 1 && tensor->numel() == counts.numel() &&
                    tensor->is_contiguous(),
                    "compact W4A16 gate/up expects int64 pointer tables");
    }
    TORCH_CHECK(offsets.is_cuda() && counts.is_cuda() &&
                offsets.scalar_type() == at::kInt && counts.scalar_type() == at::kInt &&
                offsets.dim() == 1 && counts.dim() == 1 &&
                offsets.numel() == counts.numel() && offsets.is_contiguous() &&
                counts.is_contiguous(), "invalid compact W4A16 gate/up offsets/counts");
    const int experts = counts.numel();
    if (experts == 0) return true;
    const int rows = static_cast<int>(logical_rows);
    const int k = x.size(1);
    const int n = gate_out.size(1);
    const int block = static_cast<int>(block_m);
    const int padding = block == 0 ? std::min(16, static_cast<int>(max_rows)) : block;
    const int lo = static_cast<int>(min_count);
    const int hi = static_cast<int>(max_count);
    TORCH_CHECK(max_rows > 0 && max_rows <= 16384 &&
                (block == 0 || block == 8 || block == 16 || block == 32) &&
                lo >= 1 && lo <= hi && hi <= max_rows &&
                rows >= 0 && rows <= x.size(0) &&
                gate_out.size(0) >= rows + padding &&
                up_out.size(0) >= rows + padding && x.size(0) >= rows + padding,
                "invalid compact W4A16 gate/up row range");
    const int group = static_cast<int>(group_size);
    TORCH_CHECK(group >= 8 && 128 % group == 0 && k % group == 0,
                "invalid compact W4A16 gate/up group size");
    TORCH_CHECK(cta_fanout >= 1, "compact W4A16 gate/up CTA fan-out must be positive");
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    const auto* x_ptr = reinterpret_cast<const unsigned short*>(x.data_ptr());
    auto* gate_ptr = reinterpret_cast<unsigned short*>(gate_out.data_ptr());
    auto* up_ptr = reinterpret_cast<unsigned short*>(up_out.data_ptr());
    const auto* gate_w = gate_w_ptrs.data_ptr<int64_t>();
    const auto* gate_s = gate_s_ptrs.data_ptr<int64_t>();
    const auto* up_w = up_w_ptrs.data_ptr<int64_t>();
    const auto* up_s = up_s_ptrs.data_ptr<int64_t>();
    const auto* offsets_ptr = offsets.data_ptr<int>();
    const auto* counts_ptr = counts.data_ptr<int>();
    if (block == 0) {
        if (type == at::kBFloat16)
            launch_compact_mma_gate_up_band<true>(
                x_ptr, gate_w, gate_s, up_w, up_s, offsets_ptr, counts_ptr,
                gate_ptr, up_ptr, experts, n, k, group,
                static_cast<int>(cta_fanout), lo, hi, stream);
        else
            launch_compact_mma_gate_up_band<false>(
                x_ptr, gate_w, gate_s, up_w, up_s, offsets_ptr, counts_ptr,
                gate_ptr, up_ptr, experts, n, k, group,
                static_cast<int>(cta_fanout), lo, hi, stream);
    } else if (type == at::kBFloat16)
        launch_compact_mma_gate_up<true>(
            x_ptr, gate_w, gate_s, up_w, up_s, offsets_ptr, counts_ptr,
            gate_ptr, up_ptr, experts, block, n, k, group,
            static_cast<int>(cta_fanout), lo, hi, stream);
    else
        launch_compact_mma_gate_up<false>(
            x_ptr, gate_w, gate_s, up_w, up_s, offsets_ptr, counts_ptr,
            gate_ptr, up_ptr, experts, block, n, k, group,
            static_cast<int>(cta_fanout), lo, hi, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

// Fused single-output form of gemm_gate_up_out_compact: hidden receives
// silu(gate)*up, so no gate/up intermediates are materialized and no separate
// elementwise launch is needed. Validation mirrors the split entry point; the
// two must accept exactly the same geometry so the paths stay interchangeable.
bool gemm_gate_up_silu_compact(
        const at::Tensor& x,
        const at::Tensor& gate_w_ptrs, const at::Tensor& gate_s_ptrs,
        const at::Tensor& up_w_ptrs, const at::Tensor& up_s_ptrs,
        const at::Tensor& offsets, const at::Tensor& counts,
        at::Tensor& hidden, int64_t max_rows, int64_t logical_rows,
        int64_t group_size, int64_t cta_fanout, int64_t block_m,
        int64_t min_count, int64_t max_count) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() &&
                hidden.is_cuda() && hidden.dim() == 2 &&
                hidden.is_contiguous(),
                "compact W4A16 fused gate/up expects contiguous rank-two CUDA tensors");
    const auto type = x.scalar_type();
    TORCH_CHECK((type == at::kHalf || type == at::kBFloat16) &&
                hidden.scalar_type() == type,
                "invalid compact W4A16 fused gate/up dtype");
    for (const auto* tensor : {&gate_w_ptrs, &gate_s_ptrs, &up_w_ptrs,
                               &up_s_ptrs}) {
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kLong &&
                    tensor->dim() == 1 && tensor->numel() == counts.numel() &&
                    tensor->is_contiguous(),
                    "compact W4A16 fused gate/up expects int64 pointer tables");
    }
    TORCH_CHECK(offsets.is_cuda() && counts.is_cuda() &&
                offsets.scalar_type() == at::kInt && counts.scalar_type() == at::kInt &&
                offsets.dim() == 1 && counts.dim() == 1 &&
                offsets.numel() == counts.numel() && offsets.is_contiguous() &&
                counts.is_contiguous(),
                "invalid compact W4A16 fused gate/up offsets/counts");
    const int experts = counts.numel();
    if (experts == 0) return true;
    const int rows = static_cast<int>(logical_rows);
    const int k = x.size(1);
    const int n = hidden.size(1);
    const int block = static_cast<int>(block_m);
    const int padding = block == 0 ? std::min(16, static_cast<int>(max_rows)) : block;
    const int lo = static_cast<int>(min_count);
    const int hi = static_cast<int>(max_count);
    TORCH_CHECK(max_rows > 0 && max_rows <= 16384 &&
                (block == 0 || block == 8 || block == 16 || block == 32) &&
                lo >= 1 && lo <= hi && hi <= max_rows &&
                rows >= 0 && rows <= x.size(0) &&
                hidden.size(0) >= rows + padding && x.size(0) >= rows + padding,
                "invalid compact W4A16 fused gate/up row range");
    const int group = static_cast<int>(group_size);
    TORCH_CHECK(group >= 8 && 128 % group == 0 && k % group == 0,
                "invalid compact W4A16 fused gate/up group size");
    TORCH_CHECK(cta_fanout >= 1,
                "compact W4A16 fused gate/up CTA fan-out must be positive");
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    const auto* x_ptr = reinterpret_cast<const unsigned short*>(x.data_ptr());
    auto* hidden_ptr = reinterpret_cast<unsigned short*>(hidden.data_ptr());
    const auto* gate_w = gate_w_ptrs.data_ptr<int64_t>();
    const auto* gate_s = gate_s_ptrs.data_ptr<int64_t>();
    const auto* up_w = up_w_ptrs.data_ptr<int64_t>();
    const auto* up_s = up_s_ptrs.data_ptr<int64_t>();
    const auto* offsets_ptr = offsets.data_ptr<int>();
    const auto* counts_ptr = counts.data_ptr<int>();
    if (block == 0) {
        if (type == at::kBFloat16)
            launch_compact_mma_gate_up_silu_band<true>(
                x_ptr, gate_w, gate_s, up_w, up_s, offsets_ptr, counts_ptr,
                hidden_ptr, experts, n, k, group,
                static_cast<int>(cta_fanout), lo, hi, stream);
        else
            launch_compact_mma_gate_up_silu_band<false>(
                x_ptr, gate_w, gate_s, up_w, up_s, offsets_ptr, counts_ptr,
                hidden_ptr, experts, n, k, group,
                static_cast<int>(cta_fanout), lo, hi, stream);
    } else if (type == at::kBFloat16)
        launch_compact_mma_gate_up_silu<true>(
            x_ptr, gate_w, gate_s, up_w, up_s, offsets_ptr, counts_ptr,
            hidden_ptr, experts, block, n, k, group,
            static_cast<int>(cta_fanout), lo, hi, stream);
    else
        launch_compact_mma_gate_up_silu<false>(
            x_ptr, gate_w, gate_s, up_w, up_s, offsets_ptr, counts_ptr,
            hidden_ptr, experts, block, n, k, group,
            static_cast<int>(cta_fanout), lo, hi, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

bool expert_out_compact(const at::Tensor& x,
                        const at::Tensor& gate_w_ptrs, const at::Tensor& gate_s_ptrs,
                        const at::Tensor& up_w_ptrs, const at::Tensor& up_s_ptrs,
                        const at::Tensor& down_w_ptrs, const at::Tensor& down_s_ptrs,
                        const at::Tensor& offsets, const at::Tensor& counts,
                        at::Tensor& hidden, at::Tensor& out, int64_t max_rows,
                        int64_t logical_rows, int64_t group_size, int64_t cta_fanout,
                        int64_t gate_fanout) {
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
    TORCH_CHECK(cta_fanout >= 1, "compact W4A16 CTA fan-out must be positive");
    TORCH_CHECK(gate_fanout >= 1, "compact W4A16 gate fan-out must be positive");
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
                                  intermediate, group, static_cast<int>(cta_fanout),
                                  static_cast<int>(gate_fanout), stream);
    else
        launch_compact_rows<false>(rows_cap, x_ptr, gate_w, gate_s, up_w, up_s,
                                   down_w, down_s, offsets_ptr, counts_ptr, hidden_ptr,
                                   out_ptr, experts, static_cast<int>(max_rows), width,
                                   intermediate, group, static_cast<int>(cta_fanout),
                                   static_cast<int>(gate_fanout), stream);
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
    module.def("fp4_gemm_out", &fp4_gemm_out, "Packed NVFP4/MXFP4 W4A16 GEMM");
    module.def("fp4_grouped_out", &fp4_grouped_out, "Segmented NVFP4/MXFP4 expert GEMM");
    module.def("gemm_out", &gemm_out, "Fused packed-INT4 x FP16/BF16 GEMM");
    module.def("gemm_out_variant", &gemm_out_variant,
               "Diagnostic packed-INT4 GEMM with an explicit MMA tile variant");
    module.def("expert_out", &expert_out, "Fused packed-INT4 expert FFN for M1-4");
    module.def("expert_out_compact", &expert_out_compact,
               "Route-major packed-INT4 expert FFN with device offsets");
    module.def("gemm_out_compact", &gemm_out_compact,
               "Route-major packed-INT4 MMA projection with device offsets");
    module.def("gemm_gate_up_out_compact", &gemm_gate_up_out_compact,
               "Route-major fused packed-INT4 gate/up MMA with device offsets");
    module.def("gemm_gate_up_silu_compact", &gemm_gate_up_silu_compact,
               "Route-major packed-INT4 gate/up MMA with fused SiLU*up epilogue");
}
