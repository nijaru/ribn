#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

static inline float bf16_load(ushort value) {
    return as_type<float>((uint)value << 16);
}

static inline float4 bf16_load4(ushort4 value) {
    return as_type<float4>(uint4(value) << 16u);
}

// NaN keeps a canonical encoding rather than rounding its payload to infinity.
static inline ushort bf16_rne(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
    return (ushort)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static inline uint fast_rne_shift(uint value, uint shift) {
    if (shift == 0) return value;
    if (shift >= 32) return 0;
    uint truncated = value >> shift;
    uint remainder = value & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    return truncated + uint(remainder > halfway || (remainder == halfway && (truncated & 1u)));
}

static inline ushort fast_bf16_product_rne(ushort a, ushort b) {
    uint sign = (a ^ b) & 0x8000u;
    uint ea = (a >> 7) & 255u, eb = (b >> 7) & 255u;
    uint ma = a & 127u, mb = b & 127u;
    if (ea == 255u || eb == 255u) {
        if ((ea == 255u && ma != 0) || (eb == 255u && mb != 0)
            || ((a & 0x7fffu) == 0 && eb == 255u)
            || ((b & 0x7fffu) == 0 && ea == 255u)) return 0x7fc0u;
        return ushort(sign | 0x7f80u);
    }
    uint product = (ea == 0 ? ma : ma | 128u) * (eb == 0 ? mb : mb | 128u);
    if (product == 0) return ushort(sign);
    int top = 31 - int(clz(product));
    int power = int(max(ea, 1u)) + int(max(eb, 1u)) - 268;
    int exponent = top + power + 127;
    if (exponent <= 0) return ushort(sign | fast_rne_shift(product, uint(-power - 133)));
    uint rounded = fast_rne_shift(product, uint(top - 7));
    if (rounded == 256u) { rounded = 128u; ++exponent; }
    if (exponent >= 255) return ushort(sign | 0x7f80u);
    return ushort(sign | (uint(exponent) << 7) | (rounded - 128u));
}

// ============================================================================
// 1. Coalesced SIMD-group GEMV for M = 1 (2 columns per SIMD group, 4 SGs/TG)
//    Native bfloat4 + 2x K-unroll; low register count for maximum memory-level
//    parallelism across all 40 M3 Max GPU cores.
// ============================================================================
kernel void fast_bf16_gemv_m1(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n0 = (tg_id * 4u + sg) * 2u;
    if (n0 >= N) return;

    if ((K & 3u) == 0u) {
        uint k4 = K >> 2;
        device const bfloat4* x4 = (device const bfloat4*)input;
        if (n0 + 1u < N) {
            device const bfloat4* w0 = (device const bfloat4*)(weight + (ulong)(n0 + 0u) * K);
            device const bfloat4* w1 = (device const bfloat4*)(weight + (ulong)(n0 + 1u) * K);
            float acc0 = 0.0f, acc1 = 0.0f;
            uint i = lane;
            for (; i + 32u < k4; i += 64u) {
                float4 xa = float4(x4[i]);
                float4 xb = float4(x4[i + 32u]);
                acc0 += dot(xa, float4(w0[i])) + dot(xb, float4(w0[i + 32u]));
                acc1 += dot(xa, float4(w1[i])) + dot(xb, float4(w1[i + 32u]));
            }
            if (i < k4) {
                float4 xa = float4(x4[i]);
                acc0 += dot(xa, float4(w0[i]));
                acc1 += dot(xa, float4(w1[i]));
            }
            acc0 = simd_sum(acc0);
            acc1 = simd_sum(acc1);
            if (lane == 0u) {
                output[n0 + 0u] = bf16_rne(acc0);
                output[n0 + 1u] = bf16_rne(acc1);
            }
        } else {
            device const bfloat4* w0 = (device const bfloat4*)(weight + (ulong)n0 * K);
            float acc0 = 0.0f;
            for (uint i = lane; i < k4; i += 32u) {
                acc0 += dot(float4(x4[i]), float4(w0[i]));
            }
            acc0 = simd_sum(acc0);
            if (lane == 0u) output[n0] = bf16_rne(acc0);
        }
    } else {
        for (uint r = 0u; r < 2u && (n0 + r) < N; ++r) {
            device const bfloat* wr = weight + (ulong)(n0 + r) * K;
            float acc = 0.0f;
            for (uint i = lane; i < K; i += 32u) {
                acc += float(input[i]) * float(wr[i]);
            }
            acc = simd_sum(acc);
            if (lane == 0u) output[n0 + r] = bf16_rne(acc);
        }
    }
}

// ============================================================================
// 2. Weight-reusing SIMD-group batched GEMV for M = 2..16
//    2D grid (gx = ceil(M / BM_MAX), gy = ceil(N / 4)) dispatches M-tiles for
//    the SAME weight columns concurrently so they share weights in L2/SLC.
// ============================================================================
template<uint BM_MAX>
static inline void gemv_batch_impl(
    device const bfloat* input,
    device const bfloat* weight,
    device ushort* output,
    uint M,
    uint N,
    uint K,
    uint2 tg_pos,
    uint tid)
{
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint m0 = tg_pos.x * BM_MAX;
    uint n0 = (tg_pos.y * 2u + sg) * 2u;
    if (m0 >= M || n0 >= N) return;

    uint m_rem = min(BM_MAX, M - m0);
    float acc0[BM_MAX];
    float acc1[BM_MAX];
    for (uint m = 0u; m < BM_MAX; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    bool has_second = (n0 + 1u) < N;
    if ((K & 3u) == 0u) {
        uint k4 = K >> 2;
        device const bfloat4* w0 = (device const bfloat4*)(weight + (ulong)n0 * K);
        device const bfloat4* w1 = has_second
            ? (device const bfloat4*)(weight + (ulong)(n0 + 1u) * K)
            : w0;
        device const bfloat4* x_ptrs[BM_MAX];
        for (uint m = 0u; m < BM_MAX; ++m) {
            uint row = m0 + min(m, m_rem - 1u);
            x_ptrs[m] = (device const bfloat4*)(input + (ulong)row * K);
        }
        for (uint i = lane; i < k4; i += 32u) {
            float4 b0 = float4(w0[i]);
            float4 b1 = has_second ? float4(w1[i]) : float4(0.0f);
            for (uint m = 0u; m < BM_MAX; ++m) {
                float4 xv = float4(x_ptrs[m][i]);
                acc0[m] += dot(xv, b0);
                if (has_second) acc1[m] += dot(xv, b1);
            }
        }
    } else {
        device const bfloat* w0 = weight + (ulong)n0 * K;
        device const bfloat* w1 = has_second ? (weight + (ulong)(n0 + 1u) * K) : w0;
        for (uint i = lane; i < K; i += 32u) {
            float b0 = float(w0[i]);
            float b1 = has_second ? float(w1[i]) : 0.0f;
            for (uint m = 0u; m < BM_MAX; ++m) {
                uint row = m0 + min(m, m_rem - 1u);
                float xv = float(input[(ulong)row * K + i]);
                acc0[m] += xv * b0;
                if (has_second) acc1[m] += xv * b1;
            }
        }
    }

    for (uint m = 0u; m < BM_MAX; ++m) {
        float s0 = simd_sum(acc0[m]);
        float s1 = has_second ? simd_sum(acc1[m]) : 0.0f;
        if (lane == 0u && m < m_rem) {
            output[(ulong)(m0 + m) * N + n0] = bf16_rne(s0);
            if (has_second) output[(ulong)(m0 + m) * N + n0 + 1u] = bf16_rne(s1);
        }
    }
}

kernel void fast_bf16_gemv_m2(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    gemv_batch_impl<2>(input, weight, output, M, N, K, tg_pos, tid);
}

kernel void fast_bf16_gemv_m4(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    gemv_batch_impl<4>(input, weight, output, M, N, K, tg_pos, tid);
}

kernel void fast_bf16_gemv_m8(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    gemv_batch_impl<8>(input, weight, output, M, N, K, tg_pos, tid);
}

// ============================================================================
// 3. Mixed-Precision BF16->FP32 Tiled GEMM (BM=32, BN=32, BK=32, 4 SGs)
//    Used for M > 16. Stores BF16 tiles in 5 KiB threadgroup memory (8-byte
//    aligned) and accumulates via hardware simdgroup_bfloat8x8 -> float8x8 MMA.
//    Clamped in-bounds row pointers keep the inner K loop branch-free even for
//    non-multiple-of-32 M/N edge tiles (e.g. M=18, M=511).
// ============================================================================
kernel void fast_bf16_gemm_m32(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint BM = 32u;
    constexpr uint BN = 32u;
    constexpr uint BK = 32u;
    constexpr uint STRIDE = 40u;

    uint m0 = tg_pos.x * BM;
    uint n0 = tg_pos.y * BN;
    if (m0 >= M || n0 >= N) return;

    threadgroup bfloat4 smem_v[640]; // (32 + 32) * 40 bfloats = 5120 bytes (1280 floats)
    threadgroup bfloat* As = (threadgroup bfloat*)smem_v;
    threadgroup bfloat* Bs = As + BM * STRIDE;

    uint sg = tid >> 5;
    uint sg_m = sg >> 1;
    uint sg_n = sg & 1u;

    simdgroup_float8x8 acc[2][2];
    for (uint i = 0u; i < 2u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint ar = tid >> 3;        // 0..15
    uint ac = (tid & 7u) << 2; // 0, 4, ..., 28
    bool aligned_k = (K & 3u) == 0u;

    // Clamped in-bounds row offsets: out-of-bounds tile rows never affect valid
    // outputs (gr < M && gc < N), keeping the main K loop 100% branch-free.
    ulong a_row0 = (ulong)min(m0 + ar, M - 1u) * K + ac;
    ulong a_row1 = (ulong)min(m0 + 16u + ar, M - 1u) * K + ac;
    ulong b_row0 = (ulong)min(n0 + ar, N - 1u) * K + ac;
    ulong b_row1 = (ulong)min(n0 + 16u + ar, N - 1u) * K + ac;

    uint k_fast = aligned_k ? (K & ~31u) : 0u;
    for (uint k0 = 0u; k0 < k_fast; k0 += BK) {
        *(threadgroup bfloat4*)(As + ar * STRIDE + ac) =
            *(device const bfloat4*)(input + a_row0 + k0);
        *(threadgroup bfloat4*)(As + (16u + ar) * STRIDE + ac) =
            *(device const bfloat4*)(input + a_row1 + k0);
        *(threadgroup bfloat4*)(Bs + ar * STRIDE + ac) =
            *(device const bfloat4*)(weight + b_row0 + k0);
        *(threadgroup bfloat4*)(Bs + (16u + ar) * STRIDE + ac) =
            *(device const bfloat4*)(weight + b_row1 + k0);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_bfloat8x8 a_frag[2];
            simdgroup_bfloat8x8 b_frag[2];
            simdgroup_load(a_frag[0], As + (sg_m * 16u + 0u) * STRIDE + kk, STRIDE, ulong2(0, 0), false);
            simdgroup_load(a_frag[1], As + (sg_m * 16u + 8u) * STRIDE + kk, STRIDE, ulong2(0, 0), false);
            simdgroup_load(b_frag[0], Bs + (sg_n * 16u + 0u) * STRIDE + kk, STRIDE, ulong2(0, 0), true);
            simdgroup_load(b_frag[1], Bs + (sg_n * 16u + 8u) * STRIDE + kk, STRIDE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a_frag[0], b_frag[0], acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a_frag[0], b_frag[1], acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a_frag[1], b_frag[0], acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a_frag[1], b_frag[1], acc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint k0 = k_fast; k0 < K; k0 += BK) {
        bool full_k = (k0 + BK <= K) && aligned_k;
        for (uint pass = 0u; pass < 2u; ++pass) {
            uint lr = pass * 16u + ar;
            uint gr_a = m0 + lr;
            uint dst_a = lr * STRIDE + ac;
            if (gr_a < M && full_k) {
                *(threadgroup bfloat4*)(As + dst_a) =
                    *(device const bfloat4*)(input + (ulong)gr_a * K + k0 + ac);
            } else {
                for (uint c = 0u; c < 4u; ++c) {
                    uint gc = k0 + ac + c;
                    As[dst_a + c] = (gr_a < M && gc < K) ? input[(ulong)gr_a * K + gc] : bfloat(0.0f);
                }
            }

            uint gr_b = n0 + lr;
            uint dst_b = lr * STRIDE + ac;
            if (gr_b < N && full_k) {
                *(threadgroup bfloat4*)(Bs + dst_b) =
                    *(device const bfloat4*)(weight + (ulong)gr_b * K + k0 + ac);
            } else {
                for (uint c = 0u; c < 4u; ++c) {
                    uint gc = k0 + ac + c;
                    Bs[dst_b + c] = (gr_b < N && gc < K) ? weight[(ulong)gr_b * K + gc] : bfloat(0.0f);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_bfloat8x8 a_frag[2];
            simdgroup_bfloat8x8 b_frag[2];
            simdgroup_load(a_frag[0], As + (sg_m * 16u + 0u) * STRIDE + kk, STRIDE, ulong2(0, 0), false);
            simdgroup_load(a_frag[1], As + (sg_m * 16u + 8u) * STRIDE + kk, STRIDE, ulong2(0, 0), false);
            simdgroup_load(b_frag[0], Bs + (sg_n * 16u + 0u) * STRIDE + kk, STRIDE, ulong2(0, 0), true);
            simdgroup_load(b_frag[1], Bs + (sg_n * 16u + 8u) * STRIDE + kk, STRIDE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a_frag[0], b_frag[0], acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a_frag[0], b_frag[1], acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a_frag[1], b_frag[0], acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a_frag[1], b_frag[1], acc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* Cs = (threadgroup float*)smem_v;
    for (uint i = 0u; i < 2u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            simdgroup_store(
                acc[i][j],
                Cs + (sg_m * 16u + i * 8u) * BN + (sg_n * 16u + j * 8u),
                BN,
                ulong2(0, 0),
                false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < BM * BN; idx += 128u) {
        uint r = idx >> 5;
        uint c = idx & 31u;
        uint gr = m0 + r;
        uint gc = n0 + c;
        if (gr < M && gc < N) {
            output[(ulong)gr * N + gc] = bf16_rne(Cs[idx]);
        }
    }
}

// ============================================================================
// 4. Token / PLE Embedding Gather + Scale
// ============================================================================
kernel void fast_bf16_embed_gather(
    device const uint* token_ids [[buffer(0)]],
    device const ushort* table [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& total [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total) return;
    uint row = id / width;
    uint col = id % width;
    uint tok = token_ids[row];
    float v = bf16_load(table[(ulong)tok * width + col]) * scale;
    output[id] = bf16_rne(v);
}

// ============================================================================
// 5. Fused PLE Post-Projection: scale -> BF16 -> per-(m,layer) RMS -> combine -> transpose [layer, M, ple_dim]
//    1 SIMD group (32 threads) per (m, layer).
// ============================================================================
kernel void fast_bf16_ple_combine_transpose(
    device const ushort* projection [[buffer(0)]],
    device const ushort* ple_id [[buffer(1)]],
    device const ushort* ple_norm [[buffer(2)]],
    device ushort* per_layer_out [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& num_layers [[buffer(5)]],
    constant uint& ple_dim [[buffer(6)]],
    constant float& inv_sqrt_hidden [[buffer(7)]],
    constant float& epsilon [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    uint total_groups = M * num_layers;
    if (tg_id >= total_groups) return;
    uint m = tg_id / num_layers;
    uint layer = tg_id % num_layers;
    ulong packed = (ulong)num_layers * ple_dim;
    ulong in_base = (ulong)m * packed + (ulong)layer * ple_dim;
    ulong out_base = ((ulong)layer * M + m) * ple_dim;

    float energy = 0.0f;
    for (uint j = lane; j < ple_dim; j += 32u) {
        // Match exact BF16 rounding after scaling by 1/sqrt(hidden)
        ushort scaled_bits = bf16_rne(bf16_load(projection[in_base + j]) * inv_sqrt_hidden);
        float v = bf16_load(scaled_bits);
        energy += v * v;
    }
    energy = simd_sum(energy);
    float inv_rms = rsqrt(energy / (float)ple_dim + epsilon);
    constexpr float INV_SQRT_2 = 0.7071067811865475f;

    for (uint j = lane; j < ple_dim; j += 32u) {
        ushort scaled_bits = bf16_rne(bf16_load(projection[in_base + j]) * inv_sqrt_hidden);
        float v = bf16_load(scaled_bits) * inv_rms * bf16_load(ple_norm[j]);
        ushort ctx = bf16_rne(v);
        ushort sum_bits = bf16_rne(bf16_load(ctx) + bf16_load(ple_id[in_base + j]));
        per_layer_out[out_base + j] = bf16_rne(bf16_load(sum_bits) * INV_SQRT_2);
    }
}

// ============================================================================
// 6. Fast SIMD-group RMSNorm (128 threads = 4 SGs per row)
// ============================================================================
kernel void fast_bf16_rms(
    device const ushort* input [[buffer(0)]],
    device const ushort* scale [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    constant uint& has_scale [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float sg_sums[4];
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    ulong base = (ulong)row * width;

    float energy = 0.0f;
    for (uint j = tid; j < width; j += 128u) {
        float v = bf16_load(input[base + j]);
        energy += v * v;
    }
    energy = simd_sum(energy);
    if (lane == 0u) sg_sums[sg] = energy;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = sg_sums[0] + sg_sums[1] + sg_sums[2] + sg_sums[3];
    float inv_rms = rsqrt(total / (float)width + epsilon);

    for (uint j = tid; j < width; j += 128u) {
        float v = bf16_load(input[base + j]) * inv_rms;
        if (has_scale != 0u) v *= bf16_load(scale[j]);
        output[base + j] = bf16_rne(v);
    }
}

// ============================================================================
// 7. Fused Residual Add (in-place) + Next RMSNorm (128 threads = 4 SGs per row)
//    Preserves the exact BF16 rounding between residual addition and RMSNorm.
// ============================================================================
kernel void fast_bf16_add_rms(
    device ushort* residual [[buffer(0)]],
    device const ushort* delta [[buffer(1)]],
    device const ushort* scale [[buffer(2)]],
    device ushort* normed_out [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float sg_sums[4];
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    ulong base = (ulong)row * width;

    float energy = 0.0f;
    for (uint j = tid; j < width; j += 128u) {
        ushort h_bits = bf16_rne(bf16_load(residual[base + j]) + bf16_load(delta[base + j]));
        residual[base + j] = h_bits;
        float v = bf16_load(h_bits);
        energy += v * v;
    }
    energy = simd_sum(energy);
    if (lane == 0u) sg_sums[sg] = energy;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = sg_sums[0] + sg_sums[1] + sg_sums[2] + sg_sums[3];
    float inv_rms = rsqrt(total / (float)width + epsilon);

    for (uint j = tid; j < width; j += 128u) {
        float v = bf16_load(residual[base + j]) * inv_rms * bf16_load(scale[j]);
        normed_out[base + j] = bf16_rne(v);
    }
}

// ============================================================================
// 8. Fused Residual Add (in-place) + Optional Layer Scalar Multiply
// ============================================================================
kernel void fast_bf16_residual_add_scale(
    device ushort* residual [[buffer(0)]],
    device const ushort* delta [[buffer(1)]],
    device const ushort* layer_scalar [[buffer(2)]],
    constant uint& apply_scalar [[buffer(3)]],
    constant uint& total [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total) return;
    ushort sum_bits = bf16_rne(bf16_load(residual[id]) + bf16_load(delta[id]));
    if (apply_scalar != 0u) {
        sum_bits = bf16_rne(bf16_load(sum_bits) * bf16_load(layer_scalar[0]));
    }
    residual[id] = sum_bits;
}

// ============================================================================
// 9. Fused Q RMSNorm + Split-Half RoPE (in-place, 1 SIMD group per (m, head))
// ============================================================================
kernel void fast_bf16_q_norm_rope(
    device ushort* q [[buffer(0)]],
    device const ushort* q_norm [[buffer(1)]],
    device const float* inv_freq [[buffer(2)]],
    device const uint* row_pos [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    uint m = tg_id / heads;
    uint half_dim = head_dim >> 1;
    ulong base = (ulong)tg_id * head_dim;

    float energy = 0.0f;
    for (uint j = lane; j < head_dim; j += 32u) {
        float v = bf16_load(q[base + j]);
        energy += v * v;
    }
    energy = simd_sum(energy);
    float inv_rms = rsqrt(energy / (float)head_dim + epsilon);
    float pos = (float)row_pos[m];

    for (uint j = lane; j < half_dim; j += 32u) {
        float a = bf16_load(bf16_rne(bf16_load(q[base + j]) * inv_rms * bf16_load(q_norm[j])));
        float b = bf16_load(bf16_rne(bf16_load(q[base + j + half_dim]) * inv_rms * bf16_load(q_norm[j + half_dim])));
        float phase = pos * inv_freq[j];
        float c = bf16_load(bf16_rne(cos(phase)));
        float s = bf16_load(bf16_rne(sin(phase)));
        float ac = bf16_load(bf16_rne(a * c));
        float bs = bf16_load(bf16_rne(-b * s));
        float bc = bf16_load(bf16_rne(b * c));
        float a_s = bf16_load(bf16_rne(a * s));
        q[base + j] = bf16_rne(ac + bs);
        q[base + j + half_dim] = bf16_rne(bc + a_s);
    }
}

// ============================================================================
// 10. Fused K/V Norm + RoPE + Paged Device KV Cache Write (PAGE_TOKENS = 16)
//     KV pool layout: [phys_block, kv_head, 16, head_dim]
//     Block table layout: [slot, max_pages] -> phys_block
//     1 SIMD group per (m, kv_head).
// ============================================================================
kernel void fast_bf16_kv_norm_rope_write(
    device const ushort* k_in [[buffer(0)]],
    device const ushort* v_in [[buffer(1)]],
    device const ushort* k_norm [[buffer(2)]],
    device const float* inv_freq [[buffer(3)]],
    device const uint* row_slot [[buffer(4)]],
    device const uint* row_pos [[buffer(5)]],
    device const uint* block_table [[buffer(6)]],
    device ushort* k_cache [[buffer(7)]],
    device ushort* v_cache [[buffer(8)]],
    constant uint& kv_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& max_pages [[buffer(11)]],
    constant float& epsilon [[buffer(12)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    uint m = tg_id / kv_heads;
    uint kh = tg_id % kv_heads;
    uint half_dim = head_dim >> 1;
    ulong in_base = (ulong)tg_id * head_dim;

    uint slot = row_slot[m];
    uint pos = row_pos[m];
    uint phys_block = block_table[(ulong)slot * max_pages + (pos >> 4)];
    uint page_off = pos & 15u;
    ulong dst_base = (((ulong)phys_block * kv_heads + kh) * 16u + page_off) * head_dim;

    // Key: RMSNorm with k_norm + split-half RoPE
    float k_energy = 0.0f;
    float v_energy = 0.0f;
    for (uint j = lane; j < head_dim; j += 32u) {
        float kv = bf16_load(k_in[in_base + j]);
        float vv = bf16_load(v_in[in_base + j]);
        k_energy += kv * kv;
        v_energy += vv * vv;
    }
    float k_inv = rsqrt(simd_sum(k_energy) / (float)head_dim + epsilon);
    float v_inv = rsqrt(simd_sum(v_energy) / (float)head_dim + epsilon);
    float pos_f = (float)pos;

    for (uint j = lane; j < half_dim; j += 32u) {
        float a = bf16_load(bf16_rne(bf16_load(k_in[in_base + j]) * k_inv * bf16_load(k_norm[j])));
        float b = bf16_load(bf16_rne(bf16_load(k_in[in_base + j + half_dim]) * k_inv * bf16_load(k_norm[j + half_dim])));
        float phase = pos_f * inv_freq[j];
        float c = bf16_load(bf16_rne(cos(phase)));
        float s = bf16_load(bf16_rne(sin(phase)));
        float ac = bf16_load(bf16_rne(a * c));
        float bs = bf16_load(bf16_rne(-b * s));
        float bc = bf16_load(bf16_rne(b * c));
        float a_s = bf16_load(bf16_rne(a * s));
        k_cache[dst_base + j] = bf16_rne(ac + bs);
        k_cache[dst_base + j + half_dim] = bf16_rne(bc + a_s);
    }

    // Value: scale-free RMSNorm, no RoPE
    for (uint j = lane; j < head_dim; j += 32u) {
        v_cache[dst_base + j] = bf16_rne(bf16_load(v_in[in_base + j]) * v_inv);
    }
}

// ============================================================================
// 11. Batched Online-Softmax Paged FlashAttention (PAGE_TOKENS = 16)
//     4 SIMD groups (128 threads) per (m, head) splitting KV positions.
//     Gemma 4 attention scale is 1.0 (Q and K are RMS-normalized).
// ============================================================================
kernel void fast_bf16_flash_attn(
    device const ushort* q [[buffer(0)]],
    device const ushort* k_cache [[buffer(1)]],
    device const ushort* v_cache [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    device const uint* row_slot [[buffer(4)]],
    device const uint* row_pos [[buffer(5)]],
    device const uint* block_table [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& kv_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& max_pages [[buffer(10)]],
    constant uint& window [[buffer(11)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint m = tg_id / heads;
    uint head = tg_id % heads;
    uint kh = head / (heads / kv_heads);

    uint slot = row_slot[m];
    uint pos = row_pos[m];
    uint last = pos;
    uint first = (window != 0u && pos >= window - 1u) ? (pos - (window - 1u)) : 0u;

    ulong q_base = (ulong)tg_id * head_dim;
    device const uint* seq_blocks = block_table + (ulong)slot * max_pages;

    threadgroup float sm_m[4];
    threadgroup float sm_l[4];
    threadgroup float sm_o[4 * 512]; // Supports head_dim <= 512 (8 KiB)

    if (head_dim == 256u) {
        device const bfloat4* q4 = (device const bfloat4*)(q + q_base);
        float4 q0 = float4(q4[lane]);
        float4 q1 = float4(q4[lane + 32u]);
        float4 o0 = float4(0.0f);
        float4 o1 = float4(0.0f);
        float m_max = -INFINITY;
        float l_sum = 0.0f;

        for (uint j = first + sg; j <= last; j += 4u) {
            uint phys_block = seq_blocks[j >> 4];
            uint page_off = j & 15u;
            ulong kv_off = (((ulong)phys_block * kv_heads + kh) * 16u + page_off) * 256u;
            device const bfloat4* k4 = (device const bfloat4*)(k_cache + kv_off);
            device const bfloat4* v4 = (device const bfloat4*)(v_cache + kv_off);
            float partial = dot(q0, float4(k4[lane])) + dot(q1, float4(k4[lane + 32u]));
            float score = bf16_load(bf16_rne(simd_sum(partial)));

            float new_max = max(m_max, score);
            float alpha = (l_sum > 0.0f) ? exp(m_max - new_max) : 0.0f;
            float w = exp(score - new_max);
            l_sum = l_sum * alpha + w;
            m_max = new_max;
            o0 = o0 * alpha + w * float4(v4[lane]);
            o1 = o1 * alpha + w * float4(v4[lane + 32u]);
        }

        if (lane == 0u) {
            sm_m[sg] = m_max;
            sm_l[sg] = l_sum;
        }
        uint dst0 = sg * 256u + lane * 4u;
        uint dst1 = sg * 256u + (lane + 32u) * 4u;
        sm_o[dst0 + 0u] = o0.x; sm_o[dst0 + 1u] = o0.y; sm_o[dst0 + 2u] = o0.z; sm_o[dst0 + 3u] = o0.w;
        sm_o[dst1 + 0u] = o1.x; sm_o[dst1 + 1u] = o1.y; sm_o[dst1 + 2u] = o1.z; sm_o[dst1 + 3u] = o1.w;
    } else if (head_dim == 512u) {
        device const bfloat4* q4 = (device const bfloat4*)(q + q_base);
        float4 q0 = float4(q4[lane]);
        float4 q1 = float4(q4[lane + 32u]);
        float4 q2 = float4(q4[lane + 64u]);
        float4 q3 = float4(q4[lane + 96u]);
        float4 o0 = float4(0.0f), o1 = float4(0.0f), o2 = float4(0.0f), o3 = float4(0.0f);
        float m_max = -INFINITY;
        float l_sum = 0.0f;

        for (uint j = first + sg; j <= last; j += 4u) {
            uint phys_block = seq_blocks[j >> 4];
            uint page_off = j & 15u;
            ulong kv_off = (((ulong)phys_block * kv_heads + kh) * 16u + page_off) * 512u;
            device const bfloat4* k4 = (device const bfloat4*)(k_cache + kv_off);
            device const bfloat4* v4 = (device const bfloat4*)(v_cache + kv_off);
            float partial = dot(q0, float4(k4[lane]))
                          + dot(q1, float4(k4[lane + 32u]))
                          + dot(q2, float4(k4[lane + 64u]))
                          + dot(q3, float4(k4[lane + 96u]));
            float score = bf16_load(bf16_rne(simd_sum(partial)));

            float new_max = max(m_max, score);
            float alpha = (l_sum > 0.0f) ? exp(m_max - new_max) : 0.0f;
            float w = exp(score - new_max);
            l_sum = l_sum * alpha + w;
            m_max = new_max;
            o0 = o0 * alpha + w * float4(v4[lane]);
            o1 = o1 * alpha + w * float4(v4[lane + 32u]);
            o2 = o2 * alpha + w * float4(v4[lane + 64u]);
            o3 = o3 * alpha + w * float4(v4[lane + 96u]);
        }

        if (lane == 0u) {
            sm_m[sg] = m_max;
            sm_l[sg] = l_sum;
        }
        uint b = sg * 512u;
        uint d0 = b + lane * 4u;
        uint d1 = b + (lane + 32u) * 4u;
        uint d2 = b + (lane + 64u) * 4u;
        uint d3 = b + (lane + 96u) * 4u;
        sm_o[d0 + 0u] = o0.x; sm_o[d0 + 1u] = o0.y; sm_o[d0 + 2u] = o0.z; sm_o[d0 + 3u] = o0.w;
        sm_o[d1 + 0u] = o1.x; sm_o[d1 + 1u] = o1.y; sm_o[d1 + 2u] = o1.z; sm_o[d1 + 3u] = o1.w;
        sm_o[d2 + 0u] = o2.x; sm_o[d2 + 1u] = o2.y; sm_o[d2 + 2u] = o2.z; sm_o[d2 + 3u] = o2.w;
        sm_o[d3 + 0u] = o3.x; sm_o[d3 + 1u] = o3.y; sm_o[d3 + 2u] = o3.z; sm_o[d3 + 3u] = o3.w;
    } else {
        // General path for arbitrary head_dim <= 512
        float q_reg[16];
        float o_reg[16];
        uint n_elem = 0u;
        for (uint d = lane; d < head_dim; d += 32u) {
            q_reg[n_elem] = bf16_load(q[q_base + d]);
            o_reg[n_elem] = 0.0f;
            ++n_elem;
        }
        float m_max = -INFINITY;
        float l_sum = 0.0f;

        for (uint j = first + sg; j <= last; j += 4u) {
            uint phys_block = seq_blocks[j >> 4];
            uint page_off = j & 15u;
            ulong kv_off = (((ulong)phys_block * kv_heads + kh) * 16u + page_off) * head_dim;
            float partial = 0.0f;
            for (uint t = 0u; t < n_elem; ++t) {
                uint d = lane + t * 32u;
                partial += q_reg[t] * bf16_load(k_cache[kv_off + d]);
            }
            float score = bf16_load(bf16_rne(simd_sum(partial)));
            float new_max = max(m_max, score);
            float alpha = (l_sum > 0.0f) ? exp(m_max - new_max) : 0.0f;
            float w = exp(score - new_max);
            l_sum = l_sum * alpha + w;
            m_max = new_max;
            for (uint t = 0u; t < n_elem; ++t) {
                uint d = lane + t * 32u;
                o_reg[t] = o_reg[t] * alpha + w * bf16_load(v_cache[kv_off + d]);
            }
        }

        if (lane == 0u) {
            sm_m[sg] = m_max;
            sm_l[sg] = l_sum;
        }
        for (uint t = 0u; t < n_elem; ++t) {
            sm_o[sg * head_dim + lane + t * 32u] = o_reg[t];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_glob = max(max(sm_m[0], sm_m[1]), max(sm_m[2], sm_m[3]));
    float sc0 = (sm_l[0] > 0.0f) ? exp(sm_m[0] - m_glob) : 0.0f;
    float sc1 = (sm_l[1] > 0.0f) ? exp(sm_m[1] - m_glob) : 0.0f;
    float sc2 = (sm_l[2] > 0.0f) ? exp(sm_m[2] - m_glob) : 0.0f;
    float sc3 = (sm_l[3] > 0.0f) ? exp(sm_m[3] - m_glob) : 0.0f;
    float inv_l = 1.0f / (sm_l[0] * sc0 + sm_l[1] * sc1 + sm_l[2] * sc2 + sm_l[3] * sc3);

    for (uint d = tid; d < head_dim; d += 128u) {
        float acc = sm_o[0u * head_dim + d] * sc0
                  + sm_o[1u * head_dim + d] * sc1
                  + sm_o[2u * head_dim + d] * sc2
                  + sm_o[3u * head_dim + d] * sc3;
        output[q_base + d] = bf16_rne(acc * inv_l);
    }
}

// ============================================================================
// 12. Exact Table GeGLU: table[gate] * up with non-finite gate detection
// ============================================================================
kernel void fast_bf16_geglu_table(
    device const ushort* gate [[buffer(0)]],
    device const ushort* up [[buffer(1)]],
    device const ushort* table [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    device atomic_uint* error_flag [[buffer(4)]],
    constant uint& total [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total) return;
    ushort bits = gate[id];
    if ((bits & 0x7f80u) == 0x7f80u) {
        atomic_fetch_or_explicit(error_flag, 1u, memory_order_relaxed);
        output[id] = 0;
        return;
    }
    output[id] = fast_bf16_product_rne(table[bits], up[id]);
}

// ============================================================================
// 13. Gather Output Rows + Final RMSNorm (128 threads = 4 SGs per output row)
// ============================================================================
// 13a. Gather Selected Rows (128 threads per output row)
//      Used to prune non-output tokens after the final KV-owner layer writes
//      its KV cache, eliminating wasted work across shared-KV tail layers.
// ============================================================================
kernel void fast_bf16_gather_rows(
    device const ushort* src [[buffer(0)]],
    device const uint* row_indices [[buffer(1)]],
    device ushort* dst [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint out_row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint src_row = row_indices[out_row];
    ulong in_base = (ulong)src_row * width;
    ulong out_base = (ulong)out_row * width;
    for (uint j = tid; j < width; j += 128u) {
        dst[out_base + j] = src[in_base + j];
    }
}

// ============================================================================
// 13b. Gather Output Rows + Final RMSNorm (128 threads per output row)
// ============================================================================
kernel void fast_bf16_gather_rows_rms(
    device const ushort* hidden [[buffer(0)]],
    device const uint* output_rows [[buffer(1)]],
    device const ushort* scale [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint out_row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float sg_sums[4];
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint src_row = output_rows[out_row];
    ulong in_base = (ulong)src_row * width;
    ulong out_base = (ulong)out_row * width;

    float energy = 0.0f;
    for (uint j = tid; j < width; j += 128u) {
        float v = bf16_load(hidden[in_base + j]);
        energy += v * v;
    }
    energy = simd_sum(energy);
    if (lane == 0u) sg_sums[sg] = energy;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = sg_sums[0] + sg_sums[1] + sg_sums[2] + sg_sums[3];
    float inv_rms = rsqrt(total / (float)width + epsilon);

    for (uint j = tid; j < width; j += 128u) {
        float v = bf16_load(hidden[in_base + j]) * inv_rms * bf16_load(scale[j]);
        output[out_base + j] = bf16_rne(v);
    }
}

// ============================================================================
// 14. In-Place Gemma 4 Softcap + Deterministic Argmax (256 threads per row)
// ============================================================================
kernel void fast_bf16_softcap_argmax(
    device ushort* logits [[buffer(0)]],
    device uint* argmax_out [[buffer(1)]],
    device atomic_uint* error_flag [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    constant float& cap [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float sm_val[256];
    threadgroup uint sm_idx[256];

    ulong base = (ulong)row * vocab;
    float best_val = -INFINITY;
    uint best_idx = 0xffffffffu;

    for (uint j = tid; j < vocab; j += 256u) {
        ushort bits = logits[base + j];
        if ((bits & 0x7f80u) == 0x7f80u) {
            atomic_fetch_or_explicit(error_flag, 2u, memory_order_relaxed);
            logits[base + j] = 0x7fc0u;
            continue;
        }
        float val = bf16_load(bits);
        float divided = bf16_load(bf16_rne(val / cap));
        float hyperbolic = bf16_load(bf16_rne(tanh(divided)));
        ushort capped = bf16_rne(hyperbolic * cap);
        logits[base + j] = capped;
        float f = bf16_load(capped);
        if (best_idx == 0xffffffffu || f > best_val || (f == best_val && j < best_idx)) {
            best_val = f;
            best_idx = j;
        }
    }

    sm_val[tid] = best_val;
    sm_idx[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride != 0u; stride >>= 1) {
        if (tid < stride) {
            float ov = sm_val[tid + stride];
            uint oi = sm_idx[tid + stride];
            float cv = sm_val[tid];
            uint ci = sm_idx[tid];
            if (oi != 0xffffffffu && (ci == 0xffffffffu || ov > cv || (ov == cv && oi < ci))) {
                sm_val[tid] = ov;
                sm_idx[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        argmax_out[row] = sm_idx[0];
    }
}
