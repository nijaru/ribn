#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Metal 3.1 Quantized Affine Group-64 Kernels (Q4 & Q8) for Ribn
//
// Weight layout (compatible with MLX affine group_size=64):
//   - Q4: packed[N, K/8] as uint (8 LSB-first 4-bit nibbles per uint),
//         scales_biases[N, K/64] as bfloat2 (.x = scale, .y = bias).
//   - Q8: packed[N, K] as uchar,
//         scales_biases[N, K/64] as bfloat2 (.x = scale, .y = bias).
//   - Reconstructed weight: w = float(q) * float(scale) + float(bias).
//   - K must be a positive multiple of 64.
// ============================================================================

inline ushort q_bf16_rne(float x) {
    uint u = as_type<uint>(x);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        return (u & 0x007fffffu) ? ushort((u >> 16) | 0x0040u) : ushort(u >> 16);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return ushort(u >> 16);
}

inline uint q_rne_shift(uint value, uint shift) {
    if (shift == 0) return value;
    if (shift >= 32) return 0;
    uint truncated = value >> shift;
    uint remainder = value & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    return truncated + uint(remainder > halfway || (remainder == halfway && (truncated & 1u)));
}

inline ushort q_bf16_product_rne(ushort a, ushort b) {
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
    if (exponent <= 0) return ushort(sign | q_rne_shift(product, uint(-power - 133)));
    uint rounded = q_rne_shift(product, uint(top - 7));
    if (rounded == 256u) { rounded = 128u; ++exponent; }
    if (exponent >= 255) return ushort(sign | 0x7f80u);
    return ushort(sign | (uint(exponent) << 7) | (rounded - 128u));
}

inline ushort q_geglu_apply(ushort g_bits, ushort u_bits, device const ushort* gelu_table, device atomic_uint* error_flag) {
    if ((g_bits & 0x7f80u) == 0x7f80u) {
        atomic_fetch_or_explicit(error_flag, 1u, memory_order_relaxed);
        return 0;
    }
    return q_bf16_product_rne(gelu_table[g_bits], u_bits);
}

inline float q4_dot16(uint2 p, float4 x0, float4 x1, float4 x2, float4 x3) {
    ushort a = ushort(p.x & 0xFFFFu);
    ushort b = ushort(p.x >> 16);
    ushort c = ushort(p.y & 0xFFFFu);
    ushort d = ushort(p.y >> 16);
    float s = float(a & 0x000Fu) * x0.x
            + float(a & 0x00F0u) * x0.y
            + float(a & 0x0F00u) * x0.z
            + float(a & 0xF000u) * x0.w;
    s += float(b & 0x000Fu) * x1.x
       + float(b & 0x00F0u) * x1.y
       + float(b & 0x0F00u) * x1.z
       + float(b & 0xF000u) * x1.w;
    s += float(c & 0x000Fu) * x2.x
       + float(c & 0x00F0u) * x2.y
       + float(c & 0x0F00u) * x2.z
       + float(c & 0xF000u) * x2.w;
    s += float(d & 0x000Fu) * x3.x
       + float(d & 0x00F0u) * x3.y
       + float(d & 0x0F00u) * x3.z
       + float(d & 0xF000u) * x3.w;
    return s;
}

// ============================================================================
// 1. Q4 Coalesced Single-Row GEMV (M = 1)
//    2 SIMD groups (64 threads) per TG, 2 output rows per SG (4 rows/TG).
//    4 lanes per 64-weight group => 8 groups (512 weights) per iteration,
//    each lane loading uint2 (8 bytes = 16 nibbles) -> 256B coalesced L2 read.
// ============================================================================
kernel void fast_q4_gemv_m1(
    device const bfloat* input [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n0 = tg_id * 4u + sg * 2u;
    if (n0 >= N) return;

    bool has_second = (n0 + 1u) < N;
    uint num_groups = K >> 6;
    uint k_u64 = K >> 4; // uint2 per row (16 weights per uint2)
    uint sub_g = lane >> 2; // 0..7
    uint sub_l = lane & 3u; // 0..3

    device const uint2* w_base = (device const uint2*)packed;
    device const uint2* w0 = w_base + (ulong)n0 * k_u64;
    device const uint2* w1 = has_second ? (w0 + k_u64) : w0;

    device const bfloat2* sb0 = scales_biases + (ulong)n0 * num_groups;
    device const bfloat2* sb1 = has_second ? (sb0 + num_groups) : sb0;

    device const bfloat4* x4 = (device const bfloat4*)input;
    const float4 nib_scale = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint g0 = 0u; g0 < num_groups; g0 += 8u) {
        uint g = g0 + sub_g;
        if (g < num_groups) {
            uint x_idx = g * 16u + sub_l * 4u;
            float4 r0 = float4(x4[x_idx + 0u]);
            float4 r1 = float4(x4[x_idx + 1u]);
            float4 r2 = float4(x4[x_idx + 2u]);
            float4 r3 = float4(x4[x_idx + 3u]);
            float xsum = ((r0.x + r0.y) + (r0.z + r0.w))
                       + ((r1.x + r1.y) + (r1.z + r1.w))
                       + ((r2.x + r2.y) + (r2.z + r2.w))
                       + ((r3.x + r3.y) + (r3.z + r3.w));
            float4 s0 = r0 * nib_scale;
            float4 s1 = r1 * nib_scale;
            float4 s2 = r2 * nib_scale;
            float4 s3 = r3 * nib_scale;

            uint idx = g * 4u + sub_l;
            bfloat2 p_sb0 = sb0[g];
            acc0 += q4_dot16(w0[idx], s0, s1, s2, s3) * float(p_sb0.x) + xsum * float(p_sb0.y);

            if (has_second) {
                bfloat2 p_sb1 = sb1[g];
                acc1 += q4_dot16(w1[idx], s0, s1, s2, s3) * float(p_sb1.x) + xsum * float(p_sb1.y);
            }
        }
    }

    float o0 = simd_sum(acc0);
    float o1 = has_second ? simd_sum(acc1) : 0.0f;
    if (lane == 0u) {
        output[n0] = q_bf16_rne(o0);
        if (has_second) output[n0 + 1u] = q_bf16_rne(o1);
    }
}

// ============================================================================
// 2. Q4 Batched GEMV (M = 2..=4)
//    2D grid (gx = ceil(M/BM), gy = ceil(N/4)), 4 SGs (128 threads), 1 col/SG.
//    4 lanes per 64-weight group (uint2 = 16 weights per lane), dequantizing
//    each 16-weight chunk ONCE into registers and reusing across all M rows.
// ============================================================================
template <uint BM_MAX>
inline void q4_gemv_batch_impl(
    device const bfloat* input,
    device const uint* packed,
    device const bfloat2* scales_biases,
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
    uint n = tg_pos.y * 4u + sg;
    if (m0 >= M || n >= N) return;
    uint m_rem = min(BM_MAX, M - m0);

    uint num_groups = K >> 6;
    uint k_u64 = K >> 4;
    uint k4 = K >> 2;
    uint sub_g = lane >> 2;
    uint sub_l = lane & 3u;

    device const uint2* w0 = (device const uint2*)packed + (ulong)n * k_u64;
    device const bfloat2* sb0 = scales_biases + (ulong)n * num_groups;
    const float4 nib_scale = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

    float acc[BM_MAX];
    for (uint m = 0u; m < BM_MAX; ++m) {
        acc[m] = 0.0f;
    }

    for (uint g0 = 0u; g0 < num_groups; g0 += 8u) {
        uint g = g0 + sub_g;
        if (g < num_groups) {
            uint2 p0 = w0[g * 4u + sub_l];
            bfloat2 s0 = sb0[g];
            float4 sv = float(s0.x) * nib_scale;
            float bv = float(s0.y);

            ushort a = ushort(p0.x & 0xFFFFu);
            ushort b = ushort(p0.x >> 16);
            ushort c = ushort(p0.y & 0xFFFFu);
            ushort d = ushort(p0.y >> 16);
            float4 wt0 = float4(float(a & 0x000Fu), float(a & 0x00F0u), float(a & 0x0F00u), float(a & 0xF000u)) * sv + bv;
            float4 wt1 = float4(float(b & 0x000Fu), float(b & 0x00F0u), float(b & 0x0F00u), float(b & 0xF000u)) * sv + bv;
            float4 wt2 = float4(float(c & 0x000Fu), float(c & 0x00F0u), float(c & 0x0F00u), float(c & 0xF000u)) * sv + bv;
            float4 wt3 = float4(float(d & 0x000Fu), float(d & 0x00F0u), float(d & 0x0F00u), float(d & 0xF000u)) * sv + bv;

            uint x_off = g * 16u + sub_l * 4u;
            #pragma unroll
            for (uint m = 0u; m < BM_MAX; ++m) {
                uint row = m0 + min(m, m_rem - 1u);
                device const bfloat4* xr = (device const bfloat4*)(input) + (ulong)row * k4;
                acc[m] += dot(float4(xr[x_off + 0u]), wt0)
                        + dot(float4(xr[x_off + 1u]), wt1)
                        + dot(float4(xr[x_off + 2u]), wt2)
                        + dot(float4(xr[x_off + 3u]), wt3);
            }
        }
    }

    for (uint m = 0u; m < BM_MAX; ++m) {
        float o = simd_sum(acc[m]);
        if (lane == 0u && m < m_rem) {
            output[(ulong)(m0 + m) * N + n] = q_bf16_rne(o);
        }
    }
}

kernel void fast_q4_gemv_m2(
    device const bfloat* input [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    q4_gemv_batch_impl<2>(input, packed, scales_biases, output, M, N, K, tg_pos, tid);
}

kernel void fast_q4_gemv_m4(
    device const bfloat* input [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    q4_gemv_batch_impl<4>(input, packed, scales_biases, output, M, N, K, tg_pos, tid);
}

// ============================================================================
// 3. Q4 Mixed-Precision BF16->FP32 Tiled GEMM (BM=8 & BM=16, BN=32, BK=64)
//    1 full 64-weight quant group per iteration:
//    - 4 threads/row load coalesced uint2 (8 bytes = 16 nibbles = 32B/row)
//    - Half the threadgroup barriers vs BK=32; hardware simdgroup_bfloat8x8 MMA.
// ============================================================================
inline void q4_unpack_u32_to_bs(uint w, float4 sv, float bv, threadgroup bfloat* dst) {
    ushort lo = ushort(w & 0xFFFFu);
    ushort hi = ushort(w >> 16);
    *(threadgroup bfloat4*)(dst + 0u) = bfloat4(float4(
        float(lo & 0x000Fu),
        float(lo & 0x00F0u),
        float(lo & 0x0F00u),
        float(lo & 0xF000u)) * sv + bv);
    *(threadgroup bfloat4*)(dst + 4u) = bfloat4(float4(
        float(hi & 0x000Fu),
        float(hi & 0x00F0u),
        float(hi & 0x0F00u),
        float(hi & 0xF000u)) * sv + bv);
}

kernel void fast_q4_gemm_m8(
    device const bfloat* input [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint BM = 8u;
    constexpr uint BN = 32u;
    constexpr uint BK = 64u;
    constexpr uint STRIDE = 72u;

    uint m0 = tg_pos.x * BM;
    uint n0 = tg_pos.y * BN;
    if (m0 >= M || n0 >= N) return;

    threadgroup bfloat4 smem_v[720]; // As: 8*72 (144 v4), Bs: 32*72 (576 v4) = 5760 B
    threadgroup bfloat* As = (threadgroup bfloat*)smem_v;
    threadgroup bfloat* Bs = As + BM * STRIDE;

    uint sg = tid >> 5; // 0..3 -> each SG owns an 8x8 tile along N

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // As loader: 8 rows x 16 threads/row (each thread loads 1 bfloat4 = 4 bfloats)
    uint ar = tid >> 4;         // 0..7
    uint ac = (tid & 15u) << 2; // 0, 4, ..., 60

    // Bs loader: 32 rows x 4 threads/row (each thread loads uint2 = 16 Q4 weights)
    uint wr = tid >> 2;         // 0..31
    uint wc = tid & 3u;         // 0..3
    uint wc_elem = wc << 4;     // 0, 16, 32, 48

    uint num_groups = K >> 6;
    uint k_u64 = K >> 4;
    device const uint2* packed2 = (device const uint2*)packed;

    uint rm0 = min(m0 + ar, M - 1u);
    uint rn = min(n0 + wr, N - 1u);

    ulong a_row0 = (ulong)rm0 * K + ac;
    ulong w_row = (ulong)rn * k_u64 + wc;
    ulong sb_row = (ulong)rn * num_groups;
    const float4 nib_scale = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

    for (uint g = 0u; g < num_groups; ++g) {
        uint k0 = g << 6;
        *(threadgroup bfloat4*)(As + ar * STRIDE + ac) =
            *(device const bfloat4*)(input + a_row0 + k0);

        uint2 q = packed2[w_row + g * 4u];
        bfloat2 sb = scales_biases[sb_row + g];
        float4 sv = float(sb.x) * nib_scale;
        float bv = float(sb.y);
        threadgroup bfloat* b_dst = Bs + wr * STRIDE + wc_elem;
        q4_unpack_u32_to_bs(q.x, sv, bv, b_dst + 0u);
        q4_unpack_u32_to_bs(q.y, sv, bv, b_dst + 8u);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_bfloat8x8 a_frag;
            simdgroup_bfloat8x8 b_frag;
            simdgroup_load(a_frag, As + kk, STRIDE, ulong2(0, 0), false);
            simdgroup_load(b_frag, Bs + (sg * 8u) * STRIDE + kk, STRIDE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* Cs = (threadgroup float*)smem_v;
    simdgroup_store(acc, Cs + sg * 8u, BN, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < BM * BN; idx += 128u) {
        uint r = idx >> 5;
        uint c = idx & 31u;
        uint gr = m0 + r;
        uint gc = n0 + c;
        if (gr < M && gc < N) {
            output[(ulong)gr * N + gc] = q_bf16_rne(Cs[idx]);
        }
    }
}

kernel void fast_q4_gemm_m16(
    device const bfloat* input [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint BM = 16u;
    constexpr uint BN = 32u;
    constexpr uint BK = 64u;
    constexpr uint STRIDE = 72u;

    uint m0 = tg_pos.x * BM;
    uint n0 = tg_pos.y * BN;
    if (m0 >= M || n0 >= N) return;

    threadgroup bfloat4 smem_v[864]; // As: 16*72 (288 v4), Bs: 32*72 (576 v4) = 6912 B
    threadgroup bfloat* As = (threadgroup bfloat*)smem_v;
    threadgroup bfloat* Bs = As + BM * STRIDE;

    uint sg = tid >> 5;
    uint sg_m = sg >> 1;
    uint sg_n = sg & 1u;

    simdgroup_float8x8 acc[2];
    acc[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // As loader: 16 rows x 8 threads/row (each thread loads 2x bfloat4 = 8 bfloats)
    uint ar = tid >> 3;        // 0..15
    uint ac = (tid & 7u) << 3; // 0, 8, ..., 56

    // Bs loader: 32 rows x 4 threads/row (each thread loads uint2 = 16 Q4 weights)
    uint wr = tid >> 2;        // 0..31
    uint wc = tid & 3u;        // 0..3
    uint wc_elem = wc << 4;    // 0, 16, 32, 48

    uint num_groups = K >> 6;
    uint k_u64 = K >> 4;
    device const uint2* packed2 = (device const uint2*)packed;

    uint rm0 = min(m0 + ar, M - 1u);
    uint rn = min(n0 + wr, N - 1u);

    ulong a_row0 = (ulong)rm0 * K + ac;
    ulong w_row = (ulong)rn * k_u64 + wc;
    ulong sb_row = (ulong)rn * num_groups;
    const float4 nib_scale = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

    for (uint g = 0u; g < num_groups; ++g) {
        uint k0 = g << 6;
        device const bfloat4* a_ptr = (device const bfloat4*)(input + a_row0 + k0);
        *(threadgroup bfloat4*)(As + ar * STRIDE + ac + 0u) = a_ptr[0];
        *(threadgroup bfloat4*)(As + ar * STRIDE + ac + 4u) = a_ptr[1];

        uint2 q = packed2[w_row + g * 4u];
        bfloat2 sb = scales_biases[sb_row + g];
        float4 sv = float(sb.x) * nib_scale;
        float bv = float(sb.y);
        threadgroup bfloat* b_dst = Bs + wr * STRIDE + wc_elem;
        q4_unpack_u32_to_bs(q.x, sv, bv, b_dst + 0u);
        q4_unpack_u32_to_bs(q.y, sv, bv, b_dst + 8u);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_bfloat8x8 a_frag;
            simdgroup_bfloat8x8 b_frag0;
            simdgroup_bfloat8x8 b_frag1;
            simdgroup_load(a_frag, As + (sg_m * 8u) * STRIDE + kk, STRIDE, ulong2(0, 0), false);
            simdgroup_load(b_frag0, Bs + (sg_n * 16u + 0u) * STRIDE + kk, STRIDE, ulong2(0, 0), true);
            simdgroup_load(b_frag1, Bs + (sg_n * 16u + 8u) * STRIDE + kk, STRIDE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0], a_frag, b_frag0, acc[0]);
            simdgroup_multiply_accumulate(acc[1], a_frag, b_frag1, acc[1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* Cs = (threadgroup float*)smem_v;
    simdgroup_store(acc[0], Cs + (sg_m * 8u) * BN + (sg_n * 16u + 0u), BN, ulong2(0, 0), false);
    simdgroup_store(acc[1], Cs + (sg_m * 8u) * BN + (sg_n * 16u + 8u), BN, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < BM * BN; idx += 128u) {
        uint r = idx >> 5;
        uint c = idx & 31u;
        uint gr = m0 + r;
        uint gc = n0 + c;
        if (gr < M && gc < N) {
            output[(ulong)gr * N + gc] = q_bf16_rne(Cs[idx]);
        }
    }
}

// ============================================================================
// 4. Q4 Mixed-Precision BF16->FP32 Tiled GEMM (BM=32, BN=32, BK=64, 4 SGs)
//    Used for M > 16. 1 full 64-weight quant group per iteration (9 KiB smem).
// ============================================================================
kernel void fast_q4_gemm_m32(
    device const bfloat* input [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint BM = 32u;
    constexpr uint BN = 32u;
    constexpr uint BK = 64u;
    constexpr uint STRIDE = 72u;

    uint m0 = tg_pos.x * BM;
    uint n0 = tg_pos.y * BN;
    if (m0 >= M || n0 >= N) return;

    threadgroup bfloat4 smem_v[1152]; // As: 32*72 (576 v4), Bs: 32*72 (576 v4) = 9216 B
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

    uint r32 = tid >> 2;        // 0..31
    uint c4 = tid & 3u;         // 0..3
    uint c_elem = c4 << 4;      // 0, 16, 32, 48

    uint num_groups = K >> 6;
    uint k_u64 = K >> 4;
    device const uint2* packed2 = (device const uint2*)packed;

    uint rm = min(m0 + r32, M - 1u);
    uint rn = min(n0 + r32, N - 1u);

    ulong a_row = (ulong)rm * K + c_elem;
    ulong w_row = (ulong)rn * k_u64 + c4;
    ulong sb_row = (ulong)rn * num_groups;
    const float4 nib_scale = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

    for (uint g = 0u; g < num_groups; ++g) {
        uint k0 = g << 6;
        device const bfloat4* a_ptr = (device const bfloat4*)(input + a_row + k0);
        threadgroup bfloat* a_dst = As + r32 * STRIDE + c_elem;
        *(threadgroup bfloat4*)(a_dst + 0u) = a_ptr[0];
        *(threadgroup bfloat4*)(a_dst + 4u) = a_ptr[1];
        *(threadgroup bfloat4*)(a_dst + 8u) = a_ptr[2];
        *(threadgroup bfloat4*)(a_dst + 12u) = a_ptr[3];

        uint2 q = packed2[w_row + g * 4u];
        bfloat2 sb = scales_biases[sb_row + g];
        float4 sv = float(sb.x) * nib_scale;
        float bv = float(sb.y);
        threadgroup bfloat* b_dst = Bs + r32 * STRIDE + c_elem;
        q4_unpack_u32_to_bs(q.x, sv, bv, b_dst + 0u);
        q4_unpack_u32_to_bs(q.y, sv, bv, b_dst + 8u);

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
            output[(ulong)gr * N + gc] = q_bf16_rne(Cs[idx]);
        }
    }
}

// ============================================================================
// 5. Q4 Embedding Gather + Scale (1 thread per 4 elements)
// ============================================================================
kernel void fast_q4_embed_gather(
    device const uint* token_ids [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    constant uint& total_quads [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_quads) return;
    uint quads_per_row = width >> 2;
    uint num_groups = width >> 6;
    uint row = id / quads_per_row;
    uint col4 = id % quads_per_row;
    uint g = col4 >> 4;
    uint tok = token_ids[row];

    device const ushort* packed16 = (device const ushort*)packed;
    ushort bits = packed16[(ulong)tok * quads_per_row + col4];
    bfloat2 sb = scales_biases[(ulong)tok * num_groups + g];
    float sv = float(sb.x);
    float bv = float(sb.y);

    ulong dst = (ulong)row * width + (ulong)col4 * 4u;
    output[dst + 0u] = q_bf16_rne((float(bits & 0xFu) * sv + bv) * scale);
    output[dst + 1u] = q_bf16_rne((float((bits >> 4) & 0xFu) * sv + bv) * scale);
    output[dst + 2u] = q_bf16_rne((float((bits >> 8) & 0xFu) * sv + bv) * scale);
    output[dst + 3u] = q_bf16_rne((float((bits >> 12) & 0xFu) * sv + bv) * scale);
}

// ============================================================================
// 6. Q8 Coalesced Single-Row GEMV (M = 1)
//    2 SIMD groups (64 threads) per TG, 4 output rows per SG (8 rows/TG).
//    4 lanes per 64-weight group => 8 groups (512 weights) per iteration,
//    each lane loading uint4 (16 bytes = 16 uchar weights) -> 512B coalesced read.
// ============================================================================
inline float q8_dot16(uint4 p, float4 x0, float4 x1, float4 x2, float4 x3) {
    uchar4 a = as_type<uchar4>(p.x);
    uchar4 b = as_type<uchar4>(p.y);
    uchar4 c = as_type<uchar4>(p.z);
    uchar4 d = as_type<uchar4>(p.w);
    return dot(float4(a), x0) + dot(float4(b), x1) + dot(float4(c), x2) + dot(float4(d), x3);
}

kernel void fast_q8_gemv_m1(
    device const bfloat* input [[buffer(0)]],
    device const uchar* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n0 = tg_id * 4u + sg * 2u;
    if (n0 >= N) return;

    bool has_second = (n0 + 1u) < N;
    uint num_groups = K >> 6;
    uint k_u128 = K >> 4; // uint4 per row (16 weights per uint4)
    uint sub_g = lane >> 2;
    uint sub_l = lane & 3u;

    device const uint4* w_base = (device const uint4*)packed;
    device const uint4* w0 = w_base + (ulong)n0 * k_u128;
    device const uint4* w1 = has_second ? (w0 + k_u128) : w0;

    device const bfloat2* sb0 = scales_biases + (ulong)n0 * num_groups;
    device const bfloat2* sb1 = has_second ? (sb0 + num_groups) : sb0;

    device const bfloat4* x4 = (device const bfloat4*)input;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint g0 = 0u; g0 < num_groups; g0 += 8u) {
        uint g = g0 + sub_g;
        if (g < num_groups) {
            uint x_idx = g * 16u + sub_l * 4u;
            float4 r0 = float4(x4[x_idx + 0u]);
            float4 r1 = float4(x4[x_idx + 1u]);
            float4 r2 = float4(x4[x_idx + 2u]);
            float4 r3 = float4(x4[x_idx + 3u]);
            float xsum = ((r0.x + r0.y) + (r0.z + r0.w))
                       + ((r1.x + r1.y) + (r1.z + r1.w))
                       + ((r2.x + r2.y) + (r2.z + r2.w))
                       + ((r3.x + r3.y) + (r3.z + r3.w));

            uint idx = g * 4u + sub_l;
            bfloat2 p_sb0 = sb0[g];
            acc0 += q8_dot16(w0[idx], r0, r1, r2, r3) * float(p_sb0.x) + xsum * float(p_sb0.y);

            if (has_second) {
                bfloat2 p_sb1 = sb1[g];
                acc1 += q8_dot16(w1[idx], r0, r1, r2, r3) * float(p_sb1.x) + xsum * float(p_sb1.y);
            }
        }
    }

    float o0 = simd_sum(acc0);
    float o1 = has_second ? simd_sum(acc1) : 0.0f;
    if (lane == 0u) {
        output[n0] = q_bf16_rne(o0);
        if (has_second) output[n0 + 1u] = q_bf16_rne(o1);
    }
}

// ============================================================================
// 7. Q8 Batched GEMV (M = 2..=4)
//    4 SGs (128 threads), 1 col/SG, unpacking each 16-weight chunk ONCE.
// ============================================================================
template <uint BM_MAX>
inline void q8_gemv_batch_impl(
    device const bfloat* input,
    device const uchar* packed,
    device const bfloat2* scales_biases,
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
    uint n = tg_pos.y * 4u + sg;
    if (m0 >= M || n >= N) return;
    uint m_rem = min(BM_MAX, M - m0);

    uint num_groups = K >> 6;
    uint k_u128 = K >> 4;
    uint k4 = K >> 2;
    uint sub_g = lane >> 2;
    uint sub_l = lane & 3u;

    device const uint4* w0 = (device const uint4*)packed + (ulong)n * k_u128;
    device const bfloat2* sb0 = scales_biases + (ulong)n * num_groups;

    float acc[BM_MAX];
    for (uint m = 0u; m < BM_MAX; ++m) {
        acc[m] = 0.0f;
    }

    for (uint g0 = 0u; g0 < num_groups; g0 += 8u) {
        uint g = g0 + sub_g;
        if (g < num_groups) {
            uint4 p0 = w0[g * 4u + sub_l];
            bfloat2 s0 = sb0[g];
            float sv = float(s0.x);
            float bv = float(s0.y);
            float4 wt0 = float4(as_type<uchar4>(p0.x)) * sv + bv;
            float4 wt1 = float4(as_type<uchar4>(p0.y)) * sv + bv;
            float4 wt2 = float4(as_type<uchar4>(p0.z)) * sv + bv;
            float4 wt3 = float4(as_type<uchar4>(p0.w)) * sv + bv;

            uint x_off = g * 16u + sub_l * 4u;
            #pragma unroll
            for (uint m = 0u; m < BM_MAX; ++m) {
                uint row = m0 + min(m, m_rem - 1u);
                device const bfloat4* xr = (device const bfloat4*)(input) + (ulong)row * k4;
                acc[m] += dot(float4(xr[x_off + 0u]), wt0)
                        + dot(float4(xr[x_off + 1u]), wt1)
                        + dot(float4(xr[x_off + 2u]), wt2)
                        + dot(float4(xr[x_off + 3u]), wt3);
            }
        }
    }

    for (uint m = 0u; m < BM_MAX; ++m) {
        float o = simd_sum(acc[m]);
        if (lane == 0u && m < m_rem) {
            output[(ulong)(m0 + m) * N + n] = q_bf16_rne(o);
        }
    }
}

kernel void fast_q8_gemv_m2(
    device const bfloat* input [[buffer(0)]],
    device const uchar* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    q8_gemv_batch_impl<2>(input, packed, scales_biases, output, M, N, K, tg_pos, tid);
}

kernel void fast_q8_gemv_m4(
    device const bfloat* input [[buffer(0)]],
    device const uchar* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    q8_gemv_batch_impl<4>(input, packed, scales_biases, output, M, N, K, tg_pos, tid);
}

// ============================================================================
// 8. Q8 Mixed-Precision BF16->FP32 Tiled GEMM (BM=8 & BM=16, BN=32, BK=64)
//    1 full 64-weight quant group per iteration (uint4 loads).
// ============================================================================
kernel void fast_q8_gemm_m8(
    device const bfloat* input [[buffer(0)]],
    device const uchar* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint BM = 8u;
    constexpr uint BN = 32u;
    constexpr uint BK = 64u;
    constexpr uint STRIDE = 72u;

    uint m0 = tg_pos.x * BM;
    uint n0 = tg_pos.y * BN;
    if (m0 >= M || n0 >= N) return;

    threadgroup bfloat4 smem_v[720];
    threadgroup bfloat* As = (threadgroup bfloat*)smem_v;
    threadgroup bfloat* Bs = As + BM * STRIDE;

    uint sg = tid >> 5;

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint ar = tid >> 4;
    uint ac = (tid & 15u) << 2;
    uint wr = tid >> 2;
    uint wc = tid & 3u;
    uint wc_elem = wc << 4;

    uint num_groups = K >> 6;
    uint k_u128 = K >> 4;
    device const uint4* packed4 = (device const uint4*)packed;

    uint rm0 = min(m0 + ar, M - 1u);
    uint rn = min(n0 + wr, N - 1u);

    ulong a_row0 = (ulong)rm0 * K + ac;
    ulong w_row = (ulong)rn * k_u128 + wc;
    ulong sb_row = (ulong)rn * num_groups;

    for (uint g = 0u; g < num_groups; ++g) {
        uint k0 = g << 6;
        *(threadgroup bfloat4*)(As + ar * STRIDE + ac) =
            *(device const bfloat4*)(input + a_row0 + k0);

        uint4 q = packed4[w_row + g * 4u];
        bfloat2 sb = scales_biases[sb_row + g];
        float sv = float(sb.x);
        float bv = float(sb.y);
        threadgroup bfloat* b_dst = Bs + wr * STRIDE + wc_elem;
        *(threadgroup bfloat4*)(b_dst + 0u) = bfloat4(float4(as_type<uchar4>(q.x)) * sv + bv);
        *(threadgroup bfloat4*)(b_dst + 4u) = bfloat4(float4(as_type<uchar4>(q.y)) * sv + bv);
        *(threadgroup bfloat4*)(b_dst + 8u) = bfloat4(float4(as_type<uchar4>(q.z)) * sv + bv);
        *(threadgroup bfloat4*)(b_dst + 12u) = bfloat4(float4(as_type<uchar4>(q.w)) * sv + bv);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_bfloat8x8 a_frag;
            simdgroup_bfloat8x8 b_frag;
            simdgroup_load(a_frag, As + kk, STRIDE, ulong2(0, 0), false);
            simdgroup_load(b_frag, Bs + (sg * 8u) * STRIDE + kk, STRIDE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* Cs = (threadgroup float*)smem_v;
    simdgroup_store(acc, Cs + sg * 8u, BN, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < BM * BN; idx += 128u) {
        uint r = idx >> 5;
        uint c = idx & 31u;
        uint gr = m0 + r;
        uint gc = n0 + c;
        if (gr < M && gc < N) {
            output[(ulong)gr * N + gc] = q_bf16_rne(Cs[idx]);
        }
    }
}
kernel void fast_q8_gemm_m16(
    device const bfloat* input [[buffer(0)]],
    device const uchar* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint BM = 16u;
    constexpr uint BN = 32u;
    constexpr uint BK = 64u;
    constexpr uint STRIDE = 72u;

    uint m0 = tg_pos.x * BM;
    uint n0 = tg_pos.y * BN;
    if (m0 >= M || n0 >= N) return;

    threadgroup bfloat4 smem_v[864];
    threadgroup bfloat* As = (threadgroup bfloat*)smem_v;
    threadgroup bfloat* Bs = As + BM * STRIDE;

    uint sg = tid >> 5;
    uint sg_m = sg >> 1;
    uint sg_n = sg & 1u;

    simdgroup_float8x8 acc[2];
    acc[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint ar = tid >> 3;
    uint ac = (tid & 7u) << 3;
    uint wr = tid >> 2;
    uint wc = tid & 3u;
    uint wc_elem = wc << 4;

    uint num_groups = K >> 6;
    uint k_u128 = K >> 4;
    device const uint4* packed4 = (device const uint4*)packed;

    uint rm0 = min(m0 + ar, M - 1u);
    uint rn = min(n0 + wr, N - 1u);

    ulong a_row0 = (ulong)rm0 * K + ac;
    ulong w_row = (ulong)rn * k_u128 + wc;
    ulong sb_row = (ulong)rn * num_groups;

    for (uint g = 0u; g < num_groups; ++g) {
        uint k0 = g << 6;
        device const bfloat4* a_ptr = (device const bfloat4*)(input + a_row0 + k0);
        *(threadgroup bfloat4*)(As + ar * STRIDE + ac + 0u) = a_ptr[0];
        *(threadgroup bfloat4*)(As + ar * STRIDE + ac + 4u) = a_ptr[1];

        uint4 q = packed4[w_row + g * 4u];
        bfloat2 sb = scales_biases[sb_row + g];
        float sv = float(sb.x);
        float bv = float(sb.y);
        threadgroup bfloat* b_dst = Bs + wr * STRIDE + wc_elem;
        *(threadgroup bfloat4*)(b_dst + 0u) = bfloat4(float4(as_type<uchar4>(q.x)) * sv + bv);
        *(threadgroup bfloat4*)(b_dst + 4u) = bfloat4(float4(as_type<uchar4>(q.y)) * sv + bv);
        *(threadgroup bfloat4*)(b_dst + 8u) = bfloat4(float4(as_type<uchar4>(q.z)) * sv + bv);
        *(threadgroup bfloat4*)(b_dst + 12u) = bfloat4(float4(as_type<uchar4>(q.w)) * sv + bv);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_bfloat8x8 a_frag;
            simdgroup_bfloat8x8 b_frag0;
            simdgroup_bfloat8x8 b_frag1;
            simdgroup_load(a_frag, As + (sg_m * 8u) * STRIDE + kk, STRIDE, ulong2(0, 0), false);
            simdgroup_load(b_frag0, Bs + (sg_n * 16u + 0u) * STRIDE + kk, STRIDE, ulong2(0, 0), true);
            simdgroup_load(b_frag1, Bs + (sg_n * 16u + 8u) * STRIDE + kk, STRIDE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0], a_frag, b_frag0, acc[0]);
            simdgroup_multiply_accumulate(acc[1], a_frag, b_frag1, acc[1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* Cs = (threadgroup float*)smem_v;
    simdgroup_store(acc[0], Cs + (sg_m * 8u) * BN + (sg_n * 16u + 0u), BN, ulong2(0, 0), false);
    simdgroup_store(acc[1], Cs + (sg_m * 8u) * BN + (sg_n * 16u + 8u), BN, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < BM * BN; idx += 128u) {
        uint r = idx >> 5;
        uint c = idx & 31u;
        uint gr = m0 + r;
        uint gc = n0 + c;
        if (gr < M && gc < N) {
            output[(ulong)gr * N + gc] = q_bf16_rne(Cs[idx]);
        }
    }
}

// ============================================================================
// 9. Q8 Mixed-Precision BF16->FP32 Tiled GEMM (BM=32, BN=32, BK=64, 4 SGs)
// ============================================================================
kernel void fast_q8_gemm_m32(
    device const bfloat* input [[buffer(0)]],
    device const uchar* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint BM = 32u;
    constexpr uint BN = 32u;
    constexpr uint BK = 64u;
    constexpr uint STRIDE = 72u;

    uint m0 = tg_pos.x * BM;
    uint n0 = tg_pos.y * BN;
    if (m0 >= M || n0 >= N) return;

    threadgroup bfloat4 smem_v[1152];
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

    uint r32 = tid >> 2;
    uint c4 = tid & 3u;
    uint c_elem = c4 << 4;

    uint num_groups = K >> 6;
    uint k_u128 = K >> 4;
    device const uint4* packed4 = (device const uint4*)packed;

    uint rm = min(m0 + r32, M - 1u);
    uint rn = min(n0 + r32, N - 1u);

    ulong a_row = (ulong)rm * K + c_elem;
    ulong w_row = (ulong)rn * k_u128 + c4;
    ulong sb_row = (ulong)rn * num_groups;

    for (uint g = 0u; g < num_groups; ++g) {
        uint k0 = g << 6;
        device const bfloat4* a_ptr = (device const bfloat4*)(input + a_row + k0);
        threadgroup bfloat* a_dst = As + r32 * STRIDE + c_elem;
        *(threadgroup bfloat4*)(a_dst + 0u) = a_ptr[0];
        *(threadgroup bfloat4*)(a_dst + 4u) = a_ptr[1];
        *(threadgroup bfloat4*)(a_dst + 8u) = a_ptr[2];
        *(threadgroup bfloat4*)(a_dst + 12u) = a_ptr[3];

        uint4 q = packed4[w_row + g * 4u];
        bfloat2 sb = scales_biases[sb_row + g];
        float sv = float(sb.x);
        float bv = float(sb.y);
        threadgroup bfloat* b_dst = Bs + r32 * STRIDE + c_elem;
        *(threadgroup bfloat4*)(b_dst + 0u) = bfloat4(float4(as_type<uchar4>(q.x)) * sv + bv);
        *(threadgroup bfloat4*)(b_dst + 4u) = bfloat4(float4(as_type<uchar4>(q.y)) * sv + bv);
        *(threadgroup bfloat4*)(b_dst + 8u) = bfloat4(float4(as_type<uchar4>(q.z)) * sv + bv);
        *(threadgroup bfloat4*)(b_dst + 12u) = bfloat4(float4(as_type<uchar4>(q.w)) * sv + bv);

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
            output[(ulong)gr * N + gc] = q_bf16_rne(Cs[idx]);
        }
    }
}

// ============================================================================
// 10. Q8 Embedding Gather + Scale (1 thread per 4 elements)
// ============================================================================
kernel void fast_q8_embed_gather(
    device const uint* token_ids [[buffer(0)]],
    device const uchar* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    constant uint& total_quads [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_quads) return;
    uint quads_per_row = width >> 2;
    uint num_groups = width >> 6;
    uint row = id / quads_per_row;
    uint col4 = id % quads_per_row;
    uint g = col4 >> 4;
    uint tok = token_ids[row];

    device const uchar4* packed4 = (device const uchar4*)packed;
    uchar4 q = packed4[(ulong)tok * quads_per_row + col4];
    bfloat2 sb = scales_biases[(ulong)tok * num_groups + g];
    float sv = float(sb.x);
    float bv = float(sb.y);

    ulong dst = (ulong)row * width + (ulong)col4 * 4u;
    output[dst + 0u] = q_bf16_rne((float(q.x) * sv + bv) * scale);
    output[dst + 1u] = q_bf16_rne((float(q.y) * sv + bv) * scale);
    output[dst + 2u] = q_bf16_rne((float(q.z) * sv + bv) * scale);
    output[dst + 3u] = q_bf16_rne((float(q.w) * sv + bv) * scale);
}

// ============================================================================
// 11. Fused M=1 Q4/Q8 Decode Kernels:
//     - dual_gemv_m1: computes K & V (two equal-shape projections) sharing x
//     - gate_up_geglu_m1: computes Gate + Up + exact table GeGLU in one pass
//     - linear_geglu_m1: computes PleGate + exact table GeGLU against up_in
// ============================================================================
kernel void fast_q4_dual_gemv_m1(
    device const bfloat* input [[buffer(0)]],
    device const uint* packed_a [[buffer(1)]],
    device const bfloat2* sb_a [[buffer(2)]],
    device const uint* packed_b [[buffer(3)]],
    device const bfloat2* sb_b [[buffer(4)]],
    device ushort* out_a [[buffer(5)]],
    device ushort* out_b [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& K [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n = tg_id * 2u + sg;
    if (n >= N) return;

    uint num_groups = K >> 6;
    uint k_u64 = K >> 4;
    uint sub_g = lane >> 2;
    uint sub_l = lane & 3u;

    device const uint2* wa = (device const uint2*)packed_a + (ulong)n * k_u64;
    device const uint2* wb = (device const uint2*)packed_b + (ulong)n * k_u64;
    device const bfloat2* sba = sb_a + (ulong)n * num_groups;
    device const bfloat2* sbb = sb_b + (ulong)n * num_groups;
    device const bfloat4* x4 = (device const bfloat4*)input;
    const float4 nib_scale = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    for (uint g0 = 0u; g0 < num_groups; g0 += 8u) {
        uint g = g0 + sub_g;
        if (g < num_groups) {
            uint x_idx = g * 16u + sub_l * 4u;
            float4 r0 = float4(x4[x_idx + 0u]);
            float4 r1 = float4(x4[x_idx + 1u]);
            float4 r2 = float4(x4[x_idx + 2u]);
            float4 r3 = float4(x4[x_idx + 3u]);
            float xsum = ((r0.x + r0.y) + (r0.z + r0.w))
                       + ((r1.x + r1.y) + (r1.z + r1.w))
                       + ((r2.x + r2.y) + (r2.z + r2.w))
                       + ((r3.x + r3.y) + (r3.z + r3.w));
            float4 s0 = r0 * nib_scale;
            float4 s1 = r1 * nib_scale;
            float4 s2 = r2 * nib_scale;
            float4 s3 = r3 * nib_scale;

            uint idx = g * 4u + sub_l;
            bfloat2 pa = sba[g];
            acc_a += q4_dot16(wa[idx], s0, s1, s2, s3) * float(pa.x) + xsum * float(pa.y);
            bfloat2 pb = sbb[g];
            acc_b += q4_dot16(wb[idx], s0, s1, s2, s3) * float(pb.x) + xsum * float(pb.y);
        }
    }

    float oa = simd_sum(acc_a);
    float ob = simd_sum(acc_b);
    if (lane == 0u) {
        out_a[n] = q_bf16_rne(oa);
        out_b[n] = q_bf16_rne(ob);
    }
}

kernel void fast_q4_gate_up_geglu_m1(
    device const bfloat* input [[buffer(0)]],
    device const uint* gate_packed [[buffer(1)]],
    device const bfloat2* gate_sb [[buffer(2)]],
    device const uint* up_packed [[buffer(3)]],
    device const bfloat2* up_sb [[buffer(4)]],
    device const ushort* gelu_table [[buffer(5)]],
    device ushort* output [[buffer(6)]],
    device atomic_uint* error_flag [[buffer(7)]],
    constant uint& N [[buffer(8)]],
    constant uint& K [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n = tg_id * 2u + sg;
    if (n >= N) return;

    uint num_groups = K >> 6;
    uint k_u64 = K >> 4;
    uint sub_g = lane >> 2;
    uint sub_l = lane & 3u;

    device const uint2* wg = (device const uint2*)gate_packed + (ulong)n * k_u64;
    device const uint2* wu = (device const uint2*)up_packed + (ulong)n * k_u64;
    device const bfloat2* sbg = gate_sb + (ulong)n * num_groups;
    device const bfloat2* sbu = up_sb + (ulong)n * num_groups;
    device const bfloat4* x4 = (device const bfloat4*)input;
    const float4 nib_scale = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (uint g0 = 0u; g0 < num_groups; g0 += 8u) {
        uint g = g0 + sub_g;
        if (g < num_groups) {
            uint x_idx = g * 16u + sub_l * 4u;
            float4 r0 = float4(x4[x_idx + 0u]);
            float4 r1 = float4(x4[x_idx + 1u]);
            float4 r2 = float4(x4[x_idx + 2u]);
            float4 r3 = float4(x4[x_idx + 3u]);
            float xsum = ((r0.x + r0.y) + (r0.z + r0.w))
                       + ((r1.x + r1.y) + (r1.z + r1.w))
                       + ((r2.x + r2.y) + (r2.z + r2.w))
                       + ((r3.x + r3.y) + (r3.z + r3.w));
            float4 s0 = r0 * nib_scale;
            float4 s1 = r1 * nib_scale;
            float4 s2 = r2 * nib_scale;
            float4 s3 = r3 * nib_scale;

            uint idx = g * 4u + sub_l;
            bfloat2 pg = sbg[g];
            acc_g += q4_dot16(wg[idx], s0, s1, s2, s3) * float(pg.x) + xsum * float(pg.y);
            bfloat2 pu = sbu[g];
            acc_u += q4_dot16(wu[idx], s0, s1, s2, s3) * float(pu.x) + xsum * float(pu.y);
        }
    }

    float og = simd_sum(acc_g);
    float ou = simd_sum(acc_u);
    if (lane == 0u) {
        output[n] = q_geglu_apply(q_bf16_rne(og), q_bf16_rne(ou), gelu_table, error_flag);
    }
}

kernel void fast_q4_linear_geglu_m1(
    device const bfloat* input [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device const ushort* up_in [[buffer(3)]],
    device const ushort* gelu_table [[buffer(4)]],
    device ushort* output [[buffer(5)]],
    device atomic_uint* error_flag [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& K [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n0 = tg_id * 4u + sg * 2u;
    if (n0 >= N) return;

    bool has_second = (n0 + 1u) < N;
    uint num_groups = K >> 6;
    uint k_u64 = K >> 4;
    uint sub_g = lane >> 2;
    uint sub_l = lane & 3u;

    device const uint2* w_base = (device const uint2*)packed;
    device const uint2* w0 = w_base + (ulong)n0 * k_u64;
    device const uint2* w1 = has_second ? (w0 + k_u64) : w0;
    device const bfloat2* sb0 = scales_biases + (ulong)n0 * num_groups;
    device const bfloat2* sb1 = has_second ? (sb0 + num_groups) : sb0;
    device const bfloat4* x4 = (device const bfloat4*)input;
    const float4 nib_scale = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint g0 = 0u; g0 < num_groups; g0 += 8u) {
        uint g = g0 + sub_g;
        if (g < num_groups) {
            uint x_idx = g * 16u + sub_l * 4u;
            float4 r0 = float4(x4[x_idx + 0u]);
            float4 r1 = float4(x4[x_idx + 1u]);
            float4 r2 = float4(x4[x_idx + 2u]);
            float4 r3 = float4(x4[x_idx + 3u]);
            float xsum = ((r0.x + r0.y) + (r0.z + r0.w))
                       + ((r1.x + r1.y) + (r1.z + r1.w))
                       + ((r2.x + r2.y) + (r2.z + r2.w))
                       + ((r3.x + r3.y) + (r3.z + r3.w));
            float4 s0 = r0 * nib_scale;
            float4 s1 = r1 * nib_scale;
            float4 s2 = r2 * nib_scale;
            float4 s3 = r3 * nib_scale;

            uint idx = g * 4u + sub_l;
            bfloat2 p_sb0 = sb0[g];
            acc0 += q4_dot16(w0[idx], s0, s1, s2, s3) * float(p_sb0.x) + xsum * float(p_sb0.y);
            if (has_second) {
                bfloat2 p_sb1 = sb1[g];
                acc1 += q4_dot16(w1[idx], s0, s1, s2, s3) * float(p_sb1.x) + xsum * float(p_sb1.y);
            }
        }
    }

    float o0 = simd_sum(acc0);
    float o1 = has_second ? simd_sum(acc1) : 0.0f;
    if (lane == 0u) {
        output[n0] = q_geglu_apply(q_bf16_rne(o0), up_in[n0], gelu_table, error_flag);
        if (has_second) {
            output[n0 + 1u] = q_geglu_apply(q_bf16_rne(o1), up_in[n0 + 1u], gelu_table, error_flag);
        }
    }
}

kernel void fast_q8_dual_gemv_m1(
    device const bfloat* input [[buffer(0)]],
    device const uchar* packed_a [[buffer(1)]],
    device const bfloat2* sb_a [[buffer(2)]],
    device const uchar* packed_b [[buffer(3)]],
    device const bfloat2* sb_b [[buffer(4)]],
    device ushort* out_a [[buffer(5)]],
    device ushort* out_b [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& K [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n = tg_id * 2u + sg;
    if (n >= N) return;

    uint num_groups = K >> 6;
    uint k_u128 = K >> 4;
    uint sub_g = lane >> 2;
    uint sub_l = lane & 3u;

    device const uint4* wa = (device const uint4*)packed_a + (ulong)n * k_u128;
    device const uint4* wb = (device const uint4*)packed_b + (ulong)n * k_u128;
    device const bfloat2* sba = sb_a + (ulong)n * num_groups;
    device const bfloat2* sbb = sb_b + (ulong)n * num_groups;
    device const bfloat4* x4 = (device const bfloat4*)input;

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    for (uint g0 = 0u; g0 < num_groups; g0 += 8u) {
        uint g = g0 + sub_g;
        if (g < num_groups) {
            uint x_idx = g * 16u + sub_l * 4u;
            float4 r0 = float4(x4[x_idx + 0u]);
            float4 r1 = float4(x4[x_idx + 1u]);
            float4 r2 = float4(x4[x_idx + 2u]);
            float4 r3 = float4(x4[x_idx + 3u]);
            float xsum = ((r0.x + r0.y) + (r0.z + r0.w))
                       + ((r1.x + r1.y) + (r1.z + r1.w))
                       + ((r2.x + r2.y) + (r2.z + r2.w))
                       + ((r3.x + r3.y) + (r3.z + r3.w));

            uint idx = g * 4u + sub_l;
            bfloat2 pa = sba[g];
            acc_a += q8_dot16(wa[idx], r0, r1, r2, r3) * float(pa.x) + xsum * float(pa.y);
            bfloat2 pb = sbb[g];
            acc_b += q8_dot16(wb[idx], r0, r1, r2, r3) * float(pb.x) + xsum * float(pb.y);
        }
    }

    float oa = simd_sum(acc_a);
    float ob = simd_sum(acc_b);
    if (lane == 0u) {
        out_a[n] = q_bf16_rne(oa);
        out_b[n] = q_bf16_rne(ob);
    }
}

kernel void fast_q8_gate_up_geglu_m1(
    device const bfloat* input [[buffer(0)]],
    device const uchar* gate_packed [[buffer(1)]],
    device const bfloat2* gate_sb [[buffer(2)]],
    device const uchar* up_packed [[buffer(3)]],
    device const bfloat2* up_sb [[buffer(4)]],
    device const ushort* gelu_table [[buffer(5)]],
    device ushort* output [[buffer(6)]],
    device atomic_uint* error_flag [[buffer(7)]],
    constant uint& N [[buffer(8)]],
    constant uint& K [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n = tg_id * 2u + sg;
    if (n >= N) return;

    uint num_groups = K >> 6;
    uint k_u128 = K >> 4;
    uint sub_g = lane >> 2;
    uint sub_l = lane & 3u;

    device const uint4* wg = (device const uint4*)gate_packed + (ulong)n * k_u128;
    device const uint4* wu = (device const uint4*)up_packed + (ulong)n * k_u128;
    device const bfloat2* sbg = gate_sb + (ulong)n * num_groups;
    device const bfloat2* sbu = up_sb + (ulong)n * num_groups;
    device const bfloat4* x4 = (device const bfloat4*)input;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (uint g0 = 0u; g0 < num_groups; g0 += 8u) {
        uint g = g0 + sub_g;
        if (g < num_groups) {
            uint x_idx = g * 16u + sub_l * 4u;
            float4 r0 = float4(x4[x_idx + 0u]);
            float4 r1 = float4(x4[x_idx + 1u]);
            float4 r2 = float4(x4[x_idx + 2u]);
            float4 r3 = float4(x4[x_idx + 3u]);
            float xsum = ((r0.x + r0.y) + (r0.z + r0.w))
                       + ((r1.x + r1.y) + (r1.z + r1.w))
                       + ((r2.x + r2.y) + (r2.z + r2.w))
                       + ((r3.x + r3.y) + (r3.z + r3.w));

            uint idx = g * 4u + sub_l;
            bfloat2 pg = sbg[g];
            acc_g += q8_dot16(wg[idx], r0, r1, r2, r3) * float(pg.x) + xsum * float(pg.y);
            bfloat2 pu = sbu[g];
            acc_u += q8_dot16(wu[idx], r0, r1, r2, r3) * float(pu.x) + xsum * float(pu.y);
        }
    }

    float og = simd_sum(acc_g);
    float ou = simd_sum(acc_u);
    if (lane == 0u) {
        output[n] = q_geglu_apply(q_bf16_rne(og), q_bf16_rne(ou), gelu_table, error_flag);
    }
}

kernel void fast_q8_linear_geglu_m1(
    device const bfloat* input [[buffer(0)]],
    device const uchar* packed [[buffer(1)]],
    device const bfloat2* scales_biases [[buffer(2)]],
    device const ushort* up_in [[buffer(3)]],
    device const ushort* gelu_table [[buffer(4)]],
    device ushort* output [[buffer(5)]],
    device atomic_uint* error_flag [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& K [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n0 = tg_id * 4u + sg * 2u;
    if (n0 >= N) return;

    bool has_second = (n0 + 1u) < N;
    uint num_groups = K >> 6;
    uint k_u128 = K >> 4;
    uint sub_g = lane >> 2;
    uint sub_l = lane & 3u;

    device const uint4* w_base = (device const uint4*)packed;
    device const uint4* w0 = w_base + (ulong)n0 * k_u128;
    device const uint4* w1 = has_second ? (w0 + k_u128) : w0;
    device const bfloat2* sb0 = scales_biases + (ulong)n0 * num_groups;
    device const bfloat2* sb1 = has_second ? (sb0 + num_groups) : sb0;
    device const bfloat4* x4 = (device const bfloat4*)input;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint g0 = 0u; g0 < num_groups; g0 += 8u) {
        uint g = g0 + sub_g;
        if (g < num_groups) {
            uint x_idx = g * 16u + sub_l * 4u;
            float4 r0 = float4(x4[x_idx + 0u]);
            float4 r1 = float4(x4[x_idx + 1u]);
            float4 r2 = float4(x4[x_idx + 2u]);
            float4 r3 = float4(x4[x_idx + 3u]);
            float xsum = ((r0.x + r0.y) + (r0.z + r0.w))
                       + ((r1.x + r1.y) + (r1.z + r1.w))
                       + ((r2.x + r2.y) + (r2.z + r2.w))
                       + ((r3.x + r3.y) + (r3.z + r3.w));

            uint idx = g * 4u + sub_l;
            bfloat2 p_sb0 = sb0[g];
            acc0 += q8_dot16(w0[idx], r0, r1, r2, r3) * float(p_sb0.x) + xsum * float(p_sb0.y);
            if (has_second) {
                bfloat2 p_sb1 = sb1[g];
                acc1 += q8_dot16(w1[idx], r0, r1, r2, r3) * float(p_sb1.x) + xsum * float(p_sb1.y);
            }
        }
    }

    float o0 = simd_sum(acc0);
    float o1 = has_second ? simd_sum(acc1) : 0.0f;
    if (lane == 0u) {
        output[n0] = q_geglu_apply(q_bf16_rne(o0), up_in[n0], gelu_table, error_flag);
        if (has_second) {
            output[n0 + 1u] = q_geglu_apply(q_bf16_rne(o1), up_in[n0 + 1u], gelu_table, error_flag);
        }
    }
}
