// Exact BF16 product encodings, avoiding FP32 subnormal flushing on the GPU.
// Two BF16 significands have at most 16 product bits, so FP32 multiplication is
// exact down through the smallest nonzero BF16 rounding midpoint (2^-134).
// Smaller products round to BF16 zero even if FP32 first rounds to that midpoint.
// Direct integer RNE therefore preserves the CPU's FP32-product/BF16-RNE policy.
static inline uint rne_shift(uint value, uint shift) {
    if (shift == 0) return value;
    if (shift >= 32) return 0;
    uint truncated = value >> shift;
    uint remainder = value & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    return truncated + uint(remainder > halfway || (remainder == halfway && (truncated & 1u)));
}

static inline ushort bf16_product_rne(ushort a, ushort b) {
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
    if (exponent <= 0) return ushort(sign | rne_shift(product, uint(-power - 133)));
    uint rounded = rne_shift(product, uint(top - 7));
    if (rounded == 256u) { rounded = 128u; ++exponent; }
    if (exponent >= 255) return ushort(sign | 0x7f80u);
    return ushort(sign | (uint(exponent) << 7) | (rounded - 128u));
}

kernel void bf16_lookup_product(
    device const ushort* gate [[buffer(0)]],
    device const ushort* up [[buffer(1)]],
    device const ushort* table [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    device atomic_uint* invalid_gate [[buffer(4)]],
    constant uint& total [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total) return;
    ushort bits = gate[id];
    if ((bits & 0x7f80u) == 0x7f80u) {
        atomic_fetch_or_explicit(invalid_gate, 1u, memory_order_relaxed);
        output[id] = 0;
        return;
    }
    output[id] = bf16_product_rne(table[bits], up[id]);
}
