// No cuda_bf16.h dependency: preserve native encodings at the storage boundary.
__device__ float bf16_load(unsigned short value) {
    return __uint_as_float((unsigned int)value << 16);
}
__device__ unsigned short bf16_rne(float value) {
    unsigned int bits = __float_as_uint(value);
    // Preserve NaN classification, even if rounding its payload would wrap to Inf.
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
    return (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
