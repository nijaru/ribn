// Native rotary factors: FP32 phase and trig on device, BF16 duplicate halves.
// Pinned eager path: phase = float(position) * inverse_frequency[j];
// cosine/sine are evaluated in FP32, scaling is exactly 1.0, then cast to BF16.
extern "C" __global__ void bf16_rotary_factors(
    const float* inverse_frequencies, const unsigned int* positions,
    unsigned short* cosine, unsigned short* sine,
    unsigned int rows, unsigned int half
) {
    unsigned long long total = (unsigned long long)rows * half;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int j = (unsigned int)(i % half);
    unsigned long long row = i / half;
    float phase = __fmul_rn(__uint2float_rn(positions[row]), inverse_frequencies[j]);
    unsigned int width = half * 2u;
    unsigned long long base = row * width + j;
    unsigned short c = bf16_rne(cosf(phase));
    unsigned short s = bf16_rne(sinf(phase));
    cosine[base] = c;
    cosine[base + half] = c;
    sine[base] = s;
    sine[base + half] = s;
}
