// Eager BF16 arithmetic: round BOTH products before their addition, not one
// fused FP32 rotation followed by a single BF16 conversion.
__device__ float bf16_product(float a, float b) {
    return bf16_load(bf16_rne(__fmul_rn(a, b)));
}
extern "C" __global__ void bf16_rotary_apply(
    const unsigned short* input, const unsigned short* cosine,
    const unsigned short* sine, unsigned short* output,
    unsigned int columns, unsigned int width, unsigned long long pairs
) {
    unsigned long long p = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) return;
    unsigned int half = width / 2;
    unsigned long long head = p / half;
    unsigned int j = (unsigned int)(p % half);
    unsigned long long row = head / (columns / width);
    unsigned long long x = head * width + j;
    unsigned long long factor = row * width + j;
    float a = bf16_load(input[x]);
    float b = bf16_load(input[x + half]);
    float ac = bf16_product(a, bf16_load(cosine[factor]));
    float bs = bf16_product(-b, bf16_load(sine[factor]));
    float bc = bf16_product(b, bf16_load(cosine[factor + half]));
    float as = bf16_product(a, bf16_load(sine[factor + half]));
    output[x] = bf16_rne(__fadd_rn(ac, bs));
    output[x + half] = bf16_rne(__fadd_rn(bc, as));
}
