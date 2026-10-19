extern "C" __global__ void bf16_rms(
    const unsigned short* input, const unsigned short* scale,
    unsigned short* output, unsigned int width, float epsilon
) {
    __shared__ float sums[256];
    unsigned long long base = (unsigned long long)blockIdx.x * width;
    float energy = 0.0f;
    for (unsigned int j = threadIdx.x; j < width; j += 256) {
        float value = bf16_load(input[base + j]);
        energy = __fadd_rn(energy, __fmul_rn(value, value));
    }
    sums[threadIdx.x] = energy;
    __syncthreads();
    for (unsigned int stride = 128; stride != 0; stride /= 2) {
        if (threadIdx.x < stride)
            sums[threadIdx.x] = __fadd_rn(sums[threadIdx.x], sums[threadIdx.x + stride]);
        __syncthreads();
    }
    float variance = __fdiv_rn(sums[0], (float)width);
    float inverse = powf(__fadd_rn(variance, epsilon), -0.5f);
    for (unsigned int j = threadIdx.x; j < width; j += 256) {
        float value = __fmul_rn(bf16_load(input[base + j]), inverse);
        if (scale != 0) value = __fmul_rn(value, bf16_load(scale[j]));
        output[base + j] = bf16_rne(value);
    }
}
