// Elementwise Gemma layer math. Each output is one BF16 round-to-nearest-even;
// soft cap deliberately rounds its divide, tanh and multiply steps separately,
// matching the host decoder's stored policy.
extern "C" __global__ void bf16_add(
    const unsigned short* left,
    const unsigned short* right,
    unsigned short* output,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = bf16_rne(bf16_load(left[i]) + bf16_load(right[i]));
}

extern "C" __global__ void bf16_multiply(
    const unsigned short* left,
    const unsigned short* right,
    unsigned short* output,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = bf16_rne(bf16_load(left[i]) * bf16_load(right[i]));
}

extern "C" __global__ void bf16_scale(
    const unsigned short* input,
    float factor,
    unsigned short* output,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = bf16_rne(bf16_load(input[i]) * factor);
}

extern "C" __global__ void bf16_gelu_tanh(
    const unsigned short* input,
    unsigned short* output,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float x = bf16_load(input[i]);
    float inner = 0.7978846f * x * (1.0f + 0.044715f * x * x);
    output[i] = bf16_rne(0.5f * x * (1.0f + tanhf(inner)));
}

extern "C" __global__ void bf16_softcap(
    const unsigned short* input,
    float cap,
    unsigned short* output,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned short divided = bf16_rne(bf16_load(input[i]) / cap);
    unsigned short hyperbolic = bf16_rne(tanhf(bf16_load(divided)));
    output[i] = bf16_rne(bf16_load(hyperbolic) * cap);
}

extern "C" __global__ void bf16_ple_combine(
    const unsigned short* context,
    const unsigned short* token,
    unsigned short* output,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned short sum = bf16_rne(bf16_load(context[i]) + bf16_load(token[i]));
    output[i] = bf16_rne(bf16_load(sum) * 0.70710677f);
}
