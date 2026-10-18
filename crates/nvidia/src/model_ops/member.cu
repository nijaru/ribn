// Wire format: see member_descriptor.rs. All offsets are u64 words.
__device__ __forceinline__ const unsigned long long* member_record(
    const unsigned long long* descriptor, int member
) {
    return descriptor + 3 + (long long)member * (4 + 2 * (descriptor[1] + descriptor[2]));
}

extern "C" __global__ void gdn_conv_members(
    const unsigned long long* descriptor, int slot,
    const float* input, const float* conv_weight, float* output, int channels
) {
    const int member = blockIdx.y;
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const unsigned long long* record = member_record(descriptor, member);
    float* history = (float*)record[4 + 2 * descriptor[1] + 2 * slot + 1];
    const long long index = (long long)member * channels + channel;
    float* h = history + channel * 3;
    const float* w = conv_weight + channel * 4;
    const float acc = h[0] * w[0] + h[1] * w[1] + h[2] * w[2] + input[index] * w[3];
    h[0] = h[1];
    h[1] = h[2];
    h[2] = input[index];
    output[index] = acc / (1.0f + expf(-acc));
}

// Copies only: normalization remains in the existing l2_norm_heads_batch kernel.
extern "C" __global__ void gdn_gather_qk_members(
    const float* conv, float* q, float* k, int channels, int qk_width
) {
    const int member = blockIdx.y;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= qk_width) return;
    const long long src = (long long)member * channels + index;
    const long long dst = (long long)member * qk_width + index;
    q[dst] = conv[src];
    k[dst] = conv[src + qk_width];
}

extern "C" __global__ void gdn_state_members(
    const unsigned long long* descriptor, int slot,
    const float* q_normed, const float* k_normed, const float* conv_activated,
    const float* decay, const float* beta, float* output,
    int v_heads, int k_heads, int head_dim, int v_offset
) {
    const int member = blockIdx.y;
    const unsigned long long* record = member_record(descriptor, member);
    float* matrix = (float*)record[4 + 2 * descriptor[1] + 2 * slot];
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long long)v_heads * head_dim) return;
    const int v_head = (int)(index / head_dim);
    const int col = (int)(index % head_dim);
    const int k_head = v_head % k_heads;
    const long long qkv_row = (long long)member * (v_offset + v_heads * head_dim);
    const long long qk_row = (long long)member * k_heads * head_dim;
    const long long gate_row = (long long)member * v_heads;
    const long long out_row = (long long)member * v_heads * head_dim;
    float* state = matrix + (long long)v_head * head_dim * head_dim;
    const float* q = q_normed + qk_row + (long long)k_head * head_dim;
    const float* k = k_normed + qk_row + (long long)k_head * head_dim;
    const float* v = conv_activated + qkv_row + (long long)v_offset + (long long)v_head * head_dim;
    float sk = 0.0f;
    for (int row = 0; row < head_dim; ++row) {
        float* element = state + row * head_dim + col;
        const float decayed = *element * decay[gate_row + v_head];
        *element = decayed;
        sk += decayed * k[row];
    }
    const float d = (v[col] - sk) * beta[gate_row + v_head];
    float out = 0.0f;
    for (int row = 0; row < head_dim; ++row) {
        float* element = state + row * head_dim + col;
        const float updated = *element + k[row] * d;
        *element = updated;
        out += updated * q[row];
    }
    output[out_row + (long long)v_head * head_dim + col] = out * rsqrtf((float)head_dim);
}

extern "C" __global__ void kv_append_members(
    const unsigned long long* descriptor, int slot,
    const float* keys, const float* values, int kv_heads, int head_dim
) {
    const int member = blockIdx.y;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = kv_heads * head_dim;
    if (index >= total) return;
    const unsigned long long* record = member_record(descriptor, member);
    const int* table = (const int*)record[0];
    const int token = kv_token<true>(table, (int)record[2], (int)record[1]);
    unsigned short* cache_keys = (unsigned short*)record[4 + 2 * slot];
    unsigned short* cache_values = (unsigned short*)record[5 + 2 * slot];
    const long long offset = (long long)token * total + index;
    const long long src = (long long)member * total + index;
    cache_keys[offset] = f32_to_f16_bits(keys[src]);
    cache_values[offset] = f32_to_f16_bits(values[src]);
}

extern "C" __global__ void attn_members(
    const unsigned long long* descriptor, int slot,
    const float* q, const float* gate, float* scores, float* output,
    int stride, int q_heads, int kv_heads, int head_dim
) {
    const int member = blockIdx.y;
    const unsigned long long* record = member_record(descriptor, member);
    const int* table = (const int*)record[0];
    const unsigned short* keys = (const unsigned short*)record[4 + 2 * slot];
    const unsigned short* values = (const unsigned short*)record[5 + 2 * slot];
    const long long q_row = (long long)member * q_heads * head_dim;
    // Identical warp body to the per-member path, with disjoint score rows.
    attn_score_gqa_body<true>(q + q_row, keys, values, table, (int)record[2],
        gate + 2 * q_row, scores + (long long)member * q_heads * stride,
        output + q_row, (int)record[1] + 1, 1, stride, q_heads, kv_heads, head_dim);
}
