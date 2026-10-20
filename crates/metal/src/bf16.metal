#include <metal_stdlib>
using namespace metal;

// No bfloat16 header dependency: preserve native encodings at the storage boundary.
static inline float bf16_load(ushort value) {
    return as_type<float>((uint)value << 16);
}

// NaN keeps a canonical encoding rather than rounding its payload to infinity.
static inline ushort bf16_rne(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
    return (ushort)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

kernel void bf16_linear(
    device const ushort* input [[buffer(0)]],
    device const ushort* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& total [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total) return;
    uint row = id / columns;
    uint column = id % columns;
    float sum = 0.0f;
    uint i = 0;
    if ((inner & 3u) == 0) {
        device const ushort4* input4 =
            (device const ushort4*)(input + row * inner);
        device const ushort4* weight4 =
            (device const ushort4*)(weight + column * inner);
        for (; i < inner; i += 4) {
            ushort4 a = input4[i / 4];
            ushort4 b = weight4[i / 4];
            sum += bf16_load(a.x) * bf16_load(b.x);
            sum += bf16_load(a.y) * bf16_load(b.y);
            sum += bf16_load(a.z) * bf16_load(b.z);
            sum += bf16_load(a.w) * bf16_load(b.w);
        }
    } else {
        for (; i < inner; ++i) {
            sum += bf16_load(input[row * inner + i]) * bf16_load(weight[column * inner + i]);
        }
    }
    output[id] = bf16_rne(sum);
}

kernel void bf16_rms(
    device const ushort* input [[buffer(0)]],
    device const ushort* scale [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    constant uint& has_scale [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    threadgroup float partial[256];
    float energy = 0.0f;
    for (uint j = lane; j < width; j += 256) {
        float value = bf16_load(input[row * width + j]);
        energy += value * value;
    }
    partial[lane] = energy;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = partial[0] / (float)width;
    float inverse = pow(mean + epsilon, -0.5f);
    for (uint j = lane; j < width; j += 256) {
        float value = bf16_load(input[row * width + j]) * inverse;
        if (has_scale != 0) value *= bf16_load(scale[j]);
        output[row * width + j] = bf16_rne(value);
    }
}

// Three-pass causal/sliding attention: score max, exponential sum, weighted
// values. Scores and weights round to BF16 exactly where the pinned model does;
// the output accumulates in FP32 and rounds once. No softcap is applied here.
kernel void bf16_attention_scores(
    device const ushort* query [[buffer(0)]],
    device const ushort* keys [[buffer(1)]],
    device ushort* scores [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& kv_heads [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& key_sequence [[buffer(6)]],
    constant uint& query_start [[buffer(7)]],
    constant uint& key_start [[buffer(8)]],
    constant uint& window [[buffer(9)]],
    constant uint& total [[buffer(10)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total) return;
    uint row = id / key_sequence;
    uint j = id % key_sequence;
    uint position = row / heads;
    uint head = row % heads;
    uint kv_head = head / (heads / kv_heads);
    uint absolute = query_start + position;
    uint last = min(absolute - key_start, key_sequence - 1);
    uint first = 0;
    if (window != 0) {
        uint start = absolute >= window - 1 ? absolute - (window - 1) : 0;
        first = start > key_start ? start - key_start : 0;
        first = min(first, last);
    }
    ushort score = 0;
    if (j >= first && j <= last) {
        uint query_base = row * width;
        uint key_base = j * (kv_heads * width) + kv_head * width;
        float sum = 0.0f;
        for (uint c = 0; c < width; ++c) {
            sum += bf16_load(query[query_base + c]) * bf16_load(keys[key_base + c]);
        }
        score = bf16_rne(sum);
    }
    scores[id] = score;
}

kernel void bf16_attention_reduce(
    device const ushort* scores [[buffer(0)]],
    device float2* summary [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& key_sequence [[buffer(3)]],
    constant uint& query_start [[buffer(4)]],
    constant uint& key_start [[buffer(5)]],
    constant uint& window [[buffer(6)]],
    constant uint& total [[buffer(7)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= total) return;
    uint position = row / heads;
    uint absolute = query_start + position;
    uint last = min(absolute - key_start, key_sequence - 1);
    uint first = 0;
    if (window != 0) {
        uint start = absolute >= window - 1 ? absolute - (window - 1) : 0;
        first = start > key_start ? start - key_start : 0;
        first = min(first, last);
    }
    float maximum = -INFINITY;
    for (uint j = first; j <= last; ++j) {
        maximum = max(maximum, bf16_load(scores[row * key_sequence + j]));
    }
    float denominator = 0.0f;
    for (uint j = first; j <= last; ++j) {
        denominator += exp(bf16_load(scores[row * key_sequence + j]) - maximum);
    }
    summary[row] = float2(maximum, denominator);
}

kernel void bf16_attention_values(
    device const ushort* values [[buffer(0)]],
    device const ushort* scores [[buffer(1)]],
    device const float2* summary [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& kv_heads [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& key_sequence [[buffer(7)]],
    constant uint& query_start [[buffer(8)]],
    constant uint& key_start [[buffer(9)]],
    constant uint& window [[buffer(10)]],
    constant uint& total [[buffer(11)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total) return;
    uint row = id / width;
    uint c = id % width;
    uint position = row / heads;
    uint head = row % heads;
    uint kv_head = head / (heads / kv_heads);
    uint absolute = query_start + position;
    uint last = min(absolute - key_start, key_sequence - 1);
    uint first = 0;
    if (window != 0) {
        uint start = absolute >= window - 1 ? absolute - (window - 1) : 0;
        first = start > key_start ? start - key_start : 0;
        first = min(first, last);
    }
    float2 row_summary = summary[row];
    float accumulator = 0.0f;
    for (uint j = first; j <= last; ++j) {
        float score = bf16_load(scores[row * key_sequence + j]);
        float weight = bf16_load(bf16_rne(exp(score - row_summary.x) / row_summary.y));
        uint key_base = j * (kv_heads * width) + kv_head * width;
        accumulator += weight * bf16_load(values[key_base + c]);
    }
    output[id] = bf16_rne(accumulator);
}

// Split-half rotation with prepared BF16 factors: each product rounds to BF16
// before the FP32 addition, matching eager arithmetic.
kernel void bf16_rotary_apply(
    device const ushort* input [[buffer(0)]],
    device const ushort* cosine [[buffer(1)]],
    device const ushort* sine [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& pairs [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= pairs) return;
    uint half_width = width / 2;
    ulong head = id / half_width;
    uint j = id % half_width;
    ulong row = head / (columns / width);
    ulong x = head * width + j;
    ulong factor = row * width + j;
    float a = bf16_load(input[x]);
    float b = bf16_load(input[x + half_width]);
    float ac = bf16_load(bf16_rne(a * bf16_load(cosine[factor])));
    float bs = bf16_load(bf16_rne(-b * bf16_load(sine[factor])));
    float bc = bf16_load(bf16_rne(b * bf16_load(cosine[factor + half_width])));
    float a_sine = bf16_load(bf16_rne(a * bf16_load(sine[factor + half_width])));
    output[x] = bf16_rne(ac + bs);
    output[x + half_width] = bf16_rne(bc + a_sine);
}

// One elementwise operation per launch: 0 add, 1 multiply, 2 scale, 3 GELU
// (tanh approximation), 4 final soft cap. Unary operations bind `right` to
// `left` and ignore it.
kernel void bf16_elementwise(
    device const ushort* left [[buffer(0)]],
    device const ushort* right [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& operation [[buffer(3)]],
    constant float& scalar [[buffer(4)]],
    constant uint& total [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total) return;
    float value = bf16_load(left[id]);
    float result;
    if (operation == 0) {
        result = value + bf16_load(right[id]);
    } else if (operation == 1) {
        result = value * bf16_load(right[id]);
    } else if (operation == 2) {
        result = value * scalar;
    } else if (operation == 3) {
        float inner = 0.7978846f * value * (1.0f + 0.0447150f * value * value);
        result = 0.5f * value * (1.0f + tanh(inner));
    } else {
        float divided = bf16_load(bf16_rne(value / scalar));
        float hyperbolic = bf16_load(bf16_rne(tanh(divided)));
        result = hyperbolic * scalar;
    }
    output[id] = bf16_rne(result);
}
