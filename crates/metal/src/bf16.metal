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
    for (uint i = 0; i < inner; ++i) {
        sum += bf16_load(input[row * inner + i]) * bf16_load(weight[column * inner + i]);
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
kernel void bf16_attention(
    device const ushort* query [[buffer(0)]],
    device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]],
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
    uint position = id / heads;
    uint head = id % heads;
    uint groups = heads / kv_heads;
    uint kv_head = head / groups;
    uint absolute = query_start + position;
    uint last = min(absolute - key_start, key_sequence - 1);
    uint first = 0;
    if (window != 0) {
        uint start = absolute >= window - 1 ? absolute - (window - 1) : 0;
        first = start > key_start ? start - key_start : 0;
        first = min(first, last);
    }
    uint query_base = position * heads * width + head * width;
    uint key_stride = kv_heads * width;
    float maximum = -INFINITY;
    for (uint j = first; j <= last; ++j) {
        uint key_base = j * key_stride + kv_head * width;
        float sum = 0.0f;
        for (uint c = 0; c < width; ++c) {
            sum += bf16_load(query[query_base + c]) * bf16_load(keys[key_base + c]);
        }
        maximum = max(maximum, bf16_load(bf16_rne(sum)));
    }
    float denominator = 0.0f;
    for (uint j = first; j <= last; ++j) {
        uint key_base = j * key_stride + kv_head * width;
        float sum = 0.0f;
        for (uint c = 0; c < width; ++c) {
            sum += bf16_load(query[query_base + c]) * bf16_load(keys[key_base + c]);
        }
        denominator += exp(bf16_load(bf16_rne(sum)) - maximum);
    }
    for (uint c = 0; c < width; ++c) {
        float accumulator = 0.0f;
        for (uint j = first; j <= last; ++j) {
            uint key_base = j * key_stride + kv_head * width;
            float sum = 0.0f;
            for (uint i = 0; i < width; ++i) {
                sum += bf16_load(query[query_base + i]) * bf16_load(keys[key_base + i]);
            }
            float weight = bf16_load(
                bf16_rne(exp(bf16_load(bf16_rne(sum)) - maximum) / denominator));
            accumulator += weight * bf16_load(values[key_base + c]);
        }
        output[query_base + c] = bf16_rne(accumulator);
    }
}
