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
