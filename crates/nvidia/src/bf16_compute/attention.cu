// Causal grouped-query attention with BF16 scores/weights and FP32 softmax.
// Three passes over the same visible key range: scores, softmax weights, values.
// The softmax and value accumulation walk keys in ascending order, matching the
// host decoder; only the exponential implementation is device-specific.
__device__ void attention_range(
    unsigned int position,
    unsigned int key_sequence,
    unsigned int query_start,
    unsigned int key_start,
    int window,
    unsigned int* first,
    unsigned int* last)
{
    unsigned int absolute = query_start + position;
    *last = min(absolute - key_start, key_sequence - 1u);
    if (window > 0) {
        unsigned int offset = (unsigned int)(window - 1);
        unsigned int span = absolute > offset ? absolute - offset : 0u;
        unsigned int start = span > key_start ? span - key_start : 0u;
        *first = min(start, *last);
    } else {
        *first = 0u;
    }
}

extern "C" __global__ void bf16_attention_scores(
    const unsigned short* query,
    const unsigned short* keys,
    unsigned short* scores,
    unsigned int sequence,
    unsigned int key_sequence,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int width,
    unsigned int query_start,
    unsigned int key_start,
    int window,
    unsigned int total)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total) return;
    unsigned int key = id % key_sequence;
    unsigned int head = (id / key_sequence) % heads;
    unsigned int position = id / (key_sequence * heads);
    unsigned int first, last;
    attention_range(position, key_sequence, query_start, key_start, window, &first, &last);
    if (key < first || key > last) return;
    unsigned int groups = heads / kv_heads;
    unsigned int kv_head = head / groups;
    unsigned int query_base = position * heads * width + head * width;
    unsigned int key_base = key * kv_heads * width + kv_head * width;
    float sum = 0.0f;
    for (unsigned int column = 0; column < width; ++column) {
        sum += bf16_load(query[query_base + column]) * bf16_load(keys[key_base + column]);
    }
    scores[id] = bf16_rne(sum);
}

extern "C" __global__ void bf16_attention_softmax(
    unsigned short* scores,
    unsigned int sequence,
    unsigned int key_sequence,
    unsigned int heads,
    unsigned int query_start,
    unsigned int key_start,
    int window)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= sequence * heads) return;
    unsigned int position = id / heads;
    unsigned int first, last;
    attention_range(position, key_sequence, query_start, key_start, window, &first, &last);
    unsigned int base = id * key_sequence;
    float maximum = -3.402823466e38f; // -FLT_MAX; NVRTC has no INFINITY here
    for (unsigned int key = first; key <= last; ++key) {
        float score = bf16_load(scores[base + key]);
        maximum = fmaxf(maximum, score);
    }
    float total = 0.0f;
    for (unsigned int key = first; key <= last; ++key) {
        total += expf(bf16_load(scores[base + key]) - maximum);
    }
    for (unsigned int key = first; key <= last; ++key) {
        float weight = expf(bf16_load(scores[base + key]) - maximum) / total;
        scores[base + key] = bf16_rne(weight);
    }
}

extern "C" __global__ void bf16_attention_values(
    const unsigned short* weights,
    const unsigned short* values,
    unsigned short* output,
    unsigned int sequence,
    unsigned int key_sequence,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int width,
    unsigned int query_start,
    unsigned int key_start,
    int window,
    unsigned int total)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total) return;
    unsigned int column = id % width;
    unsigned int head = (id / width) % heads;
    unsigned int position = id / (width * heads);
    unsigned int first, last;
    attention_range(position, key_sequence, query_start, key_start, window, &first, &last);
    unsigned int groups = heads / kv_heads;
    unsigned int kv_head = head / groups;
    unsigned int weight_base = (position * heads + head) * key_sequence;
    unsigned int value_base = kv_head * width + column;
    float sum = 0.0f;
    for (unsigned int key = first; key <= last; ++key) {
        sum += bf16_load(weights[weight_base + key])
            * bf16_load(values[key * kv_heads * width + value_base]);
    }
    output[position * heads * width + head * width + column] = bf16_rne(sum);
}
