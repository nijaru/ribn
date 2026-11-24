// Common-history oracle for the pinned fixed-loop Inkling 2 Mini checkpoint.
#include "llama.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::array checkpoints{8, 32, 1023, 1024, 1025, 1032, 1033, 1034, 1035, 1036};
constexpr int vocab_size = 200058;

void replay(llama_model * model, const std::vector<llama_token> & tokens, int width,
            const std::filesystem::path & output) {
    auto params = llama_context_default_params();
    params.n_ctx = 2048;
    params.n_batch = params.n_ubatch = 128;
    params.type_k = params.type_v = GGML_TYPE_F16;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, params), llama_free);
    if (!ctx) throw std::runtime_error("context initialization failed");
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    if (!batch) throw std::runtime_error("batch initialization failed");
    int pos = 0;
    for (int end : checkpoints) {
        while (pos < end) {
            const int count = std::min(width, end - pos);
            llama_batch_ext_clear(batch.get());
            for (int i = 0; i < count; ++i) {
                const llama_pos position = pos + i;
                const int idx = llama_batch_ext_add_token(batch.get(), 0, tokens.at(position));
                if (idx != i || !llama_batch_ext_set_pos(batch.get(), idx, &position) ||
                    !llama_batch_ext_set_output_logits(batch.get(), idx, position == end - 1)) {
                    throw std::runtime_error("batch assembly failed");
                }
            }
            const int status = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (status != 0) throw std::runtime_error("decode failed: " + std::to_string(status));
            pos += count;
        }
        const float * row = llama_get_logits_ith(ctx.get(), -1);
        if (!row || !std::all_of(row, row + vocab_size, [](float x) { return std::isfinite(x); })) {
            throw std::runtime_error("missing or non-finite logits");
        }
        std::ofstream file(output / (std::to_string(width) + "-" + std::to_string(end) + ".f32"), std::ios::binary);
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.write(reinterpret_cast<const char *>(row), sizeof(float) * vocab_size);
        file.close();
        std::cout << "width=" << width << " tokens=" << end << " argmax="
                  << std::max_element(row, row + vocab_size) - row << '\n';
    }
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: replay MODEL.gguf OUTPUT_DIR");
        const std::filesystem::path output = argv[2];
        if (!std::filesystem::create_directory(output)) throw std::runtime_error("output must be a new directory");
        auto params = llama_model_default_params();
        params.n_gpu_layers = -1;
        llama_backend_init();
        if (!llama_supports_gpu_offload()) throw std::runtime_error("GPU offload unavailable");
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], params), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        if (llama_vocab_n_tokens(vocab) != vocab_size) throw std::runtime_error("unexpected vocabulary");
        const std::string text = "In the archive, the blue notebook belongs to Ada. The red one belongs to Lin.\n"
                                 "Who owns the blue notebook? Answer: Ada.\n";
        std::vector<llama_token> piece(text.size());
        const int n = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), piece.data(),
                                     static_cast<int>(piece.size()), false, false);
        if (n <= 0) throw std::runtime_error("tokenization failed");
        std::vector<llama_token> tokens;
        for (int i = 0; i < checkpoints.back(); ++i) tokens.push_back(piece[i % n]);
        std::ofstream history(output / "tokens.txt");
        history.exceptions(std::ios::failbit | std::ios::badbit);
        for (auto token : tokens) history << token << '\n';
        history.close();
        replay(model.get(), tokens, 64, output);
        replay(model.get(), tokens, 127, output);
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
