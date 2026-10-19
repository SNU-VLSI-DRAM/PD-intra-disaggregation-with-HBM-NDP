#include "config/ModelZoo.h"

#include <stdexcept>

namespace {

// name, ndec, hdim, num_heads, dhead, ff_scale, gqa_size, num_experts, top_k
struct ModelEntry {
    const char* name;
    int ndec;
    int hdim;
    int num_heads;
    int dhead;
    float ff_scale;
    int gqa_size;
    int num_experts;
    int top_k;
};

constexpr ModelEntry kModels[] = {
    {"GPT-175B",      96, 12288,  96, 128, 4.0f,      1,   1, 1},
    {"GPT-89B",       48, 12288,  96, 128, 4.0f,      1,   1, 1},
    {"GPT-13B",       40,  5120,  40, 128, 4.0f,      1,   1, 1},
    {"LLAMA-7B",      32,  4096,  32, 128, 8.0f/3,    1,   1, 1},
    {"LLAMA-7B-test", 32,  1024,   8, 128, 2.0f/3,    1,   1, 1},
    {"LLAMA-65B",     80,  8192,  64, 128, 8.0f/3,    1,   1, 1},
    {"MT-76B",        60, 10240,  40, 128, 4.0f,      1,   1, 1},
    {"MT-146B",       80, 12288,  80, 128, 4.0f,      1,   1, 1},
    {"MT-310B",       96, 16384, 128, 128, 4.0f,      1,   1, 1},
    {"MT-530B",      105, 20480, 128, 160, 4.0f,      1,   1, 1},
    {"MT-1008B",     128, 25600, 160, 160, 4.0f,      1,   1, 1},
    {"OPT-66B",       64,  9216,  72, 128, 4.0f,      1,   1, 1},
    {"LLAMA2-70B",    80,  8192,  64, 128, 8.0f/3,    8,   1, 1},
    // Qwen-235B is a MoE model: 128 experts, 8 selected per token.
    {"Qwen-235B",     94,  4096,  64, 128, 3.0f,     16, 128, 8},
};

} // namespace

ModelConfig make_model_config(const std::string& name, const std::string& dtype) {
    for (const auto& m : kModels) {
        if (name == m.name) {
            return ModelConfig{m.name, m.ndec, m.hdim, m.num_heads, m.dhead,
                               m.ff_scale, m.gqa_size, m.num_experts, m.top_k, dtype};
        }
    }
    throw std::runtime_error("Unknown model name: " + name);
}

std::vector<std::string> known_model_names() {
    std::vector<std::string> names;
    for (const auto& m : kModels) names.emplace_back(m.name);
    return names;
}
