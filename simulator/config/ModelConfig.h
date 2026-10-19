#pragma once
#include <string>

struct ModelConfig {
    std::string name;
    int ndec;
    int hdim;
    int num_heads;
    int dhead;
    float ff_scale;
    int gqa_size;
    int num_experts;
    int top_k;
    std::string dtype;
};