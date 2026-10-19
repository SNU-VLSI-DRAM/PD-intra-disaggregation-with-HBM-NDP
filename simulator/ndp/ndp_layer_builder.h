#pragma once
#include "common/Common.h"
#include "config/ModelConfig.h"
#include "config/NDPConfig.h"
#include <string>

class NDPLayerBuilder {
public:
    std::string name;
    int ndec;
    int num_heads;
    int hdim;
    float ff_scale;
    // MoE
    int num_experts = 1;
    int top_k = 1;
    int tp;
    int pp = 1;  // pipeline parallel degree
    int sp = 1;  // sequence-level parallelism
    int dhead;
    int gqa_size = 1;
    int dtype_bytes; // For simplicity, use int. You can define enum if needed.
    NDPConfig ndp_config;  // NDP configuration

    // Duplex configuration: the NDP runs the attention GEMMs (QK, softmax, SV)
    // and the MoE experts; the GPU runs everything else against an ordinary
    // DRAM model. When set, build() keeps only those layers plus the
    // sequence-parallel exchange that attention needs. QKV generation, the
    // output projection, the layernorms, the dense FFN, the MoE gating GEMM,
    // the all-to-all that distributes its result and every tensor-parallel
    // all-reduce are left to the GPU and never appear in the NDP schedule.
    bool attn_moe_only = false;

    NDPLayerBuilder(const std::string& name_, int ndec_, int num_heads_, int hdim_, int ff_scale_, int dtype_bytes_, int tp_ = 8, int pp_ = 1)
        : name(name_), ndec(ndec_), num_heads(num_heads_), hdim(hdim_), ff_scale(ff_scale_), dtype_bytes(dtype_bytes_), tp(tp_), pp(pp_) {
        dhead = hdim / num_heads;
    }
    
    // Constructor with ModelConfig and NDPConfig (much cleaner!)
    NDPLayerBuilder(const ModelConfig& config, const NDPConfig& ndp_cfg, int dtype_bytes_, int tp_ = 8, int pp_ = 1, int sp_ = 1);

    // Build layer sequence (sum_decoder + gen_decoder)
    std::vector<Layer> build(int batch, int lin, int lout, int sp = 1) const;
};
