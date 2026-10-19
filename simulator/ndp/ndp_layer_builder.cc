#include "ndp/ndp_layer_builder.h"
#include "config/ModelConfig.h"
#include "config/NDPConfig.h"
#include "ndp/ndp_ctrl.h"
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <sstream>

// Constructor with ModelConfig and NDPConfig
NDPLayerBuilder::NDPLayerBuilder(const ModelConfig& config, const NDPConfig& ndp_cfg, int dtype_bytes_, int tp_, int pp_, int sp_)
    : name(config.name), ndec(config.ndec), num_heads(config.num_heads), 
    hdim(config.hdim), ff_scale(config.ff_scale), gqa_size(config.gqa_size), dtype_bytes(dtype_bytes_),
    num_experts(config.num_experts), top_k(config.top_k), tp(tp_), pp(pp_), sp(sp_), ndp_config(ndp_cfg) {
    dhead = hdim / num_heads;
}

// Helper functions for creating layers
namespace {
    Layer make_gemm_layer(uint32_t batch_size, uint32_t hidden_dim, uint32_t out_ch, uint64_t weight_addr = 0, uint64_t kv_addr = 0, const std::string& layer_type = "GEMM") {
        Layer layer;
        layer.layer_name = layer_type; // "GEMM", "Score", or "Context"
        layer.batch_size = batch_size;
        layer.hidden_dim = hidden_dim;
        layer.out_ch = out_ch;
        
        // Set weight address (use KV address for score/context layers)
        layer.addr_w = (layer_type == "Score" || layer_type == "Context" || layer_type == "Flash_attn") ? kv_addr : weight_addr;
        
        return layer;
    }
    
    Layer make_softmax_layer(uint32_t vector_size) {
        Layer layer;
        layer.layer_name = "Softmax";
        layer.vector_size = vector_size;
        return layer;
    }
    
    Layer make_layernorm_layer(uint32_t vector_size) {
        Layer layer;
        layer.layer_name = "Layernorm";
        layer.vector_size = vector_size;
        return layer;
    }
    
    Layer make_gelu_layer(uint32_t vector_size) {
        Layer layer;
        layer.layer_name = "GELU";
        layer.vector_size = vector_size;
        return layer;
    }

    // NDP-to-NDP exchange inside one GPU package (sequence parallelism).
    // `comm_hops` > 0 charges the intra-package link latency to this layer.
    Layer make_comm_layer(uint32_t vector_size, uint32_t comm_hops = 0) {
        Layer layer;
        layer.layer_name = "Comm";
        layer.vector_size = vector_size;
        layer.comm_hops = comm_hops;
        return layer;
    }

    // GPU-level all-reduce step. `comm_hops` > 0 charges the inter-GPU NVLink
    // latency; a collective emitted as several steps tags only the first one.
    Layer make_allreduce_layer(uint32_t vector_size, uint32_t comm_hops = 0) {
        Layer layer;
        layer.layer_name = "AllReduce";
        layer.vector_size = vector_size;
        layer.comm_hops = comm_hops;
        return layer;
    }

    Layer make_pipelinecomm_layer(uint32_t vector_size, uint32_t comm_hops = 1) {
        Layer layer;
        layer.layer_name = "PipelineComm";
        layer.vector_size = vector_size;
        layer.comm_hops = comm_hops;
        return layer;
    }

    Layer finish_step(uint32_t vector_size) {
        Layer layer;
        layer.layer_name = "FinishStep";
        layer.vector_size = vector_size;
        return layer;
    }
}

std::vector<Layer> NDPLayerBuilder::build(int batch, int lin, int lout, int sp) const {
    std::vector<Layer> layers;
    int dhead = hdim / num_heads;
    // In attention, we use sequence-level parallelism instead of tensor parallelism, so each NDP handles tp/8 heads
    int tp_heads = num_heads / (tp/sp); 
    int tp_hdim = hdim / tp;
    int ff_hdim = ff_scale * tp_hdim;

    const bool is_moe = (name.find("Qwen") != std::string::npos);
    const int moe_num_experts = std::max(1, num_experts);
    const int moe_top_k = std::max(1, top_k);
    int ep = (tp/8);

    // `tp` here counts GPUs * 8 NDPs, so tp/8 is the number of GPUs in the
    // tensor-parallel group, i.e. the rank count of a GPU-level collective.
    const int gpu_ranks = std::max(1, tp / 8);
    // Link traversals charged to ONE GPU-level all-reduce. With the default
    // one-shot model on a flat NVLink5 fabric (NVL72) this is 1: every peer is
    // a single NVSwitch hop away and the switch reduces in the network, so the
    // latency does not scale with the TP degree.
    const uint32_t allreduce_hops =
        collective_latency_hops(ndp_config.collective_algo, gpu_ranks, /*is_allreduce=*/true);
    // Emit the `gpu_ranks` partial steps the existing bandwidth model expects,
    // but charge the NVLink latency to the first step only, so the collective
    // as a whole pays the hop latency exactly once.
    auto push_allreduce = [&](std::vector<Layer>& out, uint32_t vec) {
        if (gpu_ranks <= 1) return;
        for (int i = 0; i < gpu_ranks; ++i) {
            out.push_back(make_allreduce_layer(vec, i == 0 ? allreduce_hops : 0u));
        }
    };

    // Layers the GPU owns under attn_moe_only. Pushing them through these two
    // helpers rather than straight onto `layers` keeps the duplex filter visible
    // where each layer is created. The address arithmetic around the calls runs
    // either way, so the weight and KV pointers stay in step.
    auto push_gpu = [&](const Layer& l) {
        if (!attn_moe_only) layers.push_back(l);
    };
    auto push_gpu_allreduce = [&](uint32_t vec) {
        if (!attn_moe_only) push_allreduce(layers, vec);
    };

    auto build_moe_local_routing = [&](uint64_t tokens, uint64_t seed, bool randomize) {
        const uint64_t local_experts = (moe_num_experts + ep - 1) / ep;
        const uint64_t active_expert_slots = (moe_top_k + ep - 1) / ep;
        const uint64_t local_assignments = tokens * active_expert_slots;
        std::vector<uint64_t> counts(local_experts, 0);

        if (local_experts == 0 || local_assignments == 0) {
            return counts;
        }

        if (!randomize) {
            const uint64_t base = local_assignments / local_experts;
            const uint64_t rem = local_assignments % local_experts;
            for (uint64_t i = 0; i < local_experts; ++i) {
                counts[i] = base + (i < rem ? 1 : 0);
            }
            return counts;
        }

        std::mt19937_64 rng(seed);
        const double alpha = 0.5;
        std::gamma_distribution<double> gamma(alpha, 1.0);
        std::vector<double> weights(local_experts, 0.0);
        double sum = 0.0;
        for (uint64_t i = 0; i < local_experts; ++i) {
            weights[i] = gamma(rng);
            sum += weights[i];
        }

        if (sum <= 0.0) {
            const uint64_t base = local_assignments / local_experts;
            const uint64_t rem = local_assignments % local_experts;
            for (uint64_t i = 0; i < local_experts; ++i) {
                counts[i] = base + (i < rem ? 1 : 0);
            }
            return counts;
        }

        std::discrete_distribution<uint64_t> dist(weights.begin(), weights.end());
        for (uint64_t i = 0; i < local_assignments; ++i) {
            counts[dist(rng)]++;
        }
        return counts;
    };


    // Print the computed values
    std::cout << "dhead: " << dhead << std::endl;
    std::cout << "tp_heads: " << tp_heads << std::endl;
    std::cout << "tp_hdim: " << tp_hdim << std::endl;
    std::cout << "ff_hdim: " << ff_hdim << std::endl;
    std::cout << "hdim: " << hdim << std::endl;
    
    uint64_t current_weight_addr;
    uint64_t current_kv_addr;

    // Generation (gen_decoder) - NDP only handles generation
    for (int stage = 1; stage <= lout; ++stage) {
        uint64_t moe_block_index = 0;

        current_weight_addr = ndp_config.weight_addr; 
        current_kv_addr = ndp_config.kv_addr; 

        
        // Attention layers
        if (tp_heads < gqa_size) {
            auto qkv_layer = make_gemm_layer(batch, hdim, tp_hdim + 2 * int(2*tp_hdim/gqa_size), current_weight_addr);
            push_gpu(qkv_layer); // QKV (GPU-side under attn_moe_only)
            current_weight_addr += hdim * (tp_hdim + 2 * int(2*tp_hdim/gqa_size)) * 2; // 2 bytes per weight

            // Score: lane width optimization (hidden_dim direction) with batch concat optimization
            // Use lane_width from NDP config, dhead=128 -> can concat multiple heads
            int score_kheads_per_concat = std::min(ndp_config.lane_width / dhead, 2 * tp_heads / gqa_size);
            int score_concat_iterations = (2 * tp_heads / gqa_size + score_kheads_per_concat - 1) / score_kheads_per_concat;

            // If head count is low, consider batch concat to better utilize resources
            int max_batch_per_concat = int(ndp_config.lane_width / (score_kheads_per_concat * dhead)); // Maximum batches that can be concatenated
            if (stage == 1) std::cout << "Score concat: heads per concat=" << score_kheads_per_concat << ", max batch per concat=" << max_batch_per_concat << std::endl;
            if (score_kheads_per_concat <= int((ndp_config.lane_width / dhead) / 2)) { // If using less than 25% of lane width
                int batch_per_concat = std::min(max_batch_per_concat, batch);
                int batch_concat_iterations = (batch + batch_per_concat - 1) / batch_per_concat;
                for (int batch_iter = 0; batch_iter < batch_concat_iterations; ++batch_iter) {
                    int current_batch = std::min(batch_per_concat, batch - batch_iter * batch_per_concat);
                    int current_heads = score_kheads_per_concat;
                    int concat_dhead = current_batch * current_heads * dhead;  // concatenated head dimension
                    auto flash_attn_layer = make_gemm_layer(gqa_size, concat_dhead, (lin + (stage / 8)) / 2, 0, current_kv_addr, "Flash_attn");
                    layers.push_back(flash_attn_layer);             // Score: batch_size=gqa, hidden_dim=concat_dhead, out_ch=lin+stage
                    current_kv_addr += 2 * concat_dhead * (lin + (stage / 8)) * 2;
                }
            } else {
                // Original method: Repeat for each batch
                for (int b = 0; b < batch; ++b) {
                    for (int iter = 0; iter < score_concat_iterations; ++iter) {
                        int current_heads = std::min(score_kheads_per_concat, 2*tp_heads / gqa_size - iter * score_kheads_per_concat);
                        int concat_dhead = current_heads * dhead;  // concatenated head dimension
                        auto flash_attn_layer = make_gemm_layer(gqa_size, concat_dhead, (lin + (stage / 8)) / 2, 0, current_kv_addr, "Flash_attn"); // Use current KV address for score
                        layers.push_back(flash_attn_layer);             // Score: batch_size=1, hidden_dim=concat_dhead, out_ch=lin+stage
                        current_kv_addr += 2 * concat_dhead * (lin + (stage / 8)) * 2;
                    }
                }
            }
        } else {
            auto qkv_layer = make_gemm_layer(batch, hdim, tp_hdim + 2 * int(tp_hdim/gqa_size), current_weight_addr);
            push_gpu(qkv_layer); // QKV (GPU-side under attn_moe_only)
            current_weight_addr += hdim * (tp_hdim + 2 * int(tp_hdim/gqa_size)) * 2; // 2 bytes per weight

            // Score: lane width optimization (hidden_dim direction) with batch concat optimization
            // Use lane_width from NDP config, dhead=128 -> can concat multiple heads
            int score_kheads_per_concat = std::min(ndp_config.lane_width / dhead, tp_heads / gqa_size);
            int score_concat_iterations = (tp_heads / gqa_size + score_kheads_per_concat - 1) / score_kheads_per_concat;

            // If head count is low, consider batch concat to better utilize resources
            int max_batch_per_concat = int(ndp_config.lane_width / (score_kheads_per_concat * dhead)); // Maximum batches that can be concatenated
            if (stage == 1) std::cout << "Score concat: heads per concat=" << score_kheads_per_concat << ", max batch per concat=" << max_batch_per_concat << std::endl;
            if (score_kheads_per_concat <= int((ndp_config.lane_width / dhead) / 2)) { // If using less than 25% of lane width
                int batch_per_concat = std::min(max_batch_per_concat, batch);
                int batch_concat_iterations = (batch + batch_per_concat - 1) / batch_per_concat;
                for (int batch_iter = 0; batch_iter < batch_concat_iterations; ++batch_iter) {
                    int current_batch = std::min(batch_per_concat, batch - batch_iter * batch_per_concat);
                    int current_heads = score_kheads_per_concat;
                    int concat_dhead = current_batch * current_heads * dhead;  // concatenated head dimension
                    auto flash_attn_layer = make_gemm_layer(gqa_size, concat_dhead, lin + (stage / 8), 0, current_kv_addr, "Flash_attn");
                    layers.push_back(flash_attn_layer);             // Score: batch_size=gqa, hidden_dim=concat_dhead, out_ch=lin+stage
                    current_kv_addr += 2 * concat_dhead * (lin + (stage / 8)) * 2;
                }
            } else {
                // Original method: Repeat for each batch
                for (int b = 0; b < batch; ++b) {
                    for (int iter = 0; iter < score_concat_iterations; ++iter) {
                        int current_heads = std::min(score_kheads_per_concat, tp_heads / gqa_size - iter * score_kheads_per_concat);
                        int concat_dhead = current_heads * dhead;  // concatenated head dimension
                        auto flash_attn_layer = make_gemm_layer(gqa_size, concat_dhead, lin + (stage / 8), 0, current_kv_addr, "Flash_attn"); // Use current KV address for score
                        layers.push_back(flash_attn_layer);             // Score: batch_size=1, hidden_dim=concat_dhead, out_ch=lin+stage
                        current_kv_addr += 2 * concat_dhead * (lin + (stage / 8)) * 2;
                    }
                }
            }
        }
        
        
        // NDP-level communication for sequence-level parallelism (8 NDPs per GPU)
        
        // Sequence parallelism splits the KV cache across the 8 NDP units, so the
        // partial attention outputs have to cross the intra-package NDP-to-NDP
        // link before the projection. This is the only place the SP link
        // latency is charged: the later "Comm" layers are layout shuffles that
        // ride along with this same exchange, so charging them again would
        // double-count the sequence-parallel round trip.
        layers.push_back(make_comm_layer(batch * tp_heads * dhead, /*comm_hops=*/1));

        auto proj_layer = make_gemm_layer(batch, tp_hdim, hdim, current_weight_addr);
        push_gpu(proj_layer);                        // Projection
        current_weight_addr += hdim * tp_hdim * 2; // 2 bytes per weight
        
        
        // NDP-level communication
        push_gpu(make_comm_layer(batch * hdim));
        

        // All-reduce across tp/8 GPUs (tensor parallel communication)
        push_gpu_allreduce(batch * hdim);
        
        push_gpu(make_layernorm_layer(batch/8 * hdim));                           // Norm1

        push_gpu(make_comm_layer(batch * hdim));
        
        // FFN layers
        if (is_moe) {
            // Qwen MoE FFN
            // router: [batch x hdim] * [hdim x num_experts]
            auto router = make_gemm_layer(batch, hdim, (moe_num_experts + ep - 1) / ep, current_weight_addr);
            push_gpu(router);
            current_weight_addr += (uint64_t)hdim * (uint64_t)(moe_num_experts + ep - 1) / ep * 2;

            // all-gather router logits across GPU-level TP groups
            push_gpu(make_comm_layer(batch * (moe_num_experts + ep - 1) / ep));
            push_gpu_allreduce(batch * (moe_num_experts + ep - 1) / ep);

            // execute top-k experts (approximate as repeating dense FFN computation top_k times)
            const uint64_t routing_step = (uint64_t)stage;
            const uint64_t seed = 0x9E3779B97F4A7C15ULL
                ^ (routing_step * 0xBF58476D1CE4E5B9ULL)
                ^ (moe_block_index * 0x94D049BB133111EBULL);
            moe_block_index++;
            const bool randomize = true;
            auto expert_tokens = build_moe_local_routing((uint64_t)batch, seed, randomize);
            {
                std::ostringstream oss;
                oss << "[NDP MoE routing] step " << stage
                    << " tokens " << batch
                    << " local_experts " << expert_tokens.size()
                    << " counts=";
                for (size_t i = 0; i < expert_tokens.size(); ++i) {
                    if (i > 0) oss << ',';
                    oss << expert_tokens[i];
                }
            }
            for (uint64_t count : expert_tokens) {
                if (count == 0) continue;
                auto ff1_layer = make_gemm_layer((uint32_t)count, hdim, ff_hdim * ep / 8, current_weight_addr);
                layers.push_back(ff1_layer);                     // FF1
                current_weight_addr += (uint64_t)hdim * (uint64_t)ff_hdim * (uint64_t)ep / 8 * 2;

                auto ff2_layer = make_gemm_layer((uint32_t)count, hdim, ff_hdim * ep / 8, current_weight_addr);
                layers.push_back(ff2_layer);                     // FF2 (gate)
                current_weight_addr += (uint64_t)hdim * (uint64_t)ff_hdim * (uint64_t)ep / 8 * 2;

                layers.push_back(make_gelu_layer((uint32_t)(count * (uint64_t)ff_hdim * (uint64_t)ep / 8))); // GLU

                auto ff3_layer = make_gemm_layer((uint32_t)count, ff_hdim * ep / 8, hdim, current_weight_addr);
                layers.push_back(ff3_layer);                     // FF3
                current_weight_addr += (uint64_t)ff_hdim * (uint64_t)ep / 8 * (uint64_t)hdim * 2;
            }

            // reduce-scatter combine across GPU-level TP groups
            //layers.push_back(make_comm_layer(batch * hdim));
            push_gpu_allreduce(batch * hdim);
        } else if (name.find("LLAMA") != std::string::npos) {
            // LLAMA SwiGLU
            auto ff1_layer = make_gemm_layer(batch, hdim, ff_hdim, current_weight_addr);
            push_gpu(ff1_layer);                     // FF1
            current_weight_addr += hdim * ff_hdim * 2; // 2 bytes per weight
            
            auto ff2_layer = make_gemm_layer(batch, hdim, ff_hdim, current_weight_addr);
            push_gpu(ff2_layer);                     // FF2 (gate)
            current_weight_addr += hdim * ff_hdim * 2; // 2 bytes per weight
            
            push_gpu(make_gelu_layer(batch * ff_hdim));                          // GLU
            
            auto ff3_layer = make_gemm_layer(batch, ff_hdim, hdim, current_weight_addr);
            push_gpu(ff3_layer);                     // FF3
            current_weight_addr += ff_hdim * hdim * 2; // 2 bytes per weight
        } else {
            // Standard FFN
            auto ff1_layer = make_gemm_layer(batch, hdim, ff_hdim, current_weight_addr);
            push_gpu(ff1_layer);                     // FF1
            current_weight_addr += hdim * ff_hdim * 2; // 2 bytes per weight
            
            push_gpu(make_gelu_layer(batch * ff_hdim));                          // Activation
            
            auto ff2_layer = make_gemm_layer(batch, ff_hdim, hdim, current_weight_addr);
            push_gpu(ff2_layer);                     // FF2
            current_weight_addr += ff_hdim * hdim * 2; // 2 bytes per weight
        }

        push_gpu(make_layernorm_layer(batch/8 * hdim));                        // Norm2

        // NDP-level communication 
        push_gpu(make_comm_layer(batch * hdim));
        

        // All-reduce across tp/8 GPUs (tensor parallel communication)
        push_gpu_allreduce(batch * hdim);
        
        push_gpu(make_comm_layer(batch * hdim));
        
        layers.push_back(finish_step(1));                                     // Finish step
    }
    
    return layers;
}
