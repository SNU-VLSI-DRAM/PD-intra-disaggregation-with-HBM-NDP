// Layer graph construction: expands a model configuration into the ordered
// list of GEMM / attention / communication layers one decoder block runs,
// for prefill, chunked prefill and decode, dense and MoE.

#include "gpu/GPU.h"
#include <random>

// Serialized link traversals charged to the collective implemented by `name`.
// The default (collective_algo = 0) models a flat 1-hop NVLink5 fabric with
// in-switch reduction, e.g. a GB200 NVL72 domain: every GPU of the TP group is
// exactly one NVSwitch hop away and the switch reduces and multicasts, so the
// whole collective is charged a single traversal no matter how large tp is.
uint32_t GPUmodel::collective_hops_for_layer(const std::string &name) const {
    const int ranks = std::max(1, tp);
    if (name == "all_reduce_1" || name == "all_reduce_2") {
        return collective_latency_hops(_config.collective_algo, ranks, /*is_allreduce=*/true);
    }
    if (name == "reduce_scatter" || name == "all_to_all") {
        // One-directional collectives: no reduce-then-broadcast round trip.
        return collective_latency_hops(_config.collective_algo, ranks, /*is_allreduce=*/false);
    }
    if (name == "pipeline_communication") {
        // Point-to-point send to the next pipeline stage: always a single hop,
        // and it does not involve the TP group at all.
        return (pp > 1) ? 1u : 0u;
    }
    return 0u;
}

void GPUmodel::build_layers(int batch_size, int context_len, int hidden_dim, int num_heads, int dtype_bytes) {
    layers_.clear();
    moe_routing_debug_.clear();
    int layer_id = 0;
    const uint64_t gqa_size = _config.gqa_size;
    const uint64_t BT = static_cast<uint64_t>(batch_size) * context_len;
    const uint64_t H = hidden_dim;
    const uint64_t Htp = H / std::max(1, tp);
    const uint64_t Htp_KV = Htp / gqa_size;
    const uint64_t heads = num_heads;
    const uint64_t heads_per_gpu = heads / std::max<uint64_t>(1, tp);
    const uint64_t dh = H / heads; // per head dim
    const uint64_t ff_dim = static_cast<uint64_t>(_config.ff_scale * Htp);
    const bool is_moe = _config.is_moe;
    const uint64_t num_experts = std::max(1, _config.num_experts);
    const uint64_t top_k = std::max(1, _config.top_k);
    uint64_t moe_block_index = 0;

    auto build_moe_local_routing = [&](uint64_t tokens, uint64_t seed, bool randomize) {
        const uint64_t local_experts = (num_experts + tp - 1) / tp;
        const uint64_t active_expert_slots = (top_k + tp - 1) / tp;
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

        // Skewed routing: sample expert weights from a low-concentration Dirichlet prior.
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

    auto add_ffn_block = [&](uint64_t tokens) {
        // Standard dense FFN (existing behavior)
        if (is_moe) {
            if (_config.is_ffn_alter) {
                layers_.push_back({"FFN_1", layer_id++, tokens, ff_dim * tp / 8, H, 1, true});
                layers_.push_back({"FFN_2", layer_id++, tokens, ff_dim * tp / 8, H, 1, true});
                layers_.push_back({"activation", layer_id++, tokens, ff_dim * tp / 8, 1, 1, false});
                layers_.push_back({"FFN_3", layer_id++, tokens, H, ff_dim * tp / 8, 1, true});
            } else {
                layers_.push_back({"FFN_1", layer_id++, tokens, 4 * Htp * tp / 8, H, 1, true});
                layers_.push_back({"activation", layer_id++, tokens, 4 * Htp * tp / 8, 1, 1, false});
                layers_.push_back({"FFN_2", layer_id++, tokens, H, 4 * Htp * tp / 8, 1, true});
            }
        } else {
            if (_config.is_ffn_alter) {
                layers_.push_back({"FFN_1", layer_id++, tokens, ff_dim, H, 1, true});
                layers_.push_back({"FFN_2", layer_id++, tokens, ff_dim, H, 1, true});
                layers_.push_back({"activation", layer_id++, tokens, ff_dim, 1, 1, false});
                layers_.push_back({"FFN_3", layer_id++, tokens, H, ff_dim, 1, true});
            } else {
                layers_.push_back({"FFN_1", layer_id++, tokens, 4 * Htp, H, 1, true});
                layers_.push_back({"activation", layer_id++, tokens, 4 * Htp, 1, 1, false});
                layers_.push_back({"FFN_2", layer_id++, tokens, H, 4 * Htp, 1, true});
            }
        }
    };

    auto add_moe_ffn_block = [&](uint64_t tokens) {
        // Simple MoE model:
        // 1) router: [tokens x H] * [H x num_experts]
        // 2) all-gather routing results across TP ranks (so each rank can select top-k experts)
        // 3) execute top_k experts for these tokens (approximate by scaling the FFN compute)
        // 4) reduce-scatter combine across TP ranks
        layers_.push_back({"moe_router", layer_id++, tokens, (num_experts + tp - 1) / tp, H, 1, true});
        if (tp > 1) {
            // Model as communication-only layer over the router logits [tokens x num_experts].
            layers_.push_back({"all_to_all", layer_id++, tokens, H, 1, (top_k + tp - 1) / tp, false});
        }
        const uint64_t routing_step = (uint64_t)decode_steps_done + 1;
        const uint64_t seed = 0x9E3779B97F4A7C15ULL
            ^ (routing_step * 0xBF58476D1CE4E5B9ULL)
            ^ (moe_block_index * 0x94D049BB133111EBULL);
        moe_block_index++;
        const bool randomize = !_config.is_prefill;
        auto expert_tokens = build_moe_local_routing(tokens, seed, randomize);
        if (randomize) {
            moe_routing_debug_.push_back({layer_id, tokens, expert_tokens});
        }
        for (uint64_t count : expert_tokens) {
            if (count == 0) continue;
            add_ffn_block(count);
        }
        if (tp > 1) {
            // Combine is implemented as reduce-scatter over [tokens x H].
            layers_.push_back({"reduce_scatter", layer_id++, tokens, H, 1, 1, false});
        } else {
            // Degenerate case: no communication.
            layers_.push_back({"moe_combine", layer_id++, tokens, H, 1, 1, true});
        }
    };

    if (_config.chunked_prefill && _config.is_prefill) {
        std::cout << "Building chunked prefill layers with chunk size " << _config.chunk_size << std::endl;
        const uint64_t chunk_size = _config.chunk_size;
        if (_config.is_GQA) {           
            // attention for prefill chunk and decoding tokens
            if (tp > (heads / gqa_size)) {
                layers_.push_back({"QKV_gen", layer_id++, chunk_size + batch_size, Htp + 2 * (2 * Htp_KV), H, 1, true});
                layers_.push_back({"attn_flash", layer_id++, chunk_size/batch_size, (uint64_t)context_len / 4, dh * heads_per_gpu * 2 / gqa_size, (uint64_t)batch_size, true});
                layers_.push_back({"attn_flash", layer_id++, 1, (uint64_t)context_len/2, dh * heads_per_gpu * 2 / gqa_size, (uint64_t)batch_size, true});
            } else {
                layers_.push_back({"QKV_gen", layer_id++, chunk_size + batch_size, Htp + 2 * (Htp_KV), H, 1, true});
                layers_.push_back({"attn_flash", layer_id++, chunk_size/batch_size, (uint64_t)context_len / 2, dh * heads_per_gpu / gqa_size, (uint64_t)batch_size, true});
                layers_.push_back({"attn_flash", layer_id++, 1, (uint64_t)context_len, dh * heads_per_gpu / gqa_size, (uint64_t)batch_size, true});
            }
        } else {
            layers_.push_back({"QKV_gen", layer_id++, chunk_size + batch_size, 3 * Htp, H, 1, true});
            layers_.push_back({"attn_flash", layer_id++, chunk_size/batch_size, (uint64_t)context_len / 2, dh * heads_per_gpu, (uint64_t)batch_size, true});
            layers_.push_back({"attn_flash", layer_id++, 1, (uint64_t)context_len, dh * heads_per_gpu, (uint64_t)batch_size, true});
        }
        // projection: [BT x Htp] * [Htp x H]
        layers_.push_back({"projection", layer_id++, chunk_size + batch_size, H, Htp, 1, true}); 
        // all-reduce
        if (tp > 1) {
            layers_.push_back({"all_reduce_1", layer_id++, chunk_size + batch_size, H, 1, 1, false}); // no output writeback
        }
        // residual add [BT x H] + [BT x H]// layernorm1: treat as memory-dominated with small compute; model as (BT x H)
        layers_.push_back({"layernorm_1", layer_id++, chunk_size + batch_size, H, 1, 1, true});
        // FFN / MoE FFN layers
        if (is_moe) {
            add_moe_ffn_block(chunk_size + batch_size);
        } else {
            add_ffn_block(chunk_size + batch_size);
        }
        if (tp > 1) {
            layers_.push_back({"all_reduce_2", layer_id++, chunk_size + batch_size, H, 1, 1, false}); // no output writeback
        }
        // residual add [BT x H] + [BT x H]// layernorm2: treat as memory-dominated with small compute; model as (BT x H)
        layers_.push_back({"layernorm_2", layer_id++, chunk_size + batch_size, H, 1, 1, true});
        if (pp > 1) {
            layers_.push_back({"pipeline_communication", layer_id++, chunk_size + batch_size, H, 1, 1, false}); // no output writeback
        }
        return;
    }

    if (_config.is_prefill) {
        // QKV: treat as GEMM [BT x H] * [H x 3Htp] (3 projections sharded on columns)
        if (_config.is_GQA) {
            if (tp > (heads / gqa_size)) {
                layers_.push_back({"QKV_gen", layer_id++, BT, Htp + 2 * (2 * Htp_KV), H, 1, true});
                layers_.push_back({"attn_flash", layer_id++, (uint64_t)context_len, (uint64_t)context_len/2, dh * heads_per_gpu * 2 / gqa_size, (uint64_t)batch_size, true});
            } else {
                layers_.push_back({"QKV_gen", layer_id++, BT, Htp + 2 * (Htp_KV), H, 1, true});
                layers_.push_back({"attn_flash", layer_id++, (uint64_t)context_len, (uint64_t)context_len, dh * heads_per_gpu / gqa_size, (uint64_t)batch_size, true});
            }
        } else {
            layers_.push_back({"QKV_gen", layer_id++, BT, 3 * Htp, H, 1, true});
            layers_.push_back({"attn_flash", layer_id++, (uint64_t)context_len, (uint64_t)context_len, dh * heads_per_gpu, (uint64_t)batch_size, true});
        }
        // projection: [BT x Htp] * [Htp x H]
        layers_.push_back({"projection", layer_id++, BT, H, Htp, 1, true}); 
        // all-reduce
        if (tp > 1) {
            layers_.push_back({"all_reduce_1", layer_id++, BT, H, 1, 1, false}); // no output writeback
        }
        // residual add [BT x H] + [BT x H]// layernorm1: treat as memory-dominated with small compute; model as (BT x H)
        layers_.push_back({"layernorm_1", layer_id++, BT, H, 1, 1, true});
        // FFN / MoE FFN layers
        if (is_moe) {
            add_moe_ffn_block(BT);
        } else {
            add_ffn_block(BT);
        }
        if (tp > 1) {
            layers_.push_back({"all_reduce_2", layer_id++, BT, H, 1, 1, false}); // no output writeback
        }
        // residual add [BT x H] + [BT x H]// layernorm2: treat as memory-dominated with small compute; model as (BT x H)
        layers_.push_back({"layernorm_2", layer_id++, BT, H, 1, 1, true});
        if (pp > 1) {
            layers_.push_back({"pipeline_communication", layer_id++, BT, H, 1, 1, false}); // no output writeback
        }
    } else {
        // QKV: treat as GEMM [BT x H] * [H x 3Htp] (3 projections sharded on columns)
        if (_config.is_GQA) {
            if (tp > (heads / gqa_size)) {
                layers_.push_back({"QKV_gen", layer_id++, (uint64_t)batch_size, Htp + 2 * (2 * Htp_KV), H, 1, true});
                layers_.push_back({"attn_flash", layer_id++, (uint64_t)1, (uint64_t)context_len/2, dh * heads_per_gpu * 2 / gqa_size, (uint64_t)batch_size, true});
            } else {
                layers_.push_back({"QKV_gen", layer_id++, (uint64_t)batch_size, Htp + 2 * (Htp_KV), H, 1, true});
                layers_.push_back({"attn_flash", layer_id++, (uint64_t)1, (uint64_t)context_len, dh * heads_per_gpu / gqa_size, (uint64_t)batch_size, true});
            }
        } else {
            layers_.push_back({"QKV_gen", layer_id++, (uint64_t)batch_size, 3 * Htp, H, 1, true});
            layers_.push_back({"attn_flash", layer_id++, (uint64_t)1, (uint64_t)context_len, dh * heads_per_gpu, (uint64_t)batch_size, true});
        }
        // projection: [BT x Htp] * [Htp x H]
        layers_.push_back({"projection", layer_id++, (uint64_t)batch_size, H, Htp, 1, true});
        if (tp > 1) {
            layers_.push_back({"all_reduce_1", layer_id++, (uint64_t)batch_size, H, 1, 1, false}); // no output writeback
        }
        // residual add// layernorm1: treat as memory-dominated with small compute; model as (BT x H)
        layers_.push_back({"layernorm_1", layer_id++, (uint64_t)batch_size, H, 1, 1, true});
        // FFN / MoE FFN layers
        if (is_moe) {
            add_moe_ffn_block((uint64_t)batch_size);
        } else {
            add_ffn_block((uint64_t)batch_size);
        }
        if (tp > 1) {
            layers_.push_back({"all_reduce_2", layer_id++, (uint64_t)batch_size, H, 1, 1, false}); // no output writeback
        }
        // residual add
        // layernorm2: treat as memory-dominated with small compute; model as (BT x H)
        layers_.push_back({"layernorm_2", layer_id++, (uint64_t)batch_size, H, 1, 1, true});
        if (pp > 1) {
            layers_.push_back({"pipeline_communication", layer_id++, (uint64_t)batch_size, H, 1, 1, false}); // no output writeback
        }
    }
}
