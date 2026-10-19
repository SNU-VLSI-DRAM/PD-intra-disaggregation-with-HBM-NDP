// Tiling: picks a per-layer (Mb, Nb, Kb) blocking that fits in L2 and expands
// each layer into the GPU_Tile list the execution loop consumes, assigning
// every tile its activation / weight / output address ranges.

#include "gpu/GPU.h"

std::tuple<uint64_t, uint64_t, uint64_t> GPUmodel::search_best_tile_blocking(const LayerDesc &ld, uint64_t dtype_bytes, uint64_t target_l2_bytes) {
// ATTACC-like heuristic: choose block maximizing arithmetic intensity AI = flops / bytes_moved
    // Subject to L2 and L1 capacity constraints with a two-level (macro/micro) scheme.
    auto bytes_block = [&](uint64_t M, uint64_t N, uint64_t K) -> long double {
        return (long double)(M*K + K*N + M*N) * dtype_bytes; // A + B + partial C
    };

    // Uniform tiling mode: pick a roughly cubic block (Mb~Nb~Kb) that fits in L2.
    // Each dim is clamped to ld.M/N/K so naturally-small dims stay small. This skips
    // the reuse-maximizing traffic search entirely.
    if (_config.gpu_tiling_mode == 1) {
        // Solve 3*s^2*dtype_bytes <= target_l2_bytes  =>  s <= sqrt(L2 / (3*dtype))
        long double ideal = std::sqrt((long double)target_l2_bytes
                                      / (3.0L * (long double)std::max<uint64_t>(1, dtype_bytes)));
        uint64_t s = (uint64_t)std::max<long double>(1.0L, std::floor(ideal));
        uint64_t Mb = std::min<uint64_t>(ld.M, s);
        uint64_t Nb = std::min<uint64_t>(ld.N, s);
        uint64_t Kb = std::min<uint64_t>(ld.K, s);
        // Decode keeps no K blocking (K is typically tiny in generation; matches reuse mode).
        if (!_config.is_prefill) Kb = ld.K;
        // Shrink iso-cubically if the chosen tile still exceeds L2 (e.g., one dim was
        // promoted past s by !is_prefill above). Halve until it fits or hits 1.
        while (bytes_block(Mb, Nb, Kb) > (long double)target_l2_bytes) {
            uint64_t* big = &Mb;
            if (Nb > *big) big = &Nb;
            if (Kb > *big) big = &Kb;
            if (*big <= 1) break;
            *big = std::max<uint64_t>(1, (*big) / 2);
        }
        return {std::max<uint64_t>(1, Mb), std::max<uint64_t>(1, Nb), std::max<uint64_t>(1, Kb)};
    }

    // We minimize off-DRAM traffic approximated as:
    // Traffic = m_tiles*n_tiles*k_tiles*(A_tile + B_tile)*dtype + m_tiles*n_tiles*C_write*dtype
    unsigned __int128 best_traffic = (unsigned __int128)(-1);
    uint64_t best_Mb=1,best_Nb=1,best_Kb=1;
    if (ld.name == "moe_router") {
        for(uint64_t Mb=ld.M; Mb>=1; Mb/=2){
            for(uint64_t Nb=ld.N; Nb>=1; Nb/=2){
                uint64_t Kb = std::max<uint64_t>(1, ld.K/64);
                if (!_config.is_prefill) {
                    Kb = ld.K; // no K blocking for generation
                }
                long double need = bytes_block(Mb,Nb,Kb);
                if(need > (long double)target_l2_bytes) continue;
                uint64_t mtiles = ceil_div(ld.M,Mb);
                uint64_t ntiles = ceil_div(ld.N,Nb);
                uint64_t ktiles = ceil_div(ld.K,Kb);

                unsigned __int128 traffic;
                // reuse input A across all n tiles; input B across all m tiles
                if (ld.M >= ld.N) {
                    traffic = (unsigned __int128)((ld.M * ld.K + mtiles*(ld.K * ld.N)) * dtype_bytes) + (unsigned __int128)(ld.M * ld.N * dtype_bytes);
                }
                // reuse input B across all m tiles; input A across all n tiles
                else {
                    traffic = (unsigned __int128)((ld.N * ld.K + ntiles*(ld.K * ld.M)) * dtype_bytes) + (unsigned __int128)(ld.M * ld.N * dtype_bytes);
                }
                
                if(traffic < best_traffic){ best_traffic=traffic; best_Mb=Mb; best_Nb=Nb; best_Kb=Kb; }
                if(Nb==1) break;
            }
            if(Mb==1) break;
        }
    } else {
        for(uint64_t Mb=ld.M; Mb>=1; Mb/=2){
            for(uint64_t Nb=ld.N; Nb>=64; Nb/=2){
                uint64_t Kb = std::max<uint64_t>(1, ld.K/64);
                if (ld.name == "attn_flash" || ld.name == "QxK" || ld.name == "SxV") {
                    Kb = ld.K / 16;
                }
                if (!_config.is_prefill) {
                    Kb = ld.K; // no K blocking for generation
                }
                if (_config.chunked_prefill && ld.name == "attn_flash") {
                    Kb = ld.K; // no K blocking for chunked prefill
                }
                long double need = bytes_block(Mb,Nb,Kb);
                if(need > (long double)target_l2_bytes) continue;
                uint64_t mtiles = ceil_div(ld.M,Mb);
                uint64_t ntiles = ceil_div(ld.N,Nb);
                uint64_t ktiles = ceil_div(ld.K,Kb);

                unsigned __int128 traffic;
                // reuse input A across all n tiles; input B across all m tiles
                if (ld.M >= ld.N) {
                    if (_config.is_GQA && (ld.name == "QxK" || ld.name == "SxV")) {
                        traffic = (unsigned __int128)((_config.gqa_size * ld.M * ld.K + mtiles*(ld.K * ld.N)) * dtype_bytes) + 
                                (unsigned __int128)(_config.gqa_size * ld.M * ld.N * dtype_bytes);
                    } else if (ld.name == "attn_flash") {
                        // attention matrix is not stored
                        if (_config.is_GQA)
                            traffic = (unsigned __int128)((_config.gqa_size * ld.M * ld.K + 2*mtiles*(ld.K * ld.N)) * dtype_bytes) + 
                                    (unsigned __int128)(_config.gqa_size * ld.M * ld.K * dtype_bytes);
                        else
                            traffic = (unsigned __int128)((ld.M * ld.K + 2*mtiles*(ld.K * ld.N)) * dtype_bytes) + (unsigned __int128)(ld.M * ld.K * dtype_bytes);
                    } else if (ld.name == "softmax" || ld.name == "activation") {
                        // treat as memory-dominated with small compute; model as (M x N)
                        traffic = (unsigned __int128)(2 * ld.M * ld.N * dtype_bytes);
                    } else if (ld.name == "layernorm_1" || ld.name == "layernorm_2") {
                        traffic = (unsigned __int128)(3 * ld.M * ld.N) * dtype_bytes;
                    } else {
                        traffic = (unsigned __int128)((ld.M * ld.K + mtiles*(ld.K * ld.N)) * dtype_bytes) + (unsigned __int128)(ld.M * ld.N * dtype_bytes);
                    }
                }
                // reuse input B across all m tiles; input A across all n tiles
                else {
                    if (_config.is_GQA && (ld.name == "QxK" || ld.name == "SxV")) {
                        traffic = (unsigned __int128)((ld.N * ld.K + ntiles*(_config.gqa_size * ld.K * ld.M)) * dtype_bytes) + 
                                (unsigned __int128)(_config.gqa_size * ld.M * ld.N * dtype_bytes);
                    } else if (ld.name == "attn_flash") {
                        // attention matrix is not stored
                        if (_config.is_GQA)
                            traffic = (unsigned __int128)((2 * ld.N * ld.K + ntiles*(_config.gqa_size * ld.K * ld.M)) * dtype_bytes) + 
                                    (unsigned __int128)(_config.gqa_size * ld.M * ld.K * dtype_bytes);
                        else
                            traffic = (unsigned __int128)((2 * ld.N * ld.K + ntiles*(ld.K * ld.M)) * dtype_bytes) + (unsigned __int128)(ld.M * ld.K * dtype_bytes);
                    } else if (ld.name == "softmax" || ld.name == "activation") {
                        // treat as memory-dominated with small compute; model as (M x N)
                        traffic = (unsigned __int128)(2 * ld.M * ld.N * dtype_bytes);
                    } else if (ld.name == "layernorm_1" || ld.name == "layernorm_2") {
                        traffic = (unsigned __int128)(3 * ld.M * ld.K) * dtype_bytes;
                    } else {
                        traffic = (unsigned __int128)((ld.N * ld.K + ntiles*(ld.K * ld.M)) * dtype_bytes) + (unsigned __int128)(ld.M * ld.N * dtype_bytes);
                    }
                }
                
                if(traffic < best_traffic){ best_traffic=traffic; best_Mb=Mb; best_Nb=Nb; best_Kb=Kb; }
                if(Nb==1) break;
            }
            if(Mb==1) break;
        }
    }
    return {best_Mb,best_Nb,best_Kb};
}

void GPUmodel::partition_layer_into_tiles(const LayerDesc &ld, uint64_t dtype_bytes,
                                          uint64_t target_l2_bytes) {

    if (ld.name == "all_reduce_1" || ld.name == "all_reduce_2") {
        // model as memory-dominated with small compute; model as (M x N)
        GPU_Tile tile;
        tile.layer_id = ld.layer_id;
        tile.tile_index = (uint64_t)tiles_.size();
        tile.m = ld.M;
        tile.n = ld.N;
        tile.k = 1;
        tile.flops = (_config.tp - 1) * 2 * ld.M * ld.N / _config.tp; // assume all-reduce overhead = 2 * (tp-1) x BSH * k / tp
        tile.activation_base_addr = 0;
        tile.activation_end_addr = 0;
        tile.param_base_addr = 0;
        tile.param_end_addr = 0;
        tile.output_base_addr = 0;
        tile.output_end_addr = 0;
        tile.communication = true;
        tile.comm_hops = collective_hops_for_layer(ld.name);
        tiles_.push_back(tile);
        return;
    } else if (ld.name == "all_to_all") {
        // model as memory-dominated with small compute; model as (M x N)
        GPU_Tile tile;
        tile.layer_id = ld.layer_id;
        tile.tile_index = (uint64_t)tiles_.size();
        tile.m = ld.M;
        tile.n = ld.N;
        tile.k = 1;
        tile.flops = ld.replicas * (tp - 1) * ld.M * ld.N / tp; // assume all-to-all overhead = (tp-1) x BSH * k / tp
        tile.activation_base_addr = 0;
        tile.activation_end_addr = 0;
        tile.param_base_addr = 0;
        tile.param_end_addr = 0;
        tile.output_base_addr = 0;
        tile.output_end_addr = 0;
        tile.communication = true;
        tile.comm_hops = collective_hops_for_layer(ld.name);
        tiles_.push_back(tile);
        return;
    } else if (ld.name == "reduce_scatter") {
        // model as memory-dominated with small compute; model as (M x N)
        GPU_Tile tile;
        tile.layer_id = ld.layer_id;
        tile.tile_index = (uint64_t)tiles_.size();
        tile.m = ld.M;
        tile.n = ld.N;
        tile.k = 1;
        tile.flops = (tp - 1) * ld.M * ld.N / tp; // assume reduce_scatter overhead = (tp-1) x BSH / tp
        tile.activation_base_addr = 0;
        tile.activation_end_addr = 0;
        tile.param_base_addr = 0;
        tile.param_end_addr = 0;
        tile.output_base_addr = 0;
        tile.output_end_addr = 0;
        tile.communication = true;
        tile.comm_hops = collective_hops_for_layer(ld.name);
        tiles_.push_back(tile);
        return;
    } else if (ld.name == "pipeline_communication") {
        // model as memory-dominated with small compute; model as (M x N)
        GPU_Tile tile;
        tile.layer_id = ld.layer_id;
        tile.tile_index = (uint64_t)tiles_.size();
        tile.m = ld.M;
        tile.n = ld.N;
        tile.k = 1;
        tile.flops = ld.M * ld.N; // assume pipeline communication as MN ops (for simplicity)
        tile.activation_base_addr = 0;
        tile.activation_end_addr = 0;
        tile.param_base_addr = 0;
        tile.param_end_addr = 0;
        tile.output_base_addr = 0;
        tile.output_end_addr = 0;
        tile.communication = true;
        tile.comm_hops = collective_hops_for_layer(ld.name);
        tiles_.push_back(tile);
        return;
    }

    auto [best_Mb, best_Nb, best_Kb] = search_best_tile_blocking(ld, dtype_bytes, target_l2_bytes);

    const uint64_t heads_per_gpu = _config.num_heads / std::max<uint64_t>(1, tp);

    uint64_t m_tiles = ceil_div(ld.M, best_Mb);
    uint64_t n_tiles = ceil_div(ld.N, best_Nb);
    uint64_t k_tiles = ceil_div(ld.K, best_Kb);
    // Loop order: k -> m -> n (so A(m,k) reused across all n; B(k,n) reused across all m)
    // Use global sequential logical address space across ALL tiles (all layers, replicas)

    addr_type activation_base_addr = _config.base_activation_addr;
    addr_type param_base_addr;
    addr_type current_layer_param_addr;
    if (ld.name == "QxK" || ld.name == "SxV") {
        current_layer_param_addr = next_tile_KV_addr_;
    } else {
        current_layer_param_addr = next_tile_param_addr_;
    }
    param_base_addr = current_layer_param_addr;
    addr_type output_base_addr = _config.base_output_addr;

    for (uint64_t rep = 0; rep < ld.replicas; ++rep) {
        // Flat bitmaps for reuse tracking (0 = not loaded, 1 = loaded)
        std::vector<uint8_t> loaded_A(m_tiles * k_tiles, 0);     // index = m_idx * k_tiles + k_idx
        std::vector<uint8_t> loaded_B(k_tiles * n_tiles, 0);     // index = k_idx * n_tiles + n_idx
        std::vector<uint8_t> loaded_S(m_tiles * n_tiles, 0);     // softmax: index = m_idx * n_tiles + n_idx

        if (ld.M >= ld. N) {
            for (uint64_t m_idx = 0; m_idx < m_tiles; ++m_idx) {
                uint64_t m_extent = std::min<uint64_t>(best_Mb, ld.M - m_idx * best_Mb);

                // reload weight params
                if (ld.name == "QxK" || ld.name == "SxV" || ld.name == "attn_flash") {
                    next_tile_KV_addr_ = current_layer_param_addr;

                } else {
                    next_tile_param_addr_ = current_layer_param_addr;
                }
                loaded_B.assign(k_tiles * n_tiles, 0);
                for (uint64_t k_idx = 0; k_idx < k_tiles; ++k_idx) {
                    uint64_t k_extent = std::min<uint64_t>(best_Kb, ld.K - k_idx * best_Kb);
                    for (uint64_t n_idx = 0; n_idx < n_tiles; ++n_idx) {
                        uint64_t n_extent = std::min<uint64_t>(best_Nb, ld.N - n_idx * best_Nb);

                        uint64_t bytes_A = 0;
                        uint64_t bytes_B = 0;

                        if (ld.name == "softmax" || ld.name == "activation") {
                            size_t s_idx = m_idx * n_tiles + n_idx;
                            if (!loaded_S[s_idx]) {
                                bytes_A = m_extent * n_extent * dtype_bytes / 8; // utilize only one memory cube for simplicity
                                loaded_S[s_idx] = 1;
                            }
                        } else if (ld.name == "attn_flash") {
                            size_t a_idx = m_idx * k_tiles + k_idx;
                            if (_config.is_GQA) {
                                if (!loaded_A[a_idx]) {
                                    bytes_A = _config.gqa_size * m_extent * k_extent * dtype_bytes / 8; // A: Mb x Kb
                                    loaded_A[a_idx] = 1;
                                }
                            } else {
                                if (!loaded_A[a_idx]) {
                                    bytes_A = m_extent * k_extent * dtype_bytes / 8; // A: Mb x Kb
                                    loaded_A[a_idx] = 1;
                                }
                            }
                            size_t b_idx = k_idx * n_tiles + n_idx;
                            if (!loaded_B[b_idx]) {
                                bytes_B = 2 * k_extent * n_extent * dtype_bytes / 8; // B: Kb x Nb
                                loaded_B[b_idx] = 1;
                            }
                        } else if (ld.name == "layernorm_1" || ld.name == "layernorm_2") {
                            bytes_A = 2 * m_extent * n_extent * dtype_bytes / 8; // A: Mb x Kb
                        } else if ((ld.name == "QxK" || ld.name == "SxV") && _config.is_GQA) {
                            size_t a_idx = m_idx * k_tiles + k_idx;
                            if (!loaded_A[a_idx]) {
                                bytes_A = _config.gqa_size * m_extent * k_extent * dtype_bytes / 8; // A: Mb x Kb
                                loaded_A[a_idx] = 1;
                            }
                            size_t b_idx = k_idx * n_tiles + n_idx;
                            if (!loaded_B[b_idx]) {
                                bytes_B = k_extent * n_extent * dtype_bytes / 8; // B: Kb x Nb
                                loaded_B[b_idx] = 1;
                            }
                        } else {
                            size_t a_idx = m_idx * k_tiles + k_idx;
                            if (!loaded_A[a_idx]) {
                                bytes_A = m_extent * k_extent * dtype_bytes / 8; // A: Mb x Kb
                                loaded_A[a_idx] = 1;
                            }
                            size_t b_idx = k_idx * n_tiles + n_idx;
                            if (!loaded_B[b_idx]) {
                                bytes_B = k_extent * n_extent * dtype_bytes / 8; // B: Kb x Nb
                                loaded_B[b_idx] = 1;
                            }
                        }

                        uint64_t write_bytes = 0;
                        // Only write final accumulation for a (m,n) block after last k-slice
                        if (ld.write_output && (k_idx + 1 == k_tiles)) {
                            if (ld.name == "attn_flash") {
                                if (_config.is_GQA)
                                    write_bytes = _config.gqa_size * m_extent * k_extent * dtype_bytes / 8;
                                else
                                    write_bytes = m_extent * k_extent * dtype_bytes / 8;
                            } else {
                                write_bytes = m_extent * n_extent * dtype_bytes / 8;
                            }
                        }

                        uint64_t flops;

                        if (ld.name == "softmax" || ld.name == "layernorm_1" || ld.name == "layernorm_2") {
                            flops = (uint64_t)(5 * (long double)m_extent * n_extent * 160);
                        } else if (ld.name == "activation") {
                            flops = (uint64_t)(9 * (long double)m_extent * n_extent * 160);
                        } else if (ld.name == "attn_flash") {
                            if (_config.is_GQA)
                                flops = _config.gqa_size * (2 * m_extent * n_extent * k_extent + 5 * m_extent * n_extent + 2 * m_extent * n_extent * k_extent); // Flash attention does 4x the work for same output size
                            else
                                flops = 2 * m_extent * n_extent * k_extent + 5 * m_extent * n_extent + 2 * m_extent * n_extent * k_extent; // Flash attention does 4x the work for same output size
                            if (_config.is_prefill && !_config.chunked_prefill) flops /= 2; // using causal mask
                        } else {
                            flops = 2 * m_extent * n_extent * k_extent;
                        }

                        uint64_t activation_bytes = bytes_A; // input activation bytes
                        uint64_t param_bytes = bytes_B; // weight parameter bytes
                        uint64_t output_bytes = write_bytes; // output activation bytes

                        activation_base_addr = next_tile_activation_addr_;
                        output_base_addr = next_tile_output_addr_;

                        if (ld.name == "QxK" || ld.name == "SxV" || ld.name == "attn_flash") {
                            param_base_addr = next_tile_KV_addr_;
                        } else {
                            param_base_addr = next_tile_param_addr_;
                        }

                        addr_type activation_end_addr = activation_base_addr + activation_bytes; // in bytes
                        addr_type param_end_addr = param_base_addr + param_bytes; // in bytes
                        addr_type output_end_addr = output_base_addr + output_bytes; // in bytes

                        GPU_Tile t{ld.layer_id, (uint64_t)tiles_.size(), m_extent, n_extent, k_extent,
                                flops, activation_base_addr, activation_end_addr, param_base_addr, param_end_addr, 
                                output_base_addr, output_end_addr};

                        next_tile_activation_addr_ = activation_end_addr;
                        next_tile_output_addr_ = output_end_addr;
                        
                        if (ld.name == "QxK" || ld.name == "SxV" || ld.name == "attn_flash") {
                            // update global sequential address space
                            next_tile_KV_addr_ = param_end_addr;
                        } else {
                            // update global sequential address space
                            next_tile_param_addr_ = param_end_addr;
                        }
                        tiles_.push_back(t);
                    }
                }
            }
        } else {
            for (uint64_t n_idx = 0; n_idx < n_tiles; ++n_idx) {
                uint64_t n_extent = std::min<uint64_t>(best_Nb, ld.N - n_idx * best_Nb);

                // reload input params
                next_tile_activation_addr_ = _config.base_activation_addr;

                loaded_A.assign(m_tiles * k_tiles, 0);
                for (uint64_t k_idx = 0; k_idx < k_tiles; ++k_idx) {
                    uint64_t k_extent = std::min<uint64_t>(best_Kb, ld.K - k_idx * best_Kb);

                    for (uint64_t m_idx = 0; m_idx < m_tiles; ++m_idx) {
                        uint64_t m_extent = std::min<uint64_t>(best_Mb, ld.M - m_idx * best_Mb);

                        uint64_t bytes_A = 0;
                        uint64_t bytes_B = 0;

                        if (ld.name == "softmax" || ld.name == "activation") {
                            size_t s_idx = m_idx * n_tiles + n_idx;
                            if (!loaded_S[s_idx]) {
                                bytes_A = m_extent * n_extent * dtype_bytes / 8; // utilize only one memory cube for simplicity
                                loaded_S[s_idx] = 1;
                            }
                        } else if (ld.name == "attn_flash") {
                            size_t a_idx = m_idx * k_tiles + k_idx;
                            if (_config.is_GQA) {
                                if (!loaded_A[a_idx]) {
                                    bytes_A = _config.gqa_size * m_extent * k_extent * dtype_bytes / 8; // A: Mb x Kb
                                    loaded_A[a_idx] = 1;
                                }
                            } else {
                                if (!loaded_A[a_idx]) {
                                    bytes_A = m_extent * k_extent * dtype_bytes / 8; // A: Mb x Kb
                                    loaded_A[a_idx] = 1;
                                }
                            }
                            size_t b_idx = k_idx * n_tiles + n_idx;
                            if (!loaded_B[b_idx]) {
                                bytes_B = 2 * k_extent * n_extent * dtype_bytes / 8; // B: Kb x Nb
                                loaded_B[b_idx] = 1;
                            }
                        } else if (ld.name == "layernorm_1" || ld.name == "layernorm_2") {
                            bytes_A = 2 * m_extent * n_extent * dtype_bytes / 8; // A: Mb x Kb
                        } else if ((ld.name == "QxK" || ld.name == "SxV") && _config.is_GQA) {
                            size_t a_idx = m_idx * k_tiles + k_idx;
                            if (!loaded_A[a_idx]) {
                                bytes_A = _config.gqa_size * m_extent * k_extent * dtype_bytes / 8; // A: Mb x Kb
                                loaded_A[a_idx] = 1;
                            }
                            size_t b_idx = k_idx * n_tiles + n_idx;
                            if (!loaded_B[b_idx]) {
                                bytes_B = k_extent * n_extent * dtype_bytes / 8; // B: Kb x Nb
                                loaded_B[b_idx] = 1;
                            }
                        } else {
                            size_t a_idx = m_idx * k_tiles + k_idx;
                            if (!loaded_A[a_idx]) {
                                bytes_A = m_extent * k_extent * dtype_bytes / 8; // A: Mb x Kb
                                loaded_A[a_idx] = 1;
                            }
                            size_t b_idx = k_idx * n_tiles + n_idx;
                            if (!loaded_B[b_idx]) {
                                bytes_B = k_extent * n_extent * dtype_bytes / 8; // B: Kb x Nb
                                loaded_B[b_idx] = 1;
                            }
                        }

                        uint64_t write_bytes = 0;
                        // Only write final accumulation for a (m,n) block after last k-slice
                        if (ld.write_output && (k_idx + 1 == k_tiles)) {
                            if (ld.name == "attn_flash") {
                                if (_config.is_GQA)
                                    write_bytes = _config.gqa_size * m_extent * k_extent * dtype_bytes / 8;
                                else
                                    write_bytes = m_extent * k_extent * dtype_bytes / 8;
                            } else {
                                write_bytes = m_extent * n_extent * dtype_bytes / 8;
                            }
                        }

                        uint64_t flops;

                        if (ld.name == "softmax" || ld.name == "layernorm_1" || ld.name == "layernorm_2") {
                            flops = (uint64_t)(5 * (long double)m_extent * n_extent * 160);
                        } else if (ld.name == "activation") {
                            flops = (uint64_t)(9 * (long double)m_extent * n_extent * 160);
                        } else if (ld.name == "attn_flash") {
                            if (_config.is_GQA)
                                flops = _config.gqa_size * (2 * m_extent * n_extent * k_extent + 5 * m_extent * n_extent + 2 * m_extent * n_extent * k_extent); // Flash attention does 4x the work for same output size
                            else
                                flops = (2 * m_extent * n_extent * k_extent + 5 * m_extent * n_extent + 2 * m_extent * n_extent * k_extent); // Flash attention does 4x the work for same output size
                            if (_config.is_prefill && !_config.chunked_prefill) flops /= 2; // using causal mask
                        } else {
                            flops = 2 * m_extent * n_extent * k_extent;
                        }
                        
                        uint64_t activation_bytes = bytes_A; // input activation bytes
                        uint64_t param_bytes = bytes_B; // weight parameter bytes
                        uint64_t output_bytes = write_bytes; // output activation bytes

                        activation_base_addr = next_tile_activation_addr_;
                        output_base_addr = next_tile_output_addr_;

                        if (ld.name == "QxK" || ld.name == "SxV" || ld.name == "attn_flash") {
                            param_base_addr = next_tile_KV_addr_;
                        } else {
                            param_base_addr = next_tile_param_addr_;
                        }

                        addr_type activation_end_addr = activation_base_addr + activation_bytes; // in bytes
                        addr_type param_end_addr = param_base_addr + param_bytes; // in bytes
                        addr_type output_end_addr = output_base_addr + output_bytes; // in bytes
                        GPU_Tile t{ld.layer_id, (uint64_t)tiles_.size(), m_extent, n_extent, k_extent,
                                flops, activation_base_addr, activation_end_addr, param_base_addr, param_end_addr, 
                                output_base_addr, output_end_addr};

                        next_tile_activation_addr_ = activation_end_addr;
                        next_tile_output_addr_ = output_end_addr;

                        if (ld.name == "QxK" || ld.name == "SxV" || ld.name == "attn_flash") {
                            // update global sequential address space
                            next_tile_KV_addr_ = param_end_addr;
                        } else {
                            // update global sequential address space
                            next_tile_param_addr_ = param_end_addr;
                        }
                        tiles_.push_back(t);
                    }
                }
            }
        }
    }
}
