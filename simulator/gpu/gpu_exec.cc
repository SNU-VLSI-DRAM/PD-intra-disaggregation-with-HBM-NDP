// Execution loop: launches tiles in order, issues their memory traffic,
// applies the communication cost model to collective tiles, and retires
// tiles as DRAM responses arrive.

#include "gpu/GPU.h"

void GPUmodel::init() {
    tiles_.clear();
    gpu_cycle_ = 0;
    next_tile_launch_idx_ = 0;

    // Todo: load base addresses from simulation config
    next_tile_activation_addr_ = _config.base_activation_addr; // reset global sequential address space
    next_tile_param_addr_ = _config.base_param_addr;
    next_tile_KV_addr_ = _config.base_KV_addr;
    next_tile_output_addr_ = _config.base_output_addr;
    build_layers(_config.batch_size, _config.context_len, _config.hidden_dim, _config.num_heads, _config.dtype_bytes);
    for (const auto &ld : layers_) {
        partition_layer_into_tiles(ld, _config.dtype_bytes, _config.target_l2_bytes);
    }
    stat_compute_cycles = 0;
    stat_memory_cycles = 0;
    decode_steps_done = 0;
    decode_steps_total = _config.lout;
    step_start_cycle = 0;
    total_decode_cycles = 0;
    total_decode_bytes = 0;
    std::cout << "Total tiles generated: " << tiles_.size() << "\n";
    if (!_config.is_prefill && decode_steps_total > 1) {
        std::cout << "[Decode Multi-Step] Will run " << decode_steps_total << " decode steps\n";
    }
}

void GPUmodel::reset_for_next_step() {
    // Re-build layers and tiles for next decode step, preserving gpu_cycle_
    tiles_.clear();
    layers_.clear();
    next_tile_launch_idx_ = 0;
    next_tile_activation_addr_ = _config.base_activation_addr;
    next_tile_param_addr_ = _config.base_param_addr;
    next_tile_KV_addr_ = _config.base_KV_addr;
    next_tile_output_addr_ = _config.base_output_addr;
    current_mem_cycle = 0;
    outstanding_requests_ = 0;
    // Clear any leftover queues
    while (!_memory_request_queue.empty()) _memory_request_queue.pop();
    while (!_memory_response_queue.empty()) _memory_response_queue.pop();

    build_layers(_config.batch_size, _config.context_len, _config.hidden_dim, _config.num_heads, _config.dtype_bytes);
    for (const auto &ld : layers_) {
        partition_layer_into_tiles(ld, _config.dtype_bytes, _config.target_l2_bytes);
    }
    step_start_cycle = gpu_cycle_;
    gpu_done = false;
}

void GPUmodel::generate_memory_requests_for_tile(const GPU_Tile &t) {
    bool is_attention = false;
    std::string layer_name = "";
    if (t.layer_id >= 0 && t.layer_id < (int)layers_.size()) {
        layer_name = layers_[t.layer_id].name;
    }
    if (layer_name.find("attn") != std::string::npos || layer_name.find("attn_flash") != std::string::npos) {
        is_attention = true;
    }

    if (_config.pim_mode && is_attention) {
        const uint64_t total_bytes = (t.activation_end_addr - t.activation_base_addr)
                                   + (t.param_end_addr - t.param_base_addr)
                                   + (t.output_end_addr - t.output_base_addr);
        // GEMV units per channel engaged by one COMP; see SimulationConfig.
        const uint64_t pim_parallel_level = (uint64_t)std::max(1, _config.pim_parallel_level);
        const uint64_t bytes_per_comp = std::max<uint64_t>(1, _config.atomic_bytes * pim_parallel_level);
        uint64_t num_comps = ceil_div(total_bytes, bytes_per_comp);
        if (_config.is_GQA) {
            num_comps *= std::max<uint64_t>(1, _config.gqa_size);
        }
        std::cout << "Tile " << t.tile_index << " (Layer " << t.layer_id << "): issuing " << num_comps << " COMP requests for attention PIM execution\n";
        uint64_t burst_bits = 0;
        while ((1ULL << burst_bits) < std::max<uint64_t>(1, _config.atomic_bytes)) {
            burst_bits++;
        }

        auto remove_bit_window = [](uint64_t x, uint64_t pos, uint64_t width) {
            if (width == 0) return x;
            const uint64_t low_mask = (pos == 64) ? ~0ULL : ((1ULL << pos) - 1ULL);
            const uint64_t low = x & low_mask;
            const uint64_t high = x >> (pos + width);
            return low | (high << pos);
        };
        auto insert_bit_window = [](uint64_t x, uint64_t window, uint64_t pos, uint64_t width) {
            if (width == 0) return x;
            const uint64_t low_mask = (pos == 64) ? ~0ULL : ((1ULL << pos) - 1ULL);
            const uint64_t low = x & low_mask;
            const uint64_t high = x >> pos;
            const uint64_t win_mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1ULL);
            return low | ((window & win_mask) << pos) | (high << (pos + width));
        };

        auto mask_width = [](uint64_t mask) {
            uint64_t w = 0;
            while ((mask >> w) != 0) {
                w++;
            }
            return w;
        };

        uint64_t bg_width = has_pim_address_mapping_ ? mask_width(pim_bg_mask_) : 0;
        uint64_t ba_width = has_pim_address_mapping_ ? mask_width(pim_ba_mask_) : 0;
        uint64_t bg_abs_pos = has_pim_address_mapping_ ? (burst_bits + (uint64_t)pim_bg_pos_) : 0;
        uint64_t ba_abs_pos = has_pim_address_mapping_ ? (burst_bits + (uint64_t)pim_ba_pos_) : 0;

        uint64_t compressed_base = t.activation_base_addr;
        uint64_t bg_window = 0;
        uint64_t ba_window = 0;

        if (has_pim_address_mapping_ && (bg_width > 0 || ba_width > 0)) {
            if (bg_width > 0) {
                bg_window = (t.activation_base_addr >> bg_abs_pos) & pim_bg_mask_;
            }
            if (ba_width > 0) {
                ba_window = (t.activation_base_addr >> ba_abs_pos) & pim_ba_mask_;
            }

            // Remove higher-position field first, then lower-position field.
            if (bg_abs_pos > ba_abs_pos) {
                if (bg_width > 0) compressed_base = remove_bit_window(compressed_base, bg_abs_pos, bg_width);
                if (ba_width > 0) compressed_base = remove_bit_window(compressed_base, ba_abs_pos, ba_width);
            } else {
                if (ba_width > 0) compressed_base = remove_bit_window(compressed_base, ba_abs_pos, ba_width);
                if (bg_width > 0) compressed_base = remove_bit_window(compressed_base, bg_abs_pos, bg_width);
            }
        }

        // Attention tile start marker for PIM path (fire-and-forget, no response tracking in GPU queue).
        {
            auto gwrite_req = std::make_shared<MemoryAccess>();
            gwrite_req->dram_address = t.activation_base_addr;
            gwrite_req->is_ndp = false;
            gwrite_req->size = _config.atomic_bytes;
            gwrite_req->req_type = MemoryAccessType::GWRITE;
            gwrite_req->request = true;
            gwrite_req->start_cycle = gpu_cycle_;
            _memory_request_queue.push(gwrite_req);
        }

        // Issue response-producing COMP requests for attention PIM latency measurement.
        for (uint64_t i = 0; i < num_comps; i++) {
            auto comp_req = std::make_shared<MemoryAccess>();
            const uint64_t compressed_addr = compressed_base + i * _config.atomic_bytes;
            uint64_t comp_addr = compressed_addr;

            if (has_pim_address_mapping_ && (bg_width > 0 || ba_width > 0)) {
                // Inverse order of removal: insert lower-position field first.
                if (bg_abs_pos < ba_abs_pos) {
                    if (bg_width > 0) comp_addr = insert_bit_window(comp_addr, bg_window, bg_abs_pos, bg_width);
                    if (ba_width > 0) comp_addr = insert_bit_window(comp_addr, ba_window, ba_abs_pos, ba_width);
                } else {
                    if (ba_width > 0) comp_addr = insert_bit_window(comp_addr, ba_window, ba_abs_pos, ba_width);
                    if (bg_width > 0) comp_addr = insert_bit_window(comp_addr, bg_window, bg_abs_pos, bg_width);
                }
            }

            comp_req->dram_address = comp_addr;
            comp_req->is_ndp = false;
            comp_req->size = _config.atomic_bytes;
            comp_req->req_type = MemoryAccessType::COMP;
            comp_req->request = true;
            comp_req->start_cycle = gpu_cycle_;
            _memory_request_queue.push(comp_req);
        }

        // Attention tile end marker for PIM path (fire-and-forget, no response tracking in GPU queue).
        {
            auto readres_req = std::make_shared<MemoryAccess>();
            readres_req->dram_address = t.activation_base_addr;
            readres_req->is_ndp = false;
            readres_req->size = _config.atomic_bytes;
            readres_req->req_type = MemoryAccessType::READRES;
            readres_req->request = true;
            readres_req->start_cycle = gpu_cycle_;
            _memory_request_queue.push(readres_req);
        }

        return;
    }

    if (t.activation_end_addr > t.activation_base_addr) {
        for (addr_type addr = t.activation_base_addr; addr < t.activation_end_addr; addr += _config.atomic_bytes) {
            auto read_req = std::make_shared<MemoryAccess>();
            read_req->dram_address = addr;
            read_req->is_ndp = false;
            read_req->size = _config.atomic_bytes; //always read burst size
            read_req->req_type = MemoryAccessType::READ;
            read_req->request = true;
            read_req->start_cycle = gpu_cycle_;
            _memory_request_queue.push(read_req);
        }
    }
    if (t.param_end_addr > t.param_base_addr) {
        for (addr_type addr = t.param_base_addr; addr < t.param_end_addr; addr += _config.atomic_bytes) {
            auto read_req = std::make_shared<MemoryAccess>();
            read_req->dram_address = addr;
            read_req->is_ndp = false;
            read_req->size = _config.atomic_bytes; //always read burst size
            read_req->req_type = MemoryAccessType::READ;
            read_req->request = true;
            read_req->start_cycle = gpu_cycle_;
            _memory_request_queue.push(read_req);
        }
    }
    if (t.output_end_addr > t.output_base_addr) {
        for (addr_type addr = t.output_base_addr; addr < t.output_end_addr; addr += _config.atomic_bytes) {
            auto write_req = std::make_shared<MemoryAccess>();
            write_req->dram_address = addr;
            write_req->is_ndp = false;
            write_req->size = _config.atomic_bytes; //always write burst size
            write_req->req_type = MemoryAccessType::WRITE;
            write_req->request = true;
            write_req->start_cycle = gpu_cycle_;
            _memory_request_queue.push(write_req);
        }
    }
}

void GPUmodel::try_launch_tiles() {
    // Launch tiles in order if they are not started and prior layer tiles done when layer changes
    while (next_tile_launch_idx_ < tiles_.size()) {
        GPU_Tile &t = tiles_[next_tile_launch_idx_];
        // Enforce per-layer ordering (simple, no inter-layer overlap). If not first tile, ensure previous tile done.
        if (t.tile_index > 0) {
            const GPU_Tile &prev = tiles_[t.tile_index - 1];
            if (!prev.done) break;
        }
        // Launch this tile now
        t.launch_cycle = gpu_cycle_;

        if (t.communication) {
            t.mem_ready_cycle = t.launch_cycle; // assume all-reduce is instantaneous
            // alpha-beta cost model: T = hops * link_latency + volume / bandwidth.
            //
            // beta (bandwidth) term: unchanged, the per-collective volume that
            // partition_layer_into_tiles() stored in t.flops divided by the
            // fabric bandwidth.
            cycle_type communication_cycles = (uint64_t)std::max<long double>(1.0L, (long double)t.flops / interconnect_bw_); // assume all-reduce as 2*MN ops
            // alpha (latency) term: t.comm_hops link traversals. On a flat
            // NVLink5 fabric such as NVL72 this is 1 by default, so a collective
            // pays the 800 ns hop exactly once regardless of the TP degree.
            // It is charged once per collective layer, not per message, because
            // each collective is modelled as a single indivisible tile.
            communication_cycles += (cycle_type)t.comm_hops * link_latency_cycles_;
            t.compute_ready_cycle = t.launch_cycle + communication_cycles;
            t.memory_issued = true;
            t.compute_started = true;
            ++next_tile_launch_idx_;
            break; // launch only one all-reduce at a time
        }

        // Todo: generate memory requests for stats/tracing
        if (!t.memory_issued) {

            bool is_attention = false;
            std::string layer_name = "";
            if (t.layer_id >= 0 && t.layer_id < (int)layers_.size()) {
                layer_name = layers_[t.layer_id].name;
            }
            if (layer_name.find("attn") != std::string::npos || layer_name.find("attn_flash") != std::string::npos) {
                is_attention = true;
            }

            if (_config.pim_mode && is_attention) {
                // PIM attention path: issue COMP requests and wait for DRAM callbacks.
                generate_memory_requests_for_tile(t);
                t.memory_issued = true;
                t.mem_ready_cycle = ~0ULL;
            } else if (_config.effective_memory_bw_GBps > 0.0) {
                // Analytical BW mode: compute memory latency from profiled bandwidth
                // BW in GB/s = bytes/ns (since 1 cycle = 1 ns)
                uint64_t total_bytes = (t.activation_end_addr - t.activation_base_addr)
                                     + (t.param_end_addr - t.param_base_addr)
                                     + (t.output_end_addr - t.output_base_addr);
                cycle_type mem_cycles = (uint64_t)std::max<long double>(
                    1.0L, (long double)total_bytes / (long double)_config.effective_memory_bw_GBps);
                t.mem_ready_cycle = t.launch_cycle + mem_cycles;
                t.memory_issued = true;
            } else {
                // DRAM simulation mode: generate memory requests
                generate_memory_requests_for_tile(t);
                t.memory_issued = true;
                // set memory ready cycle to unknown (to be set when memory responses are received)
                t.mem_ready_cycle = ~0ULL; // unknown yet
            }
        }

        // Compute cycles
        cycle_type compute_cycles = (uint64_t)std::max<long double>(1.0L, (long double)t.flops / flops_per_cycle_);
        t.compute_ready_cycle = t.launch_cycle + compute_cycles;
        t.memory_issued = true;
        t.compute_started = true;
        ++next_tile_launch_idx_;
    }
}

void GPUmodel::receive_memory_response(MemoryAccess* resp) {
    // Mark tiles as memory-ready if all their memory requests have been served
    addr_type addr = resp->dram_address;

    auto erase_matching_response = [&](MemoryAccessType type, addr_type a) -> bool {
        if (_memory_response_queue.empty()) return false;
        std::queue<std::shared_ptr<MemoryAccess>> tmp;
        bool erased = false;
        while (!_memory_response_queue.empty()) {
            auto cur = _memory_response_queue.front();
            _memory_response_queue.pop();
            if (!erased && cur && cur->req_type == type && cur->dram_address == a) {
                erased = true;
                continue;
            }
            tmp.push(std::move(cur));
        }
        while (!tmp.empty()) {
            _memory_response_queue.push(std::move(tmp.front()));
            tmp.pop();
        }
        return erased;
    };

    for (auto &t : tiles_) {
        if (t.done || !t.memory_issued || t.mem_ready_cycle != ~0ULL) {
            current_mem_cycle = 0;
            continue;
        }
        //if (_config.pim_mode && resp->req_type == MemoryAccessType::COMP) {
            //_memory_response_queue.pop(); // remove the COMP request corresponding to this response
            //std::cout << "Received COMP response for tile " << t.tile_index << " of layer " << t.layer_id << " at cycle " << current_mem_cycle << "\n";
            //std::cout << "_memory_request_queue size: " << _memory_request_queue.size() << ", _memory_response_queue size: " << _memory_response_queue.size() << "\n"; 
        //} else 
        (void)erase_matching_response(resp->req_type, addr);
        current_mem_cycle = std::max(current_mem_cycle, resp->finish_cycle);
        if (_memory_request_queue.empty() && _memory_response_queue.empty() && t.mem_ready_cycle == ~0ULL) {
            t.mem_ready_cycle = current_mem_cycle;
        }
    }
}

void GPUmodel::advance_inflight() {
    bool mem_busy = false;
    bool comp_busy = false;
    for (auto &t : tiles_) {
        if (t.done || !t.memory_issued) continue;
        if (gpu_cycle_ >= t.mem_ready_cycle && gpu_cycle_ >= t.compute_ready_cycle) {
            if (t.tile_index % 50 == 0 && _config.is_prefill) std::cout << "Tile " << t.tile_index << " of layer " << t.layer_id << " done at cycle " << gpu_cycle_ << "\n";
            uint32_t transfer_size = (t.activation_end_addr - t.activation_base_addr) + (t.param_end_addr - t.param_base_addr) + (t.output_end_addr - t.output_base_addr);
            t.done = true;
            stat_compute_cycles += (t.compute_ready_cycle - t.launch_cycle);
            stat_memory_cycles += (t.mem_ready_cycle - t.launch_cycle);
        } else {
            if (gpu_cycle_ < t.mem_ready_cycle) mem_busy = true; else mem_busy = mem_busy || false;
            if (gpu_cycle_ < t.compute_ready_cycle) comp_busy = true; else comp_busy = comp_busy || false;
        }
    }
}

void GPUmodel::cycle() {
    // Advance in-flight tiles
    advance_inflight();
    // Try launching new tiles (simple immediate launch model)
    try_launch_tiles();
    ++gpu_cycle_;
    if (tiles_.size() > 0 && tiles_.back().done) {
        if (gpu_done == false) {
            // Current step just finished
            cycle_type step_cycles = gpu_cycle_ - step_start_cycle;
            decode_steps_done++;
            total_decode_cycles += step_cycles;

            if (!_config.is_prefill && !moe_routing_debug_.empty()) {
                for (const auto &entry : moe_routing_debug_) {
                    std::ostringstream oss;
                    oss << "  layer " << entry.layer_id
                        << " tokens " << entry.tokens
                        << " counts=";
                    for (size_t i = 0; i < entry.counts.size(); ++i) {
                        if (i > 0) oss << ',';
                        oss << entry.counts[i];
                    }
                    std::cout << oss.str() << "\n";
                }
            }

            if (!_config.is_prefill && decode_steps_total > 1) {
                // Multi-step decode mode
                // Calculate bytes transferred this step
                uint64_t step_bytes = 0;
                for (const auto &t : tiles_) {
                    step_bytes += (t.activation_end_addr - t.activation_base_addr)
                                + (t.param_end_addr - t.param_base_addr)
                                + (t.output_end_addr - t.output_base_addr);
                }
                // BW in GB/s: bytes / cycles, where 1 cycle = 1 ns -> bytes/ns = GB/s
                double step_bw = (step_cycles > 0) ? (double)step_bytes / (double)step_cycles : 0.0;
                total_decode_bytes += step_bytes;
                double avg_bw = (total_decode_cycles > 0) ? (double)total_decode_bytes / (double)total_decode_cycles : 0.0;

                double avg_cycles = (double)total_decode_cycles / (double)decode_steps_done;
                std::cout << "[Decode Step " << decode_steps_done << "/" << decode_steps_total
                          << "] finished at cycle " << gpu_cycle_
                          << " | step cycles: " << step_cycles
                          << " | avg cycles/step: " << avg_cycles
                          << " | step BW: " << step_bw << " GB/s"
                          << " | avg BW: " << avg_bw << " GB/s"
                          << "\n";

                if (decode_steps_done < decode_steps_total) {
                    // More steps to go — reset and continue
                    reset_for_next_step();
                    return; // don't set gpu_done
                }
                // All steps done — print final summary
                std::cout << "[Decode Complete] " << decode_steps_done << " steps in "
                          << gpu_cycle_ << " total cycles"
                          << " | avg " << (double)total_decode_cycles / decode_steps_done
                          << " cycles/step\n";
            } else {
                // Single step (prefill or lout=1 decode)
                std::cout << "GPU model done at cycle " << gpu_cycle_ << "\n";
            }

            std::cout << "total compute cycles: " << stat_compute_cycles << "\n";
            std::cout << "total memory cycles: " << stat_memory_cycles << "\n";
            std::cout << "average memory utilization: " << (double)stat_memory_cycles / (double)gpu_cycle_ << "\n";
            for (const auto &ld : layers_) {
                std::cout << "Layer " << ld.layer_id << " (" << ld.name << "): ";
                uint64_t layer_total_cycle = 0;
                uint64_t layer_compute_cycle = 0;
                uint64_t layer_memory_cycle = 0;
                for (const auto &t : tiles_) {
                    if (t.layer_id == ld.layer_id) {
                        layer_total_cycle += std::max((t.compute_ready_cycle - t.launch_cycle), (t.mem_ready_cycle - t.launch_cycle));
                        layer_compute_cycle += (t.compute_ready_cycle - t.launch_cycle);
                        layer_memory_cycle += (t.mem_ready_cycle - t.launch_cycle);
                    }
                }
                std::cout << "cycles = " << layer_total_cycle << "\n";
                std::cout << "   compute cycles = " << layer_compute_cycle << "\n";
                std::cout << "   memory cycles = " << layer_memory_cycle << "\n";
            }
        }
        gpu_done = true;
    }
}
