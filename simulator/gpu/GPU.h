#pragma once
#include <fstream>
#include <queue>
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <optional>
#include <unordered_map>

#include "common/Common.h"

class GPUmodel {
public:
    GPUmodel(SimulationConfig config);

    cycle_type gpu_cycle_;         // simulation cycle counter

    std::queue<std::shared_ptr<MemoryAccess>> _memory_request_queue;
    std::queue<std::shared_ptr<MemoryAccess>> _memory_response_queue;
    // Outstanding requests in flight to DRAM, against max_outstanding_
    uint64_t outstanding_requests_ = 0;
    uint64_t max_outstanding_ = 0; // set from sm_count * sm_bw_scaling_factor
    // Simulation state
    bool gpu_done;

    struct GPU_Tile {
        int layer_id;            // which layer
        uint64_t tile_index;     // index within that layer
        uint64_t m, n, k;        // GEMM logical tile problem size (after blocking)
        uint64_t flops;          // FLOPs for this tile
        addr_type activation_base_addr; // base addr for input activations
        addr_type activation_end_addr;  // end addr for input activations (exclusive)
        addr_type param_base_addr;      // base addr for weight parameters
        addr_type param_end_addr;       // end addr for weight parameters (exclusive)
        addr_type output_base_addr;     // base addr for output activations
        addr_type output_end_addr;      // end addr for output activations (exclusive)
        cycle_type launch_cycle = 0;      // cycle when tile was launched
        cycle_type mem_ready_cycle = 0;   // cycle when all memory done
        cycle_type compute_ready_cycle = 0; // cycle when compute done
        bool communication = false;
        uint32_t comm_hops = 0;  // serialized link traversals charged to this collective
        bool memory_issued = false;
        bool compute_started = false;
        bool done = false;
    };

    struct LayerDesc {
        std::string name;
        int layer_id;
        // Full problem dimensions per layer (logical, before tiling)
        uint64_t M, N, K; // GEMM form (M x K) * (K x N) = (M x N)
        uint64_t replicas = 1; // e.g. heads/batch replication
        bool write_output = true;
    };

    // Initialize a prefill run: builds layer list, partitions into tiles
    void init();

    // Reset for next decoding step (multi-step decode)
    void reset_for_next_step();

    // Advance simulation by one GPU cycle (cycle time implicitly 1ns if we map TFLOPs -> cycles directly)
    void cycle();

    void receive_memory_response(MemoryAccess* resp);

    // Read bandwidth microbenchmark (SimulationConfig::read_bw_test_ns > 0).
    // Keeps the request queue topped up with sequential reads; the in-flight
    // budget enforced by the Simulator is then the only thing limiting BW.
    void fill_read_stream();
    void receive_read_stream_response(MemoryAccess* resp);
    uint64_t read_stream_bytes() const { return read_stream_bytes_; }

    void set_pim_address_mapping(int bg_pos, uint64_t bg_mask, int ba_pos, uint64_t ba_mask) {
        pim_bg_pos_ = bg_pos;
        pim_bg_mask_ = bg_mask;
        pim_ba_pos_ = ba_pos;
        pim_ba_mask_ = ba_mask;
        has_pim_address_mapping_ = true;
    }

    const std::vector<GPU_Tile> &tiles() const { return tiles_; }
    uint64_t current_cycle() const { return gpu_cycle_; }

    // Multi-step decode tracking
    int decode_steps_done = 0;       // how many decode steps completed so far
    int decode_steps_total = 1;      // total decode steps to run (lout)
    cycle_type step_start_cycle = 0; // cycle when current step started
    cycle_type total_decode_cycles = 0; // cumulative cycles across all decode steps
    uint64_t total_decode_bytes = 0;   // cumulative bytes transferred across all decode steps

private:
    struct MoeRoutingDebugEntry {
        int layer_id;
        uint64_t tokens;
        std::vector<uint64_t> counts;
    };
    // Configuration
    double per_gpu_tflops;    // Compute throughput (TFLOPs)
    int tp;                   // tensor parallel degree
    int pp;                   // pipeline parallel degree (not yet pipelined, sequential for now)
    // L1 tiling removed; only L2-sized tiles considered now

    addr_type activation_base_addr;
    addr_type param_base_addr;
    addr_type output_base_addr;

    cycle_type current_mem_cycle = 0;

    cycle_type stat_compute_cycles = 0;
    cycle_type stat_memory_cycles = 0;

    // Derived
    long double interconnect_bw_ = 1800.0L; // GB/s -> bytes/ns (assume 1800 GB/s bidirectional)
    // One-hop inter-GPU link latency in cycles (1 cycle == 1 ns). See
    // SimulationConfig::nvlink_latency_ns.
    cycle_type link_latency_cycles_ = 800;
    long double flops_per_cycle_ = 0.0L; // from TFLOPs (assuming 1 cycle = 1 ns for simplicity)
    addr_type next_tile_activation_addr_ = 0;
    addr_type next_tile_param_addr_ = 0;
    addr_type next_tile_output_addr_ = 0;
    addr_type next_tile_KV_addr_ = 0;

    // DRAM mapping info (positions are post-shift-bit indices from DRAMSim config)
    bool has_pim_address_mapping_ = false;
    int pim_bg_pos_ = -1;
    uint64_t pim_bg_mask_ = 0;
    int pim_ba_pos_ = -1;
    uint64_t pim_ba_mask_ = 0;

    // config
    SimulationConfig _config;

    // Layer & tiling data
    std::vector<LayerDesc> layers_;
    std::vector<GPU_Tile> tiles_;
    std::vector<MoeRoutingDebugEntry> moe_routing_debug_;
    size_t next_tile_launch_idx_ = 0; // index of next tile to try launching

    // Read bandwidth microbenchmark state. The map owns every read from the
    // moment it is generated until DRAM answers it, since DRAM only holds a
    // raw pointer.
    addr_type read_stream_base_ = 0;
    addr_type read_stream_offset_ = 0;
    uint64_t read_stream_bytes_ = 0; // bytes returned by DRAM so far
    std::unordered_map<MemoryAccess*, std::shared_ptr<MemoryAccess>> read_stream_inflight_;

    // Internal helpers
    void build_layers(int batch_size, int context_len, int hidden_dim, int num_heads, int dtype_bytes);
    void partition_layer_into_tiles(const LayerDesc &ld, uint64_t dtype_bytes,
                                    uint64_t target_l2_bytes);
    void try_launch_tiles();
    void advance_inflight();
    static uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }
    void generate_memory_requests_for_tile(const GPU_Tile &t);
    // Link traversals charged to the collective implemented by layer `name`.
    uint32_t collective_hops_for_layer(const std::string &name) const;
    std::tuple<uint64_t, uint64_t, uint64_t> search_best_tile_blocking(const LayerDesc &ld, uint64_t dtype_bytes, uint64_t target_l2_bytes);
};