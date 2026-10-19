#pragma once

#include <cstdint>
#include <string>

#include "common/Types.h"
#include "config/InterconnectModel.h"

// Request push policy when use_ndp is true and the GPU is not halted.
enum class NdpGpuPolicy {
    PrefillFirst = 0,  // drain the GPU queue first, then the NDP
    DecodeFirst  = 1,  // drain the NDP queue first, then the GPU
    Interleaved  = 2,  // alternate one push from each side per turn
};

// Blocking strategy used by GPUmodel::search_best_tile_blocking().
enum class GpuTilingMode {
    // Minimize total DRAM traffic by picking a blocking that maximizes one
    // dimension and reuses across the others (large fetch + small streamed dim).
    ReuseOptimized = 0,
    // Pick a roughly cubic block (Mb ~ Nb ~ Kb) that fits in L2 so per-layer
    // tile sizes don't differ much across dims. Each dim is still clamped to
    // ld.M/N/K so naturally small dims stay small.
    Uniform = 1,
};

// Everything the GPU model, the DRAM model and the top-level Simulator need.
// Built by default_sim_config() and then overridden from the command line; see
// config/CliOptions.h.
typedef struct SimulationConfig {
    // ---- Address map (param -> activation -> KV -> output, then NDP regions) ----
    addr_type base_activation_addr = 0x0;
    addr_type base_param_addr = 0x0;
    addr_type base_KV_addr = 0x0;
    addr_type base_output_addr = 0x0;
    addr_type base_activation_addr_ndp = 0x0;
    addr_type base_KV_addr_ndp = 0x0;
    addr_type base_output_addr_ndp = 0x0;

    // ---- Workload shape ----
    int batch_size = 1;
    int context_len = 1024;
    int hidden_dim = 4096;
    int num_heads = 16;
    int dtype_bytes = 2; // e.g., 2 for FP16
    int gqa_size = 1;
    double ff_scale = 4.0;
    int lout = 1;               // number of decoding steps (output tokens to generate)

    // ---- GPU compute ----
    double per_gpu_tflops_ = 2560.0; // e.g., 2.5 PFLOPs for Blackwell
    uint64_t target_l2_bytes = 128 * 1024 * 1024; // target L2 cache size for tiling
    int sm_count = 0;              // 0 = all SMs (no masking); >0 = masked SM count
    int total_sm_count = 192;      // total SMs on GPU (e.g., 192 for GB200)
    int gpu_tiling_mode = (int)GpuTilingMode::ReuseOptimized;

    // ---- GPU memory path ----
    // Analytical memory BW mode (for decoding with profiled SM-masked BW).
    double effective_memory_bw_GBps = 0.0; // 0 = use DRAM sim; >0 = analytical (bytes/ns since 1 cycle=1ns)
    // Outstanding DRAM requests allowed per active SM. With sm_count it forms
    // the whole in-flight budget, max_outstanding_ = sm_count * this, which is
    // what caps the bandwidth a masked GPU can reach -- hence the name. It has
    // no effect at all unless sm_count > 0, where the budget is unlimited.
    int sm_bw_scaling_factor = 5;
    int sector_bytes = 32;         // L2 sector size in bytes

    // ---- Parallelism ----
    int tp = 1; // tensor parallelism
    int pp = 1; // pipeline parallelism
    int sp = 1; // sequence-level parallelism

    // ---- Interconnect / collective communication ----
    // See config/InterconnectModel.h. All latencies are in ns, and 1 ns == 1
    // simulator cycle on both the GPU and the NDP.
    //
    // Latency of ONE link traversal on the inter-GPU fabric. The 800 ns default
    // is a first-order figure for a single NVLink5 GPU->GPU hop through one
    // NVSwitch tray, i.e. the distance between any two GPUs inside a GB200
    // NVL72 domain, where the fabric is flat (every GPU is exactly 1 hop away).
    double nvlink_latency_ns = 800.0;
    // Aggregate inter-GPU fabric bandwidth available to one GPU, in GB/s.
    // Numerically equal to bytes/ns because 1 cycle == 1 ns.
    double interconnect_bw_GBps = 1800.0;
    // Latency of ONE NDP-to-NDP link traversal inside a single GPU package.
    // Used for the sequence-parallel exchange between the 8 NDP units; it is
    // shorter than an inter-GPU hop because it never leaves the package.
    double sp_latency_ns = 400.0;
    // How many link traversals a single collective is charged for. Only the
    // LATENCY term is affected; the bandwidth term keeps using the usual
    // 2*(ranks-1)/ranks ring volume.
    int collective_algo = (int)CollectiveAlgo::OneShot;

    // ---- Model flags ----
    bool is_ffn_alter = false;
    bool is_GQA = true;
    bool is_prefill = true;
    bool flashattention = true;
    bool is_moe = false;
    int num_experts = 1;
    int top_k = 1;

    // ---- DRAM ----
    int dram_channels = 32;
    // How many GEMV units per channel one PIM COMP command drives at once. It
    // is the only thing that sets how much data a COMP covers -- see
    // gpu_exec.cc, where bytes_per_comp = atomic_bytes * pim_parallel_level --
    // and nothing else in the simulator reads it, so it is a PIM knob rather
    // than a DRAM geometry one despite its former name.
    int pim_parallel_level = 16;
    uint64_t atomic_bytes = 64;
    std::string dram_config_path;
    std::string log_dir;
    bool duplex = false;
    // The NDP runs attention and the MoE experts only, leaving the rest of the
    // decoder to the GPU. Independent of duplex.
    bool ndp_attn_moe_only = false;
    // The GPU is not simulated and the NDP decides when the run ends.
    bool no_gpu = false;
    // Duplex alone widens the NDP's access granularity (see ndp_ctrl.h). This
    // additionally scales its compute, by giving the PU 4x the lanes.
    bool duplex_4x_compute = false;
    bool pim_mode = false;      // enable attention PIM modeling

    // ---- Scheduling / serving mode ----
    cycle_type decoding_slo = 0; // in cycles
    bool use_ndp = true;
    int ndp_gpu_policy = (int)NdpGpuPolicy::PrefillFirst;
    bool chunked_prefill = false;
    int chunk_size = 2048;
    // Bullet mode: split SMs into prefill + decode groups running concurrently.
    bool bullet = false;
    int bullet_prefill_sm = 0;  // SMs allocated to prefill (rest go to decode)
} SimulationConfig;
