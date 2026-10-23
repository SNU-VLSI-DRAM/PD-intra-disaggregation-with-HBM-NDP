#include "config/ConfigBuilder.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

#include "config/SmBandwidthModel.h"

namespace {

uint64_t align4k(uint64_t addr) {
    constexpr uint64_t A = 4096;
    return (addr + (A - 1)) & ~(A - 1);
}

// Size of every region the simulator addresses, in bytes. The formulas are
// adapted from system.py's get_required_mem_capacity(); they only have to be
// large enough that regions do not overlap, so they are rough upper bounds.
// The trailing "/ 8" on each term is the per-GPU share within a node of 8.
struct MemoryFootprint {
    uint64_t weights;
    uint64_t activations;
    uint64_t kv_cache;
    uint64_t output;
};

MemoryFootprint memory_footprint(const ModelConfig& model, const CliOptions& opt,
                                 bool is_ffn_alter, bool is_moe) {
    const uint64_t hdim = model.hdim;
    const uint64_t ndec = model.ndec;
    const double ff_scale = model.ff_scale;
    const int tp = opt.tp;
    const int pp = opt.pp;
    const int gqa_size = model.gqa_size;
    const int batch = opt.batch;
    // Longest sequence any request reaches: prompt plus everything generated.
    const int l = opt.context_len + opt.out_len - 1;

    // Byte sizes depending on the dtype string (W16A16, W16A8, W8A16, W8A8).
    const int w_byte = 2;
    const int a_byte = 2;

    MemoryFootprint f{};
    // LLAMA-style FFNs have a 3 * ff_scale * hdim term, others 2 * ff_scale * hdim.
    const double ffn_term = (is_ffn_alter ? 3.0 : 2.0) * ff_scale * hdim;
    f.weights = (uint64_t)(ndec / pp * (hdim / tp) * (2 * hdim + 2 * hdim + ffn_term) * w_byte / 8);
    if (is_moe) {
        f.weights *= model.num_experts / tp;
    }
    f.activations = std::max((uint64_t)((uint64_t)batch * l * hdim * a_byte),
                             (uint64_t)((uint64_t)batch * 4 * hdim * a_byte)) / 8;
    f.kv_cache = (uint64_t)(ndec / pp * 2 * l * (hdim / tp / gqa_size) * a_byte) * batch / 8;
    f.output = (uint64_t)(3ULL * batch * l * hdim * a_byte) / 8;
    return f;
}

} // namespace

SimulationConfig build_sim_config(const ModelConfig& model, const CliOptions& opt) {
    SimulationConfig config;

    const bool is_ffn_alter = model.name.find("LLAMA") != std::string::npos ||
                              model.name.find("Qwen") != std::string::npos;
    const bool is_moe = model.name.find("Qwen") != std::string::npos;

    // ---- Address map: param -> activation -> KV -> output -> NDP regions ----
    const MemoryFootprint f = memory_footprint(model, opt, is_ffn_alter, is_moe);
    uint64_t base = 0;
    config.base_param_addr = base;            base = align4k(base + f.weights);
    config.base_activation_addr = base;       base = align4k(base + f.activations);
    config.base_KV_addr = base;               base = align4k(base + f.kv_cache);
    config.base_output_addr = base;           base = align4k(base + f.output);
    config.base_activation_addr_ndp = base;   base = align4k(base + 0x1000000); // 16 MB
    config.base_output_addr_ndp = base;       base = align4k(base + 0x1000000); // 16 MB
    config.base_KV_addr_ndp = base;

    // ---- Workload ----
    config.batch_size = opt.batch;
    config.context_len = opt.context_len;
    config.hidden_dim = model.hdim;
    config.num_heads = model.num_heads;
    config.dtype_bytes = 2;
    config.ff_scale = model.ff_scale;
    config.gqa_size = model.gqa_size;
    config.lout = opt.decode_steps;
    config.is_prefill = opt.prefill;

    // ---- Model flags ----
    config.is_ffn_alter = is_ffn_alter;
    config.is_GQA = (model.gqa_size > 1);
    config.flashattention = true;
    config.is_moe = is_moe;
    config.num_experts = model.num_experts;
    config.top_k = model.top_k;

    // ---- Parallelism ----
    config.tp = opt.tp;
    config.pp = opt.pp;
    config.sp = opt.sp;

    // ---- GPU ----
    config.per_gpu_tflops_ = 2560.0;              // ~2.5 PFLOPs, Blackwell class
    config.target_l2_bytes = 128 * 1024 * 1024;
    config.sm_count = opt.sm_count;
    config.total_sm_count = opt.total_sm_count;
    // Under bullet each half resolves its own factor in make_bullet_half_config().
    config.sm_bw_scaling_factor = opt.sm_bw_scaling_factor != kSmBwScalingFactorAuto
                                      ? opt.sm_bw_scaling_factor
                                      : sm_bw_scaling_factor_for(opt.sm_count);
    config.sm_bw_scaling_factor_auto = opt.sm_bw_scaling_factor == kSmBwScalingFactorAuto;
    config.effective_memory_bw_GBps = opt.effective_bw;
    config.gpu_tiling_mode = opt.gpu_tiling_mode;

    // ---- DRAM ----
    config.dram_channels = 32;
    config.pim_parallel_level = std::max(1, opt.pim_parallel_level);
    config.atomic_bytes = 64;
    config.dram_config_path = "../configs/HBM4_8Gb_x128.ini";
    config.log_dir = "./logs/BS16_16K_GPU";
    config.duplex = opt.duplex;
    config.duplex_4x_compute = opt.duplex_4x_compute;
    config.ndp_attn_moe_only = opt.ndp_attn_moe_only;
    config.no_gpu = opt.no_gpu;
    config.pim_mode = opt.pim_mode;

    // ---- Interconnect ----
    config.nvlink_latency_ns = opt.nvlink_latency_ns;
    config.sp_latency_ns = opt.sp_latency_ns;
    config.interconnect_bw_GBps = opt.interconnect_bw_GBps;
    config.collective_algo = opt.collective_algo;

    // ---- Scheduling / serving mode ----
    // Time-between-tokens budget, spread over the decoder stack.
    constexpr int target_TBT_ms = 500;
    config.decoding_slo = (cycle_type)target_TBT_ms * 1000000 / model.ndec;
    config.use_ndp = opt.use_ndp;
    config.ndp_gpu_policy = opt.ndp_gpu_policy;
    config.chunked_prefill = opt.chunked_prefill;
    config.chunk_size = opt.chunk_size;
    config.bullet = opt.bullet;
    config.bullet_prefill_sm = opt.bullet_prefill_sm;
    config.read_bw_test_ns = (cycle_type)std::max(0LL, opt.read_bw_test_ns);
    // The read test measures what the GPU alone can pull from DRAM, so the NDP
    // must not compete for it.
    if (config.read_bw_test_ns > 0) {
        config.use_ndp = false;
    }

    // Mode-specific overrides. Bullet mode runs two GPU instances against half
    // the channels, so it needs its own DRAM model and derated peak throughput.
    // --bullet-hw applies the same three without the SM split, so another
    // arrangement can be measured on the hardware bullet assumes.
    if (opt.bullet || opt.bullet_hw) {
        config.dram_channels = 16;
        config.dram_config_path = "../configs/HBM3e_8Gb_x128.ini";
        config.per_gpu_tflops_ = 2000.0;
    }
    if (opt.pim_mode) {
        config.dram_config_path = "../configs/HBM4_8Gb_x128_neupims.ini";
    }
    // Duplex needs a DRAM model that answers DUPLEX_READ with four reads' worth
    // of data, which is the wide-bus part: same page size, 256 B per
    // transaction instead of 64. Only the NDP issues DUPLEX_READ; the GPU keeps
    // issuing ordinary reads against the same device. Applied last, so it wins
    // if several modes are on at once.
    //
    // Duplex says how the NDP reaches memory, not what it computes: the NDP
    // still runs whatever schedule it was given, the whole decoder layer unless
    // --ndp-attn-moe-only narrows it.
    //
    // dram_channels is set alongside the path because it has to match the ini:
    // Simulator::cycle() drains responses over [0, dram_channels), so a count
    // below the model's would strand every response on the channels above it
    // and the run would hang.
    if (opt.duplex) {
        config.dram_config_path = "../configs/HBM4_8Gb_x128_duplex.ini";
        config.dram_channels = 32;
    }
    return config;
}

SimulationConfig make_bullet_half_config(const SimulationConfig& sim, bool is_prefill) {
    SimulationConfig half = sim;
    half.is_prefill = is_prefill;
    half.sm_count = is_prefill ? sim.bullet_prefill_sm : sim.total_sm_count - sim.bullet_prefill_sm;
    if (sim.sm_bw_scaling_factor_auto) {
        half.sm_bw_scaling_factor = sm_bw_scaling_factor_for(half.sm_count);
    }
    return half;
}

NDPConfig build_ndp_config(NDPConfig ndp, const SimulationConfig& sim, const CliOptions& opt) {
    if (opt.duplex) {
        // A duplex transaction returns 4x the data, so the PU is reshaped into
        // fewer, wider lanes to consume it. Total MAC width is unchanged:
        // 16 x 1024 becomes 4 x 4096.
        ndp.num_lanes_of_PU = 4;
        ndp.lane_width = 4096;
        if (opt.duplex_4x_compute) {
            // Scale the datapath to match the wider reads instead of merely
            // reshaping it: 4 x 4096 becomes 16 x 4096, four times the MACs.
            ndp.num_lanes_of_PU *= 4;
        }
    }

    ndp.weight_addr = sim.base_param_addr;
    ndp.input_addr = sim.base_activation_addr_ndp;
    ndp.output_addr = sim.base_output_addr_ndp;
    ndp.kv_addr = sim.base_KV_addr_ndp;

    // The NDP and the GPU must agree on the interconnect, otherwise the same
    // collective costs different amounts depending on who issues it.
    ndp.interconnect_bw = (long double)sim.interconnect_bw_GBps;
    ndp.nvlink_latency_ns = (long double)sim.nvlink_latency_ns;
    ndp.sp_latency_ns = (long double)sim.sp_latency_ns;
    ndp.collective_algo = sim.collective_algo;
    return ndp;
}

void print_config_summary(const SimulationConfig& sim, const NDPConfig& ndp) {
    std::cout << "NDP Config loaded:" << std::endl;
    std::cout << "  Buffer size: " << ndp.buffer_size << std::endl;
    std::cout << "  VU width: " << ndp.vu_width << std::endl;
    std::cout << "  Timing buffer size: " << ndp.timing_buffer_size << std::endl;
    std::cout << "  Num lanes: " << ndp.num_lanes_of_PU << std::endl;
    std::cout << "  Lane width: " << ndp.lane_width << std::endl;
    std::cout << "  DRAM config: " << sim.dram_config_path << std::endl;
    std::cout << "  Base activation addr: 0x" << std::hex << ndp.input_addr << std::dec << std::endl;
    std::cout << "  Base output addr: 0x" << std::hex << ndp.output_addr << std::dec << std::endl;
    std::cout << "  Base param addr: 0x" << std::hex << ndp.weight_addr << std::dec << std::endl;
    std::cout << "  Base KV addr: 0x" << std::hex << ndp.kv_addr << std::dec << std::endl;

    std::cout << "GPU Config generated:" << std::endl;
    std::cout << "  Batch size: " << sim.batch_size << std::endl;
    std::cout << "  Context len: " << sim.context_len << std::endl;
    std::cout << "  Hidden dim: " << sim.hidden_dim << std::endl;
    std::cout << "  Per GPU TFLOPs: " << sim.per_gpu_tflops_ << std::endl;
    std::cout << "  Base param addr: 0x" << std::hex << sim.base_param_addr << std::dec << std::endl;
    std::cout << "  Base activation addr: 0x" << std::hex << sim.base_activation_addr << std::dec << std::endl;
    std::cout << "  Base KV addr: 0x" << std::hex << sim.base_KV_addr << std::dec << std::endl;
    std::cout << "  Base output addr: 0x" << std::hex << sim.base_output_addr << std::dec << std::endl;
}
