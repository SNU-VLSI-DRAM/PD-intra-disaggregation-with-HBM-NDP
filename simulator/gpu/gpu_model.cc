// GPUmodel construction: turns a SimulationConfig into the derived rates the
// rest of the model runs on (FLOPs per cycle, fabric bandwidth and latency,
// in-flight request budget). Layer construction lives in gpu_layers.cc, tiling in
// gpu_tiling.cc, and the execution loop in gpu_exec.cc.
#include "gpu/GPU.h"
#include <random>
#include <sstream>

GPUmodel::GPUmodel(SimulationConfig config)
    : per_gpu_tflops(config.per_gpu_tflops_), tp(config.tp), pp(config.pp),
      interconnect_bw_((long double)config.interconnect_bw_GBps),
      link_latency_cycles_((cycle_type)std::llround(config.nvlink_latency_ns)),
      gpu_cycle_(0), next_tile_param_addr_(0), next_tile_activation_addr_(0), next_tile_KV_addr_(0),
      next_tile_output_addr_(0), current_mem_cycle(0), next_tile_launch_idx_(0) {
    // Assume 1 cycle ~ 1 ns for mapping throughput -> cycles.
    // bytes per cycle (ns) = (GB/s * 1e9) / 8 / 1e9 = GB/s / 8 (bytes/ns)
    flops_per_cycle_ = (per_gpu_tflops * 1e12L) / 1e9L; // FLOPs per ns cycle if 1e9 cycles ~ 1 second notionally

    // SM masking: scale compute throughput proportionally to SM fraction
    if (config.sm_count > 0 && config.total_sm_count > 0) {
        double sm_ratio = (double)config.sm_count / (double)config.total_sm_count;
        flops_per_cycle_ *= (long double)sm_ratio;
        std::cout << "[SM Masking] Using " << config.sm_count << "/" << config.total_sm_count
                  << " SMs (ratio=" << sm_ratio << "), effective TFLOPs="
                  << (per_gpu_tflops * sm_ratio) << "\n";
    }

    if (flops_per_cycle_ <= 0) flops_per_cycle_ = 1;
    _config = config;
    gpu_done = false;

    // In-flight request budget: masking SMs also caps the bandwidth the GPU
    // can reach, at sm_count * sm_bw_scaling_factor outstanding requests.
    if (config.sm_count > 0) {
        max_outstanding_ = (uint64_t)config.sm_count * config.sm_bw_scaling_factor;
    } else {
        max_outstanding_ = 0; // 0 = unlimited
    }
    outstanding_requests_ = 0;
    if (max_outstanding_ > 0) {
        std::cout << "[SM BW Model] max outstanding requests: " << max_outstanding_
                  << " (" << config.sm_count << " SMs * "
                  << config.sm_bw_scaling_factor << " per SM)\n";
    }
}
