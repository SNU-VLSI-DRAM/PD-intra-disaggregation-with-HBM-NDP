#pragma once

#include <string>

// Everything the command line can set, with the defaults the simulator used
// before named flags replaced the positional argument list.
//
// Parsing is separated from interpretation on purpose: CliOptions holds raw
// user intent, and config/ConfigBuilder.h turns it into a SimulationConfig and
// an NDPConfig.
struct CliOptions {
    // ---- What to run ----
    std::string model = "Qwen-235B";
    std::string dtype = "W16A16";
    std::string ndp_config_file = "ndp_config.json";
    bool prefill = true;         // false = decode

    // ---- Workload shape ----
    int batch = 16;
    int context_len = 16 * 1024; // input sequence length
    int out_len = 1024;          // generated length, used to size the address map
    int decode_steps = 16384;    // decode steps actually simulated (lout)

    // ---- Parallelism ----
    int tp = 8;
    int pp = 1;
    int sp = 8;

    // ---- GPU ----
    int sm_count = 0;            // 0 = all SMs (no masking)
    int total_sm_count = 148;    // GB200 default
    // See SimulationConfig::sm_bw_scaling_factor. Only matters with --sm-count.
    int sm_bw_scaling_factor = 5;
    double effective_bw = 0.0;   // 0 = DRAM sim; >0 = analytical BW in GB/s
    int gpu_tiling_mode = 0;     // 0 = reuse_optimized, 1 = uniform

    // ---- DRAM ----
    // GEMV units per channel one PIM COMP drives at once; see
    // SimulationConfig::pim_parallel_level. 16 is what the reference PIM
    // measurements used.
    int pim_parallel_level = 16;
    bool duplex = false;
    // Run the NDP on attention and the MoE experts only, leaving the rest of
    // the decoder to the GPU. Independent of --duplex, which is a memory
    // technology rather than a division of work.
    bool ndp_attn_moe_only = false;
    // Do not simulate the GPU at all and let the NDP decide when the run ends.
    // Separate from the schedule above so the NDP's half can be measured on its
    // own, or run alongside the GPU.
    bool no_gpu = false;
    // Duplex only widens the access granularity by itself. This additionally
    // scales the NDP's compute to match, by giving the PU four times the lanes.
    bool duplex_4x_compute = false;
    bool pim_mode = false;

    // ---- NDP ----
    bool use_ndp = true;
    int ndp_gpu_policy = 0;      // 0 = prefill_first, 1 = decode_first, 2 = interleaved

    // ---- Interconnect (see config/InterconnectModel.h) ----
    double nvlink_latency_ns = 800.0;    // one inter-GPU NVLink5 hop
    double sp_latency_ns = 400.0;        // one intra-package NDP-to-NDP hop
    double interconnect_bw_GBps = 1800.0;
    int collective_algo = 0;             // 0 = one_shot, 1 = two_shot, 2 = ring

    // ---- Serving mode ----
    bool bullet = false;
    // Apply bullet's hardware -- half the DRAM channels, its HBM3e model and
    // its derated peak throughput -- without splitting the GPU's SMs, so other
    // arrangements can be compared against bullet on equal hardware.
    bool bullet_hw = false;
    int bullet_prefill_sm = 133;
    bool chunked_prefill = false;
    int chunk_size = 1024;
};

// Result of parsing. Exit codes let main() distinguish "--help was asked for"
// (success, stop) from "the command line was wrong" (failure, stop).
enum class CliResult { Run, StopSuccess, StopFailure };

// Accepts --flag, --key value and --key=value. Unknown options are an error.
CliResult parse_cli(int argc, char** argv, CliOptions& out);

void print_usage(const char* program);
void print_cli_summary(const CliOptions& opt);
