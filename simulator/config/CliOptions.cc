#include "config/CliOptions.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "config/InterconnectModel.h"
#include "config/ModelZoo.h"
#include "config/SmBandwidthModel.h"

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

bool to_bool(const std::string& v) {
    const std::string s = lower(v);
    return s == "1" || s == "true" || s == "on" || s == "yes";
}

// Named alternatives for the enum-like options, so scripts can stay readable.
int to_enum(const std::string& raw, const char* option,
            const std::map<std::string, int>& names) {
    const std::string v = lower(raw);
    auto it = names.find(v);
    if (it != names.end()) return it->second;
    throw std::runtime_error("invalid value '" + raw + "' for " + option);
}

const std::map<std::string, int> kPolicyNames = {
    {"prefill_first", 0}, {"prefill", 0}, {"0", 0},
    {"decode_first", 1},  {"decode", 1},  {"1", 1},
    {"interleaved", 2},   {"interleave", 2}, {"rr", 2}, {"2", 2},
};

const std::map<std::string, int> kTilingNames = {
    {"reuse_optimized", 0}, {"reuse", 0}, {"0", 0},
    {"uniform", 1},         {"cubic", 1}, {"1", 1},
};

const std::map<std::string, int> kCollectiveNames = {
    {"one_shot", 0}, {"oneshot", 0}, {"nvls", 0}, {"0", 0},
    {"two_shot", 1}, {"twoshot", 1}, {"1", 1},
    {"ring", 2},     {"2", 2},
};

const char* policy_name(int v) {
    return v == 1 ? "decode_first" : v == 2 ? "interleaved" : "prefill_first";
}

const char* collective_name(int v) {
    return v == 1 ? "two_shot (RS+AG, 2 hops)"
         : v == 2 ? "ring (2*(ranks-1) hops)"
                  : "one_shot (NVLS, 1 hop)";
}

} // namespace

void print_usage(const char* program) {
    std::cout <<
"Usage: " << program << " [options]\n"
"\n"
"What to run\n"
"  --model NAME               model to simulate (default: Qwen-235B)\n"
"  --dtype STR                weight/activation precision tag (default: W16A16)\n"
"  --ndp-config PATH          NDP hardware JSON (default: ndp_config.json)\n"
"  --phase prefill|decode     which phase to simulate (default: prefill)\n"
"\n"
"Workload\n"
"  --batch N                  batch size (default: 16)\n"
"  --context-len N            input sequence length (default: 16384)\n"
"  --out-len N                generated length used to size the address map (default: 1024)\n"
"  --decode-steps N           decode steps to simulate (default: 16384)\n"
"\n"
"Parallelism\n"
"  --tp N                     tensor parallel degree (default: 8)\n"
"  --pp N                     pipeline parallel degree (default: 1)\n"
"  --sp N                     sequence parallel degree (default: 8)\n"
"\n"
"GPU\n"
"  --sm-count N               masked SM count, 0 = all SMs (default: 0)\n"
"  --total-sm-count N         SMs on the full GPU (default: 148)\n"
"  --sm-bw-scaling-factor N|auto\n"
"                             outstanding DRAM requests allowed per active SM, which\n"
"                             with --sm-count or --bullet caps the bandwidth an SM\n"
"                             group can reach. auto picks it per group from the SM\n"
"                             count, calibrated against B200 read bandwidth\n"
"                             (config/SmBandwidthModel.h) (default: auto)\n"
"  --effective-bw GBPS        analytical memory BW; 0 = run the DRAM model (default: 0)\n"
"  --tiling reuse|uniform     L2 blocking strategy (default: reuse)\n"
"\n"
"DRAM\n"
"  --pim-parallel-level N     GEMV units per channel one PIM COMP command drives at\n"
"                             once, which sets how much data it covers. Only --pim\n"
"                             reads it (default: 16)\n"
"  --duplex[=BOOL]            enable duplex DRAM mode: the DRAM model becomes\n"
"                             HBM4_8Gb_x128_duplex.ini, the NDP issues DUPLEX_READ and\n"
"                             its PU is reshaped to consume the wider returns. It does\n"
"                             not change what the NDP runs (default: off)\n"
"  --duplex-4x-compute[=BOOL] with --duplex, also give the NDP's PU 4x the lanes, so\n"
"                             compute scales with the wider reads (default: off)\n"
"  --ndp-attn-moe-only[=BOOL] run the NDP on attention and the MoE experts only, leaving\n"
"                             the rest of the decoder to the GPU (default: off)\n"
"  --no-gpu[=BOOL]            do not simulate the GPU; the NDP decides when the run ends.\n"
"                             Measures the NDP's half on its own (default: off)\n"
"  --pim[=BOOL]               route attention to PIM commands (default: off)\n"
"\n"
"NDP\n"
"  --ndp[=BOOL]               enable the NDP model (default: on)\n"
"  --ndp-gpu-policy P         prefill_first | decode_first | interleaved (default: prefill_first)\n"
"\n"
"Interconnect (1 simulator cycle == 1 ns)\n"
"  --nvlink-latency NS        one inter-GPU NVLink5 hop (default: 800)\n"
"  --sp-latency NS            one intra-package NDP-to-NDP hop (default: 400)\n"
"  --interconnect-bw GBPS     per-GPU fabric bandwidth (default: 1800)\n"
"  --collective one_shot|two_shot|ring\n"
"                             how many hops a collective is charged for.\n"
"                             one_shot models a flat 1-hop fabric with in-switch\n"
"                             reduction (NVLink SHARP on NVL72) and is the default;\n"
"                             ring charges 2*(tp-1) hops. (default: one_shot)\n"
"\n"
"Serving mode\n"
"  --bullet[=BOOL]            split SMs into concurrent prefill + decode groups (default: off)\n"
"  --bullet-hw[=BOOL]         use bullet's hardware (16 DRAM channels, HBM3e, 2000 TFLOPs)\n"
"                             without splitting the SMs, for like-for-like comparison\n"
"                             against bullet. Implied by --bullet (default: off)\n"
"  --bullet-prefill-sm N      SMs given to prefill in bullet mode (default: 133)\n"
"  --chunked-prefill[=BOOL]   enable chunked prefill (default: off)\n"
"  --chunk-size N             prefill chunk size (default: 1024)\n"
"\n"
"Microbenchmarks\n"
"  --read-bw-test NS          stream only reads for NS ns from every GPU instance\n"
"                             (both halves under --bullet) and report the read\n"
"                             bandwidth each one reached. Turns the NDP off\n"
"                             (default: 0 = off)\n"
"\n"
"  -h, --help                 show this message\n"
"\n"
"Models: ";
    const auto names = known_model_names();
    for (size_t i = 0; i < names.size(); ++i) {
        std::cout << names[i] << (i + 1 < names.size() ? ", " : "\n");
    }
}

CliResult parse_cli(int argc, char** argv, CliOptions& opt) {
    // Collect the arguments first so --key value and --key=value can share one
    // code path: `value_of()` consumes the next argument when needed.
    std::vector<std::string> args(argv + 1, argv + argc);

    for (size_t i = 0; i < args.size(); ++i) {
        std::string arg = args[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return CliResult::StopSuccess;
        }
        if (arg.rfind("--", 0) != 0) {
            std::cerr << "Error: unexpected positional argument '" << arg << "'.\n"
                      << "The simulator takes named options now; run with --help.\n";
            return CliResult::StopFailure;
        }

        std::string key = arg, inline_value;
        bool has_inline = false;
        if (const size_t eq = arg.find('='); eq != std::string::npos) {
            key = arg.substr(0, eq);
            inline_value = arg.substr(eq + 1);
            has_inline = true;
        }

        bool consumed_next = false;
        auto value_of = [&]() -> std::string {
            if (has_inline) return inline_value;
            if (i + 1 >= args.size()) throw std::runtime_error("missing value for " + key);
            consumed_next = true;
            return args[i + 1];
        };
        // A boolean flag may stand alone (--ndp) or carry a value (--ndp=0).
        auto flag_of = [&]() -> bool { return has_inline ? to_bool(inline_value) : true; };

        try {
            if      (key == "--model")           opt.model = value_of();
            else if (key == "--dtype")           opt.dtype = value_of();
            else if (key == "--ndp-config")      opt.ndp_config_file = value_of();
            else if (key == "--phase") {
                const std::string v = lower(value_of());
                if (v == "prefill" || v == "p")      opt.prefill = true;
                else if (v == "decode" || v == "d")  opt.prefill = false;
                else throw std::runtime_error("invalid value '" + v + "' for --phase");
            }
            else if (key == "--batch")           opt.batch = std::stoi(value_of());
            else if (key == "--context-len")     opt.context_len = std::stoi(value_of());
            else if (key == "--out-len")         opt.out_len = std::stoi(value_of());
            else if (key == "--decode-steps")    opt.decode_steps = std::stoi(value_of());
            else if (key == "--tp")              opt.tp = std::stoi(value_of());
            else if (key == "--pp")              opt.pp = std::stoi(value_of());
            else if (key == "--sp")              opt.sp = std::stoi(value_of());
            else if (key == "--sm-count")        opt.sm_count = std::stoi(value_of());
            else if (key == "--total-sm-count")  opt.total_sm_count = std::stoi(value_of());
            else if (key == "--sm-bw-scaling-factor") {
                const std::string v = lower(value_of());
                opt.sm_bw_scaling_factor = (v == "auto") ? kSmBwScalingFactorAuto : std::stoi(v);
                if (opt.sm_bw_scaling_factor < 0)
                    throw std::runtime_error("must be a positive integer or 'auto'");
            }
            else if (key == "--effective-bw")    opt.effective_bw = std::stod(value_of());
            else if (key == "--tiling")          opt.gpu_tiling_mode = to_enum(value_of(), "--tiling", kTilingNames);
            else if (key == "--pim-parallel-level") opt.pim_parallel_level = std::stoi(value_of());
            else if (key == "--duplex")          opt.duplex = flag_of();
            else if (key == "--ndp-attn-moe-only") opt.ndp_attn_moe_only = flag_of();
            else if (key == "--no-gpu")          opt.no_gpu = flag_of();
            else if (key == "--duplex-4x-compute") opt.duplex_4x_compute = flag_of();
            else if (key == "--pim")             opt.pim_mode = flag_of();
            else if (key == "--ndp")             opt.use_ndp = flag_of();
            else if (key == "--no-ndp")          opt.use_ndp = false;
            else if (key == "--ndp-gpu-policy")  opt.ndp_gpu_policy = to_enum(value_of(), "--ndp-gpu-policy", kPolicyNames);
            else if (key == "--nvlink-latency")  opt.nvlink_latency_ns = std::stod(value_of());
            else if (key == "--sp-latency")      opt.sp_latency_ns = std::stod(value_of());
            else if (key == "--interconnect-bw") opt.interconnect_bw_GBps = std::stod(value_of());
            else if (key == "--collective")      opt.collective_algo = to_enum(value_of(), "--collective", kCollectiveNames);
            else if (key == "--bullet")          opt.bullet = flag_of();
            else if (key == "--bullet-hw")       opt.bullet_hw = flag_of();
            else if (key == "--bullet-prefill-sm") opt.bullet_prefill_sm = std::stoi(value_of());
            else if (key == "--chunked-prefill") opt.chunked_prefill = flag_of();
            else if (key == "--chunk-size")      opt.chunk_size = std::stoi(value_of());
            else if (key == "--read-bw-test")    opt.read_bw_test_ns = std::stoll(value_of());
            else {
                std::cerr << "Error: unknown option '" << key << "'. Run with --help.\n";
                return CliResult::StopFailure;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << key << ": " << e.what() << "\n";
            return CliResult::StopFailure;
        }

        if (consumed_next) ++i;
    }

    // Validation that would otherwise fail deep inside the model.
    if (opt.tp < 1 || opt.pp < 1 || opt.sp < 1) {
        std::cerr << "Error: --tp/--pp/--sp must be >= 1.\n";
        return CliResult::StopFailure;
    }
    if (opt.batch < 1 || opt.context_len < 1) {
        std::cerr << "Error: --batch and --context-len must be >= 1.\n";
        return CliResult::StopFailure;
    }
    if (opt.duplex_4x_compute && !opt.duplex) {
        std::cerr << "Error: --duplex-4x-compute requires --duplex.\n";
        return CliResult::StopFailure;
    }
    if (opt.no_gpu && !opt.use_ndp) {
        std::cerr << "Error: --no-gpu and --no-ndp together leave nothing to simulate, and the"
                     " run would never finish.\n";
        return CliResult::StopFailure;
    }
    if (opt.duplex && opt.bullet) {
        std::cerr << "Error: --duplex and --bullet are mutually exclusive. Bullet splits the"
                     " GPU's SMs between prefill and decode, and duplex does not simulate the"
                     " GPU at all.\n";
        return CliResult::StopFailure;
    }
    if (opt.duplex && opt.pim_mode) {
        std::cerr << "warning: --pim routes the GPU's attention to PIM commands, and --duplex"
                     " does not simulate the GPU, so --pim has no effect here.\n";
    }
    if (opt.duplex && opt.prefill) {
        std::cerr << "warning: --duplex models the decode path, where attention and MoE"
                  << " run on the NDP; a prefill run will not exercise it.\n";
    }
    if (opt.bullet && (opt.bullet_prefill_sm <= 0 || opt.bullet_prefill_sm >= opt.total_sm_count)) {
        std::cerr << "Error: --bullet-prefill-sm must be strictly between 0 and --total-sm-count ("
                  << opt.total_sm_count << ").\n";
        return CliResult::StopFailure;
    }
    try {
        (void)make_model_config(opt.model, opt.dtype); // fail early on a bad name
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << ". Run with --help for the model list.\n";
        return CliResult::StopFailure;
    }
    return CliResult::Run;
}

void print_cli_summary(const CliOptions& opt) {
    std::cout << "Running simulation for model: " << opt.model << std::endl;
    std::cout << "Data type: " << opt.dtype << std::endl;
    std::cout << "Using config file: " << opt.ndp_config_file << std::endl;
    std::cout << "Duplex: " << (opt.duplex ? "ENABLED" : "disabled")
              << (opt.duplex && opt.duplex_4x_compute ? " (+4x NDP compute)" : "") << std::endl;
    if (opt.sm_count > 0) {
        std::cout << "SM Masking: " << opt.sm_count << "/" << opt.total_sm_count << " SMs" << std::endl;
    }
    if (opt.effective_bw > 0.0) {
        std::cout << "Analytical BW: " << opt.effective_bw << " GB/s (from profiling)" << std::endl;
    }
    std::cout << "Phase: " << (opt.prefill ? "PREFILL" : "DECODE") << std::endl;
    std::cout << "NDP: " << (opt.use_ndp ? "ON" : "OFF") << std::endl;
    if (opt.sm_count > 0) {
        std::cout << "SM BW scaling factor: ";
        if (opt.sm_bw_scaling_factor == kSmBwScalingFactorAuto) std::cout << "auto";
        else std::cout << opt.sm_bw_scaling_factor;
        std::cout << " outstanding requests per SM" << std::endl;
    }
    if (opt.pim_mode) {
        std::cout << "PIM parallel level: " << opt.pim_parallel_level
                  << " GEMV units per COMP" << std::endl;
    }
    std::cout << "PIM mode: " << (opt.pim_mode ? "ON" : "OFF") << std::endl;
    std::cout << "NDP/GPU push policy: " << policy_name(opt.ndp_gpu_policy) << std::endl;
    std::cout << "GPU tiling mode: "
              << (opt.gpu_tiling_mode == 1 ? "uniform (cubic)" : "reuse_optimized") << std::endl;
    std::cout << "Collective latency model: " << collective_name(opt.collective_algo) << std::endl;
    std::cout << "NVLink hop latency: " << opt.nvlink_latency_ns << " ns" << std::endl;
    std::cout << "NDP-to-NDP (SP) hop latency: " << opt.sp_latency_ns << " ns" << std::endl;
    if (opt.bullet) {
        std::cout << "Bullet mode: ENABLED (prefill SMs=" << opt.bullet_prefill_sm
                  << ", decode SMs=" << (opt.total_sm_count - opt.bullet_prefill_sm) << ")" << std::endl;
    }
}
