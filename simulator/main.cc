// Entry point: parse the command line, assemble the configuration, build the
// NDP layer schedule and run the simulator to completion.
//
// Everything that used to live here now has a home:
//   config/CliOptions.*     command-line parsing
//   config/ModelZoo.*       the model table
//   config/NdpConfigFile.*  the NDP hardware JSON reader
//   config/ConfigBuilder.*  address map and config assembly

#include <exception>
#include <iostream>
#include <vector>

#include "common/Common.h"
#include "config/CliOptions.h"
#include "config/ConfigBuilder.h"
#include "config/ModelZoo.h"
#include "config/NdpConfigFile.h"
#include "ndp/ndp_layer_builder.h"
#include "sim/Simulator.h"

namespace {

void report(const Simulator& sim) {
    std::cout << "Total memory requests pushed: " << sim.push_count << std::endl;
    std::cout << "Total Req_BW: " << (sim.push_count * 64) / sim.sim_cycles * 2 << " GB/s" << std::endl;
    std::cout << "Total memory responses popped: " << sim.pop_count << std::endl;
    std::cout << "Total Resp_BW: " << (sim.pop_count * 64) / sim.sim_cycles * 2 << " GB/s" << std::endl;
    std::cout << "Total simulation cycles: " << sim.sim_cycles
              << " (GPU cycles: " << sim.sim_cycles / 2 << ")" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    CliOptions opt;
    switch (parse_cli(argc, argv, opt)) {
        case CliResult::StopSuccess: return 0;
        case CliResult::StopFailure: return 1;
        case CliResult::Run: break;
    }
    print_cli_summary(opt);

    try {
        const ModelConfig model_cfg = make_model_config(opt.model, opt.dtype);
        const SimulationConfig sim_config = build_sim_config(model_cfg, opt);
        const NDPConfig ndp_config =
            build_ndp_config(parse_ndp_config(opt.ndp_config_file), sim_config, opt);
        print_config_summary(sim_config, ndp_config);

        Simulator sim(sim_config, ndp_config);

        // Only build the NDP schedule when the NDP is actually simulated. Doing
        // it unconditionally cost memory for a schedule nobody executed, and it
        // also crashed for GPU-only runs with sp = 1: the builder derives
        // heads-per-NDP as num_heads / (8 * tp / sp), which reaches 0 for a
        // grouped-query model once that quotient exceeds num_heads, and the
        // concat loops then divide by it.
        if (sim_config.use_ndp) {
            // The NDP layer builder counts NDP units, not GPUs: a node holds 8
            // GPUs and each GPU holds 8 NDP units, so its "tp" is 8 * the
            // GPU-level tp.
            NDPLayerBuilder ndp_layer_builder(model_cfg, ndp_config, sim_config.dtype_bytes,
                                              8 * sim_config.tp, sim_config.pp, sim_config.sp);
            // Duplex is an NDP-only configuration: attention and the MoE experts
            // run here, the rest of the decoder runs on the GPU against an
            // ordinary DRAM model in a separate run.
            ndp_layer_builder.attn_moe_only = sim_config.ndp_attn_moe_only;
            if (sim_config.ndp_attn_moe_only) {
                std::cout << "NDP runs attention and MoE experts only" << std::endl;
            }
            // Sequence parallelism splits the prompt across sp NDP units, so
            // each one sees context_len / sp tokens.
            const std::vector<Layer> layers =
                ndp_layer_builder.build(sim_config.batch_size,
                                        sim_config.context_len / sim_config.sp,
                                        sim_config.lout, sim_config.sp);
            std::cout << "Generated " << layers.size() << " layers for " << opt.model << std::endl;
            sim.load_layers(layers);
        } else {
            std::cout << "NDP disabled: skipping NDP layer construction" << std::endl;
        }

        std::cout << "Starting simulation..." << std::endl;
        while (sim.running()) {
            sim.cycle();
        }
        report(sim);
        if (sim_config.read_bw_test_ns > 0) {
            sim.report_read_bw();
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
