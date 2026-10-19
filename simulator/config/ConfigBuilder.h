#pragma once

#include "config/CliOptions.h"
#include "config/ModelConfig.h"
#include "config/NDPConfig.h"
#include "config/SimConfig.h"

// Turn the parsed command line plus a model description into the two
// configuration structs the simulator runs on. build_sim_config() also lays out
// the address map (param -> activation -> KV -> output -> NDP regions).
SimulationConfig build_sim_config(const ModelConfig& model, const CliOptions& opt);

// Takes the NDP hardware parameters read from the JSON file and fills in the
// address regions and interconnect settings derived from `sim`, so the NDP and
// the GPU always share one interconnect model.
NDPConfig build_ndp_config(NDPConfig from_file, const SimulationConfig& sim, const CliOptions& opt);

void print_config_summary(const SimulationConfig& sim, const NDPConfig& ndp);
