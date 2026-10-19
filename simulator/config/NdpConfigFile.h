#pragma once

#include <string>

#include "config/NDPConfig.h"

// Read the NDP hardware parameters from a flat JSON file. Keys the file does
// not mention keep the defaults set here; a missing file is a warning, not an
// error. Address keys (input_addr, output_addr, weight_addr, kv_addr) are read
// as hexadecimal, the rest as decimal.
NDPConfig parse_ndp_config(const std::string& filename);
