#pragma once

#include <string>
#include <vector>

#include "config/ModelConfig.h"

// Look up a model by name. Throws std::runtime_error if the name is unknown.
ModelConfig make_model_config(const std::string& name, const std::string& dtype);

// Every name make_model_config() accepts, for --help and error messages.
std::vector<std::string> known_model_names();
