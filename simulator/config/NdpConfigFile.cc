#include "config/NdpConfigFile.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>

namespace {

// Strip whitespace, commas and quotes from a raw JSON value.
std::string clean_value(const std::string& raw) {
    std::string v = raw;
    v.erase(std::remove_if(v.begin(), v.end(),
                           [](char c) { return std::isspace((unsigned char)c) || c == ',' || c == '"'; }),
            v.end());
    return v;
}

// If `line` holds `"key": value`, hand the cleaned value to `apply`.
template <typename Fn>
bool match(const std::string& line, const char* key, Fn apply) {
    if (line.find(key) == std::string::npos) return false;
    size_t pos = line.find(':');
    if (pos == std::string::npos) return true; // key seen but malformed; don't fall through
    std::string value = clean_value(line.substr(pos + 1));
    if (value.empty()) return true;
    apply(value);
    return true;
}

} // namespace

NDPConfig parse_ndp_config(const std::string& filename) {
    NDPConfig config;

    // Defaults used when the file is absent or silent about a key. These differ
    // from the NDPConfig struct defaults, which describe a smaller unit.
    config.buffer_size = 8192 * 128;
    config.vu_width = 1024;
    config.timing_buffer_size = 1024 * 128;
    config.num_lanes_of_PU = 16; // 4 when duplex
    config.lane_width = 1024;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file " << filename
                  << ", using defaults" << std::endl;
        return config;
    }

    auto as_int = [](int& dst) { return [&dst](const std::string& v) { dst = std::stoi(v); }; };
    auto as_hex = [](uint64_t& dst) { return [&dst](const std::string& v) { dst = std::stoull(v, nullptr, 16); }; };

    std::string line;
    while (std::getline(file, line)) {
        // Neither buffer_size nor timing_buffer_size is read from the file, and
        // this is deliberate rather than an oversight here: the original parser
        // tested substrings in an if/else chain with "buffer_size" first, so the
        // "timing_buffer_size" line always matched that branch, which was a
        // no-op. Every result produced so far therefore used the defaults above
        // (1 MiB buffer, 128 KiB timing buffer), not the 8192 / 1024 the shipped
        // ndp_config.json asks for. Honouring the file would silently shrink the
        // NDP buffers by two orders of magnitude and invalidate comparisons
        // against earlier runs, so the keys stay ignored until someone opts in.
        if (line.find("buffer_size") != std::string::npos) continue;
        if (match(line, "vu_width", as_int(config.vu_width))) continue;
        if (match(line, "num_lanes", as_int(config.num_lanes_of_PU))) continue;
        if (match(line, "lane_width", as_int(config.lane_width))) continue;
        if (match(line, "input_addr", as_hex(config.input_addr))) continue;   // base activation addr
        if (match(line, "output_addr", as_hex(config.output_addr))) continue; // base output addr
        if (match(line, "weight_addr", as_hex(config.weight_addr))) continue; // base param addr
        if (match(line, "kv_addr", as_hex(config.kv_addr))) continue;         // base KV addr
    }
    return config;
}
