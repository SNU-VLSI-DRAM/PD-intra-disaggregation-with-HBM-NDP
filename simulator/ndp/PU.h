
// PU.h
#pragma once
#include <vector>
#include <cstdint>

// Forward declaration
enum class Opcode;

class PU {
public:
    struct TimingBuffer {
        std::vector<uint64_t> addresses;
        int max_size; // in number of entries
        bool is_full() const;
        bool is_empty() const;
        void push(uint64_t addr);
        bool pop(uint64_t addr);
        bool is_issuable(uint64_t addr, int lane_width, int atomic_bytes) const;
    };

    PU(int num_lanes, int max_size, int lane_width = 1024);
    bool issue_tb(uint64_t addr, int atomic_bytes);
    void cycle();

    int num_lanes;
    int lane_width;
    bool reg_loaded;
    TimingBuffer timing_buffer;
};
    