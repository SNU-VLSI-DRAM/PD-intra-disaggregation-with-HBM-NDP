#include "ndp/PU.h"
#include <cassert>
#include <iostream>

PU::PU(int num_lanes_, int max_size_, int lane_width_)
    : num_lanes(num_lanes_), lane_width(lane_width_), reg_loaded(false) {
    // buffer size is in number of entries, not bytes
    timing_buffer.max_size = max_size_;
}

bool PU::TimingBuffer::is_full() const {
    return addresses.size() >= max_size;
    assert(false);
}

bool PU::TimingBuffer::is_empty() const {
    return addresses.empty();
}

void PU::TimingBuffer::push(uint64_t addr) {
    if (!is_full()) {
    
        addresses.push_back(addr);
    }
}

bool PU::TimingBuffer::pop(uint64_t addr) {
    if (is_empty()) {
        return false;
    }
    
    // Find and remove the specific address
    for (auto it = addresses.begin(); it != addresses.end(); ++it) {
        if (*it == addr) {
            addresses.erase(it);
            return true;
        }
    }
    return false;  // Address not found
}

bool PU::TimingBuffer::is_issuable(uint64_t addr, int lane_width, int atomic_bytes) const {
    // To be issued, timing buffer must have addresses from addr to addr + (lane_width*2 bytes / atomic_bytes)
    int required = lane_width * 2 / atomic_bytes;
    int count = 0;
    for (const auto& address : addresses) {
        if (address >= addr && address < addr + lane_width * 2) {
            count++;
        } else {
        }
    }
    return count >= required;
}

bool PU::issue_tb(uint64_t addr, int atomic_bytes) {
    if (!reg_loaded) return false;
    if (timing_buffer.is_issuable(addr, lane_width, atomic_bytes)) {
        int required = lane_width * 2 / atomic_bytes;
        int popped = 0;
        while (popped < required && !timing_buffer.is_empty()) {
            if (timing_buffer.pop(addr + popped * atomic_bytes)) {
                popped++;
            } else {
                break;
            }
        }
        return popped == required;
    }
    return false;
}

void PU::cycle() {
    // Per-cycle update logic if needed
}
