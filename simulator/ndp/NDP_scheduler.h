#pragma once

#include "ndp/ndp_ctrl.h"
#include "config/NDPConfig.h"
#include "common/Common.h"
#include <vector>
#include <queue>
#include <memory>

class NDP_scheduler {
public:
    NDP_scheduler(NDP_ctrl& ndp_ctrl, const NDPConfig& config);
    
    // Load layers and convert to instructions with tiling
    void load_layers(const std::vector<Layer>& layers);
    
    // Schedule and issue next instruction if possible
    void cycle(cycle_type current_cycle);
    
    // Check if there are pending instructions
    bool has_pending_instructions() const;
    
    // Check if NDP is currently busy
    bool is_busy() const;
    
    // Get current instruction index for debugging
    size_t get_current_instruction_index() const { return current_inst; }
    
    // Get total number of instructions
    size_t get_total_instructions() const { return instructions.size(); }
    
    // Simulation state
    bool ndp_done = false;
    NDP_ctrl& ndp;

private:
    NDPConfig ndp_config;
    std::vector<Instruction> instructions;
    size_t current_inst = 0;
    cycle_type stamp = 0;
    
    // Helper functions for instruction generation
    std::vector<Instruction> tile_gemm_instruction(const Instruction& inst, int buf_size, int num_lanes, int lane_width, bool QxK = false);
    void add_tiled_compute_instructions(const Layer& layer, bool& use_input_addr);
};