#pragma once
#include "ndp/ndp_ctrl.h"
#include "ndp/NDP_scheduler.h"
#include "config/NDPConfig.h"
#include "gpu/GPU.h"
#include "sim/Dram.h"
#include "common/Common.h"
#include <memory>
#include <vector>

class Simulator {
public:
	Simulator();
	Simulator(int buf_size, int vu_width, int pu_lanes, int tb_size, int lane_width);
	Simulator(const NDPConfig& config);
	Simulator(const SimulationConfig& sim_config, const NDPConfig& ndp_config);
	
	// Layer-based interface (recommended)
	void load_layers(const std::vector<Layer>& layers); 
	
	void cycle();

	bool running() const {
		if (sim_config.bullet) {
			return !gpu_prefill->gpu_done;
		}
		// With the GPU out of the picture the NDP decides when the run is over.
		if (sim_config.no_gpu) {
			return ndp_scheduler && !ndp_scheduler->ndp_done;
		}
		return !gpu.gpu_done;
	}
	
	// GPU initialization (if needed)
	void initialize_gpu_model(const SimulationConfig& sim_config);
	
	// NDP status queries
	bool ndp_has_pending_work() const;
	bool ndp_is_busy() const;
	DRAM dram;
	uint64_t push_count = 0;
	uint64_t pop_count = 0;
	cycle_type sim_cycles;

private:
	// Core components
	NDP_ctrl ndp;
	std::unique_ptr<NDP_scheduler> ndp_scheduler;
	GPUmodel gpu;
	// Bullet mode: separate prefill and decode GPU instances
	std::unique_ptr<GPUmodel> gpu_prefill;
	std::unique_ptr<GPUmodel> gpu_decode;
	bool bullet_turn_ = false; // false=prefill turn, true=decode turn for round-robin dispatch
	// Token bucket for GPU response bandwidth (responses popped from DRAM back to GPU).
	// Replenished by dram_channels / gpu_resp_bw_throttle_ratio each cycle.
	double gpu_resp_budget_ = 0.0;
	// For interleaved policy: alternates which side gets the first push slot each round.
	bool interleave_turn_ndp_ = false;

	
	// Configuration
	NDPConfig ndp_config;
	SimulationConfig sim_config;
	
	cycle_type gpu_req_start_cycle = 0;
	uint64_t _local_mem_accesss = 0;
	// Simulation state
	uint64_t issued_comp_requests = 0;
	uint64_t processed_comp_responses = 0;
	
};
