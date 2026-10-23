#include "sim/Simulator.h"
#include "ndp/NDP_scheduler.h"
#include "config/ConfigBuilder.h"
#include <iostream>

// Default constructor
Simulator::Simulator() : ndp(0, 4096, 1024, 8, 1024), gpu(SimulationConfig{}), dram(SimulationConfig{}), 
                         sim_cycles(0) {
	ndp_config.num_lanes_of_PU = 16;
	ndp_config.lane_width = 1024;
	ndp_scheduler = std::make_unique<NDP_scheduler>(ndp, ndp_config);
	gpu.init();
}


// Constructor with NDPConfig
Simulator::Simulator(const NDPConfig& config) 
	: ndp(0, config.buffer_size, config.vu_width, config.num_lanes_of_PU, config.timing_buffer_size, config.lane_width), 
	  gpu(SimulationConfig{}), dram(SimulationConfig{}), ndp_config(config), sim_cycles(0) {
	ndp_scheduler = std::make_unique<NDP_scheduler>(ndp, config);
	gpu.init();
}

// Constructor with both SimulationConfig and NDPConfig
Simulator::Simulator(const SimulationConfig& sim_config, const NDPConfig& ndp_config)
	: ndp(0, ndp_config.buffer_size, ndp_config.vu_width, ndp_config.num_lanes_of_PU, ndp_config.timing_buffer_size, ndp_config.lane_width),
	  gpu(sim_config), dram(sim_config), ndp_config(ndp_config), sim_config(sim_config), sim_cycles(0) {
	ndp_scheduler = std::make_unique<NDP_scheduler>(ndp, ndp_config);
	ndp.set_duplex(sim_config.duplex);
	// Wire DRAMSim3 bankgroup bit position into NDP duplex address generation.
	if (dram._mem) {
		ndp.set_duplex_bg_pos(static_cast<uint32_t>(dram._mem->GetBGPos()));
		gpu.set_pim_address_mapping(dram._mem->GetBGPos(), dram._mem->GetBGMask(),
		                            dram._mem->GetBAPos(), dram._mem->GetBAMask());
	}
	if (sim_config.bullet) {
		// Bullet mode: create two GPU instances with split SMs
		const SimulationConfig prefill_cfg = make_bullet_half_config(sim_config, true);
		const SimulationConfig decode_cfg = make_bullet_half_config(sim_config, false);

		gpu_prefill = std::make_unique<GPUmodel>(prefill_cfg);
		gpu_decode = std::make_unique<GPUmodel>(decode_cfg);
		if (dram._mem) {
			gpu_prefill->set_pim_address_mapping(dram._mem->GetBGPos(), dram._mem->GetBGMask(),
			                                     dram._mem->GetBAPos(), dram._mem->GetBAMask());
			gpu_decode->set_pim_address_mapping(dram._mem->GetBGPos(), dram._mem->GetBGMask(),
			                                    dram._mem->GetBAPos(), dram._mem->GetBAMask());
		}
		gpu_prefill->init();
		gpu_decode->init();
		std::cout << "[Bullet] Prefill SMs: " << prefill_cfg.sm_count
				  << ", Decode SMs: " << decode_cfg.sm_count << "\n";
	} else {
		gpu.init();
	}
}

void Simulator::load_layers(const std::vector<Layer>& layers) {
	if (ndp_scheduler) {
		ndp_scheduler->load_layers(layers);
	}
}


void Simulator::cycle() {
	sim_cycles++;
	
	if (sim_cycles % 2 == 0) {
		// When the NDP owns the stop condition it has to keep ticking until its
		// last instruction retires, not merely until the last one is issued.
		// Elsewhere the GPU ends the run, so the original
		// has_pending_instructions() condition is kept to leave those results
		// untouched.
		const bool ndp_active = sim_config.no_gpu
		                            ? ndp_scheduler->is_busy()
		                            : ndp_scheduler->has_pending_instructions();
		if (ndp_scheduler && ndp_active && sim_config.use_ndp) {
			ndp_scheduler->cycle(int(sim_cycles / 2));
		}
		if (sim_config.bullet) {
			if (!gpu_prefill->gpu_done) gpu_prefill->cycle();
			if (!gpu_decode->gpu_done) gpu_decode->cycle();
		} else if (!sim_config.no_gpu) {
			gpu.cycle();
		}
	} 
	
	// Cycle DRAM
	dram.cycle();

	bool test = false;

	if (sim_config.bullet) {
		// Bullet mode: interleave requests from prefill and decode GPUs
		// Round-robin: try one request at a time from each, alternating
		bool prefill_active = !gpu_prefill->_memory_request_queue.empty()
			&& (gpu_prefill->max_outstanding_ == 0 || gpu_prefill->outstanding_requests_ < gpu_prefill->max_outstanding_);
		bool decode_active = !gpu_decode->_memory_request_queue.empty()
			&& (gpu_decode->max_outstanding_ == 0 || gpu_decode->outstanding_requests_ < gpu_decode->max_outstanding_);

		while (prefill_active || decode_active) {
			bool made_progress = false;

			// Try prefill side (one request)
			if (prefill_active) {
				auto req = gpu_prefill->_memory_request_queue.front();
				req->gpu_tag = 1; // tag as prefill
				uint32_t cid = dram.get_channel_id(req.get());
				if (!dram.is_full(cid, req.get())) {
					if (dram.push(cid, req.get())) {
						gpu_prefill->_memory_request_queue.pop();
						gpu_prefill->_memory_response_queue.push(req);
						push_count++;
						gpu_prefill->outstanding_requests_++;
						made_progress = true;
					}
				}
			}

			// Try decode side (one request)
			if (decode_active) {
				auto req = gpu_decode->_memory_request_queue.front();
				req->gpu_tag = 2; // tag as decode
				uint32_t cid = dram.get_channel_id(req.get());
				if (!dram.is_full(cid, req.get())) {
					if (dram.push(cid, req.get())) {
						gpu_decode->_memory_request_queue.pop();
						gpu_decode->_memory_response_queue.push(req);
						push_count++;
						gpu_decode->outstanding_requests_++;
						made_progress = true;
					}
				}
			}

			if (!made_progress) break;

			// Re-evaluate active status
			prefill_active = !gpu_prefill->_memory_request_queue.empty()
				&& (gpu_prefill->max_outstanding_ == 0 || gpu_prefill->outstanding_requests_ < gpu_prefill->max_outstanding_);
			decode_active = !gpu_decode->_memory_request_queue.empty()
				&& (gpu_decode->max_outstanding_ == 0 || gpu_decode->outstanding_requests_ < gpu_decode->max_outstanding_);
		}
	} else if (sim_config.use_ndp) {
		// Policy-driven dispatch of GPU and NDP requests to DRAM.
		// Original behavior == ndp_gpu_policy=0 (prefill_first): GPU greedy first, then NDP greedy.
		auto try_push_gpu_once = [&]() -> bool {
			if (gpu._memory_request_queue.empty()) return false;
			auto req = gpu._memory_request_queue.front();
			uint32_t cid = dram.get_channel_id(req.get());
			if (dram.is_full(cid, req.get())) return false;
			if (!dram.push(cid, req.get())) return false;
			gpu._memory_request_queue.pop();
			gpu._memory_response_queue.push(req);
			push_count++;
			return true;
		};
		auto try_push_ndp_once = [&]() -> bool {
			if (!ndp.has_memory_request(0)) return false;
			auto req_sp = ndp.top_memory_request_shared(0);
			auto* req = req_sp.get();
			if (!req) {
				ndp.pop_memory_request(0);
				return false;
			}
			uint32_t cid = dram.get_channel_id(req);
			if (dram.is_full(cid, req)) return false;
			ndp.pop_memory_request(0);
			if (!dram.push(cid, req)) return false;
			push_count++;
			return true;
		};

		if (sim_config.ndp_gpu_policy == 1) {
			// decode_first: drain NDP first, then GPU.
			while (try_push_ndp_once()) {}
			while (try_push_gpu_once()) {}
		} else if (sim_config.ndp_gpu_policy == 2) {
			// interleaved: alternate one push from each side until neither can progress.
			bool progress = true;
			while (progress) {
				bool a, b;
				if (interleave_turn_ndp_) {
					a = try_push_ndp_once();
					b = try_push_gpu_once();
				} else {
					a = try_push_gpu_once();
					b = try_push_ndp_once();
				}
				interleave_turn_ndp_ = !interleave_turn_ndp_;
				progress = a || b;
			}
		} else {
			// prefill_first (default): drain GPU first, then NDP.
			while (try_push_gpu_once()) {}
			while (try_push_ndp_once()) {}
		}
	} else {
		// Dispatch GPU memory requests to DRAM, within the in-flight budget
		while (!gpu._memory_request_queue.empty()) {
			if (gpu.max_outstanding_ > 0 && gpu.outstanding_requests_ >= gpu.max_outstanding_) {
				break;
			}
			auto req = gpu._memory_request_queue.front();
			uint32_t cid = dram.get_channel_id(req.get());
			if (!dram.is_full(cid, req.get())) {
				if (dram.push(cid, req.get())) {
					gpu._memory_request_queue.pop();
					gpu._memory_response_queue.push(req);
					push_count++;
					gpu.outstanding_requests_++;
				}
			} else {
				break;
			}
		}
	}

	// Handle memory requests from NDP to DRAM. With use_ndp the policy dispatch
	// above already pushed them, so this only has to cover use_ndp=false, where
	// it is a no-op because nothing fills the NDP queue.
	if (!sim_config.use_ndp) {
		while (ndp.has_memory_request(0)) {
			auto req_sp = ndp.top_memory_request_shared(0);
			auto* req = req_sp.get();
			if (!req) {
				ndp.pop_memory_request(0);
				continue;
			}
			uint32_t cid = dram.get_channel_id(req);
			if (!dram.is_full(cid, req)) {
				ndp.pop_memory_request(0);
				dram.push(cid, req);
			} else {
				break;
			}
		}
	}
	
	test = false;

	// Handle memory responses from DRAM
	for (int ch = 0; ch < sim_config.dram_channels; ch++) {
		if (!dram.is_empty(ch)) {
			auto resp = dram.top(ch);
			resp->finish_cycle = sim_cycles/2;
			if (resp->is_ndp) {
				ndp.push_memory_response(resp);
			} else {
				if (sim_config.bullet) {
					// Route response to correct GPU based on gpu_tag
					GPUmodel* target = resp->gpu_tag == 1 ? gpu_prefill.get()
					                 : resp->gpu_tag == 2 ? gpu_decode.get() : nullptr;
					if (target) {
						if (target->outstanding_requests_ > 0) target->outstanding_requests_--;
						if (sim_config.read_bw_test_ns > 0) target->receive_read_stream_response(resp);
						else target->receive_memory_response(resp);
					}
				} else {
					if (gpu.outstanding_requests_ > 0) gpu.outstanding_requests_--;
					if (sim_config.read_bw_test_ns > 0) gpu.receive_read_stream_response(resp);
					else gpu.receive_memory_response(resp);
				}
			}
			pop_count++;
			dram.pop(ch);
		}
	}
}

void Simulator::report_read_bw() const {
	// sim_cycles tick at the DRAM clock, two per ns, and 1 byte/ns == 1 GB/s.
	const double ns = (double)sim_cycles / 2.0;
	auto line = [&](const char* name, const GPUmodel& g, int sms) {
		const double gbps = ns > 0 ? (double)g.read_stream_bytes() / ns : 0.0;
		std::cout << "[Read BW] " << name << ": SMs=" << sms
		          << " max_outstanding=" << g.max_outstanding_
		          << " bytes=" << g.read_stream_bytes()
		          << " BW=" << gbps << " GB/s" << std::endl;
		return gbps;
	};
	double total = 0.0;
	if (sim_config.bullet) {
		total += line("prefill", *gpu_prefill, sim_config.bullet_prefill_sm);
		total += line("decode", *gpu_decode, sim_config.total_sm_count - sim_config.bullet_prefill_sm);
	} else {
		total += line("gpu", gpu, sim_config.sm_count > 0 ? sim_config.sm_count : sim_config.total_sm_count);
	}
	std::cout << "[Read BW] total: " << total << " GB/s over " << ns << " ns" << std::endl;
}

bool Simulator::ndp_has_pending_work() const {
	if (ndp_scheduler) {
		return ndp_scheduler->has_pending_instructions();
	}
	return ndp_scheduler->get_current_instruction_index() < ndp_scheduler->get_total_instructions();
}

bool Simulator::ndp_is_busy() const {
	if (ndp_scheduler) {
		return ndp_scheduler->is_busy();
	}
	return ndp.running() || ndp_scheduler->get_current_instruction_index() < ndp_scheduler->get_total_instructions();
}


