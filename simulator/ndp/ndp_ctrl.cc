#include "ndp/ndp_ctrl.h"
#include <cassert>
#include <iomanip>


constexpr int ATOMIC_BYTES = 64;
const int dram_channels = 32;
static constexpr int DUPLEX_RETURNS = 4;

NDP_ctrl::NDP_ctrl(uint32_t id, int buf_size, int vu_width, int pu_lanes, int tb_size, int lane_width)
    : _id(id), _buf_size(buf_size), VU_width(vu_width), pu(pu_lanes, tb_size / ATOMIC_BYTES, lane_width) {}

bool NDP_ctrl::issue_instruction(const Instruction& inst) {
    // Only issue new instruction when NDP is idle (Off state)
    if (_status != Status::Off) {
        return false; // Cannot issue new instruction while busy
    }

    if (gpu_comm_waiting && !inst.is_gpu_comm) {
        return false; // Cannot issue new instruction while GPU communication is ongoing
    }
    
    if (inst.opcode == Opcode::GEMM) {
        pu.reg_loaded = inst.preloaded;
        assert(inst.batch_size <= 16);
        assert(inst.hidden_dim <= 1024);
        assert(inst.out_ch <= _buf_size / 4);
        _batch_size = inst.batch_size;
        _hidden_dim = inst.hidden_dim;
        _out_ch = inst.out_ch;
        _addr_in = inst.addr_in;
        _addr_w = inst.addr_w;
        _addr_out = inst.addr_out;
        _sent_reqs = 0;
        _finished_reqs = 0;
        int input_tile_size = _batch_size * _hidden_dim;
        _num_reqs_gemm_load = int(input_tile_size * 2 / ATOMIC_BYTES);
        _addr_in_it = _addr_in;
        _addr_w_it = _addr_w;
        _addr_w_it_tb = _addr_w;
        _status = Status::GEMM_Load;
        _instruction_start_cycle = _total_cycles;
        _gemm_load_start_cycle = _total_cycles;
        _gemm_load_requests = 0;
        _gemm_mac_requests = 0;
    } else if (inst.opcode == Opcode::Layernorm || inst.opcode == Opcode::Softmax || inst.opcode == Opcode::Gelu || inst.opcode == Opcode::Add) {
        switch (inst.opcode) {
            case Opcode::Layernorm:
                _remaining_cycles = int(5 * inst.vector_size / VU_width);
                _status = Status::Layernorm;
                _instruction_start_cycle = _total_cycles;
                break;
            case Opcode::Softmax:
                _remaining_cycles = int(5 * inst.vector_size / VU_width);
                _status = Status::Softmax;
                _instruction_start_cycle = _total_cycles;
                break;
            case Opcode::Gelu:
                _remaining_cycles = int(9 * inst.vector_size / VU_width);
                _status = Status::GELU;
                _instruction_start_cycle = _total_cycles;
                break;
            case Opcode::Add:
                _remaining_cycles = int(inst.vector_size / VU_width);
                _status = Status::Add;
                _instruction_start_cycle = _total_cycles;
                break;
            default:
                break;
        }
    } else if (inst.opcode == Opcode::LoadBuffer || inst.opcode == Opcode::FlushOutBuffer) {
        _addr_in = inst.address;
        _addr_it = _addr_in;
        _data_size = inst.data_size;
        _num_reqs = int(inst.data_size / ATOMIC_BYTES);
        _finished_reqs = 0;
        _sent_reqs = 0;
        if (inst.opcode == Opcode::LoadBuffer) {
            _status = Status::Load;
            _instruction_start_cycle = _total_cycles;
            _load_requests = 0;
        } else {
            _status = Status::Flush;
            _instruction_start_cycle = _total_cycles;
            _flush_requests = 0;
        }
    } else if (inst.opcode == Opcode::GPUComm) {
        gpu_comm_waiting = true;
        gpu_comm_cycles = inst.comm_cycles;
    } else if (inst.opcode == Opcode::FinishStep) {
        prev_ndp_decoding_interval = ndp_decoding_interval;
        if (current_step > 0) {
            ndp_decoding_interval = _total_cycles - last_step_cycle;
            last_step_cycle = _total_cycles;
        } else {
            ndp_decoding_interval = _total_cycles;
            last_step_cycle = _total_cycles;
        }
        current_step++;
        average_decoding_cycles = _total_cycles / (current_step);
        std::cout << "NDP_ctrl: Finished step " << current_step << std::endl;
        std::cout << "average decoding cycles: " << average_decoding_cycles << std::endl;
        std::cout << "Decoding interval cycles: " << ndp_decoding_interval << std::endl;
        _status = Status::Off;
    } else {
        std::cerr << "NDP_ctrl: Unknown instruction opcode" << std::endl;
        return false;
    }
    
    return true; // Instruction issued successfully
}

void NDP_ctrl::push_memory_response(MemoryAccess* resp) {
    // Find matching request in pending queue
    std::queue<std::shared_ptr<MemoryAccess>> temp_queue;
    bool found = false;
    
    while (!_memory_request_pending_queue.empty()) {
        auto pending_req = _memory_request_pending_queue.front();
        _memory_request_pending_queue.pop();
        
        if (pending_req && pending_req->dram_address == resp->dram_address &&
            pending_req->req_type == resp->req_type) {
            // Found matching request
            found = true;
            break;
        } else {
            // Keep non-matching requests in temp queue
            temp_queue.push(pending_req);
        }
    }
    
    // Restore non-matching requests back to pending queue
    while (!temp_queue.empty()) {
        _memory_request_pending_queue.push(temp_queue.front());
        temp_queue.pop();
    }
    
    if (!found) {
        std::cerr << "NDP_ctrl: Unexpected memory response - address: " << std::hex 
                  << resp->dram_address << ", type: " << (int)resp->req_type << std::dec << std::endl;
        return;
    }
    
    if (_status == Status::GEMM_Load || _status == Status::GEMM_Mac) {
        int input_tile_size = _batch_size * _hidden_dim;
        int weight_tile_size = _out_ch * _hidden_dim;
        _finished_reqs += duplex_enabled_ ? DUPLEX_RETURNS : 1;
        if (_status == Status::GEMM_Load) {
            _total_gemm_load_responses++;
        } else {
            _total_gemm_mac_responses++;
        }
        if (_finished_reqs >= _num_reqs_gemm_load) {
            pu.reg_loaded = true;
        }
    } else if (_status == Status::Load) {
        if (resp->req_type == MemoryAccessType::READ) {
            _finished_reqs++;
            _total_load_responses++;
        } else if (resp->req_type == MemoryAccessType::DUPLEX_READ) {
            _finished_reqs += duplex_enabled_ ? DUPLEX_RETURNS : 1;
            _total_load_responses++;
        }
    } else if (_status == Status::Flush) {
        if (resp->req_type == MemoryAccessType::WRITE) {
            _finished_reqs += (duplex_enabled_ ? DUPLEX_RETURNS : 1);
            _total_flush_responses++;
        }
    }
}

void NDP_ctrl::cycle() {
    // Increment total cycle counter
    _total_cycles++;
    
    // Track off status cycles
    if (_status == Status::Off) {
        if (_off_start_cycle == 0) {
            _off_start_cycle = _total_cycles;
        }
    } else {
        if (_off_start_cycle > 0) {
            _off_total_cycles += (_total_cycles - _off_start_cycle);
            _off_start_cycle = 0;
        }
    }
    
    if (gpu_comm_waiting) {
        if (gpu_comm_cycles > 0) {
            gpu_comm_cycles--;
        }
        if (gpu_comm_cycles == 0) {
            gpu_comm_waiting = false;
        }
    }

    
    switch (_status) {
        case Status::GEMM_Load:
            if (pu.reg_loaded) {
                // Update GEMM Load statistics
                _gemm_load_total_cycles += (_total_cycles - _gemm_load_start_cycle);
                _gemm_load_count++;
                _total_gemm_load_requests += _gemm_load_requests;
                
                _status = Status::GEMM_Mac;
                _gemm_mac_start_cycle = _total_cycles;
                _sent_reqs = 0;
                _num_reqs = int(_out_ch * _hidden_dim * 2 / ATOMIC_BYTES);
            } else {
                for (int ch = 0; ch < dram_channels *2; ch++) {
                    if (_sent_reqs < _num_reqs_gemm_load) {
                        // Create and push read request for input data
                        auto req = std::make_shared<MemoryAccess>();
                        req->req_type = duplex_enabled_ ? MemoryAccessType::DUPLEX_READ : MemoryAccessType::READ;
                        req->dram_address = _addr_in_it;
                        req->is_ndp = true;
                        req->size = ATOMIC_BYTES;
                        req->request = true;
                        req->start_cycle = 0;
                        req->finish_cycle = 0;
                        if (push_memory_request(req)) {
                            _addr_in_it += ATOMIC_BYTES;
                            _sent_reqs += (duplex_enabled_ ? DUPLEX_RETURNS : 1);
                            _gemm_load_requests += (duplex_enabled_ ? DUPLEX_RETURNS : 1);
                        } else {
                            // Queue full - keep request for next cycle
                            //std::cout << "NDP_ctrl: Memory request queue full, cannot send more requests this cycle\n";
                            break; // Try again next cycle
                        }
                    }
                }
                if (_sent_reqs >= _num_reqs_gemm_load) {
                    // Update GEMM Load statistics
                    _gemm_load_total_cycles += (_total_cycles - _gemm_load_start_cycle);
                    _gemm_load_count++;
                    _total_gemm_load_requests += _gemm_load_requests;
                    
                    _status = Status::GEMM_Mac;
                    _gemm_mac_start_cycle = _total_cycles;
                    _sent_reqs = 0;
                    _num_reqs = int(_out_ch * _hidden_dim * 2 / ATOMIC_BYTES);
                }
            }
            break;
        case Status::GEMM_Mac:
            for (int ch = 0; ch < dram_channels *2; ch++) {
                if (_sent_reqs < _num_reqs) {
                    // Create and push read request for weight data
                    auto req = std::make_shared<MemoryAccess>();
                    req->req_type = duplex_enabled_ ? MemoryAccessType::DUPLEX_READ : MemoryAccessType::READ;
                    req->dram_address = _addr_w_it;
                    req->is_ndp = true;
                    req->size = ATOMIC_BYTES;
                    req->request = true;
                    req->start_cycle = 0;
                    req->finish_cycle = 0;
                    if (push_memory_request(req)) {
                        _addr_w_it += ATOMIC_BYTES;
                        _sent_reqs += (duplex_enabled_ ? DUPLEX_RETURNS : 1);
                        _gemm_mac_requests += (duplex_enabled_ ? DUPLEX_RETURNS : 1);
                    } else {
                        // Queue full - keep request for next cycle
                        break; // Try again next cycle
                    }
                } else if (!_memory_request_pending_queue.empty()) {
                    break; // Wait for pending requests to complete
                }

            }
            if (_memory_request_pending_queue.empty()) {
                // Update GEMM Mac statistics
                _gemm_mac_total_cycles += (_total_cycles - _gemm_mac_start_cycle);
                _gemm_mac_count++;
                _total_gemm_mac_requests += _gemm_mac_requests;
                _status = Status::Off;
            }
            break;
        case Status::Layernorm:
        case Status::Softmax:
        case Status::GELU:
        case Status::Add:
            if (_remaining_cycles > 0) {
                _remaining_cycles--;
            }
            if (_remaining_cycles == 0) {
                // Update statistics based on current status
                uint64_t instruction_cycles = _total_cycles - _instruction_start_cycle;
                switch (_status) {
                    case Status::Layernorm:
                        _layernorm_total_cycles += instruction_cycles;
                        _layernorm_count++;
                        break;
                    case Status::Softmax:
                        _softmax_total_cycles += instruction_cycles;
                        _softmax_count++;
                        break;
                    case Status::GELU:
                        _gelu_total_cycles += instruction_cycles;
                        _gelu_count++;
                        break;
                    case Status::Add:
                        _add_total_cycles += instruction_cycles;
                        _add_count++;
                        break;
                    default:
                        break;
                }
                // std::cout << "NDP_ctrl: Completed " << 
                //     (_status == Status::Layernorm ? "Layernorm" : 
                //     _status == Status::Softmax ? "Softmax" : 
                //     _status == Status::GELU ? "GELU" : 
                //     _status == Status::Add ? "Add" : "Unknown") 
                //     << " operation" << std::endl;
                _status = Status::Off;
            }
            break;
        case Status::Load:
            if (_finished_reqs >= _num_reqs) {
                // Update Load statistics
                _load_total_cycles += (_total_cycles - _instruction_start_cycle);
                _load_count++;
                _total_load_requests += _load_requests;
                _status = Status::Off;
            }
            for (int ch = 0; ch < dram_channels; ch++) {
                if (_sent_reqs < _num_reqs) {
                    // Create and push read request for loading data to buffer
                    auto req = std::make_shared<MemoryAccess>();
                    req->req_type = duplex_enabled_ ? MemoryAccessType::DUPLEX_READ : MemoryAccessType::READ;
                    req->dram_address = _addr_it;
                    req->is_ndp = true;
                    req->size = ATOMIC_BYTES;
                    req->request = true;
                    req->start_cycle = 0;
                    req->finish_cycle = 0;
                    if (push_memory_request(req)) {
                        _addr_it += ATOMIC_BYTES;
                        _sent_reqs += (duplex_enabled_ ? DUPLEX_RETURNS : 1);
                        _load_requests += (duplex_enabled_ ? DUPLEX_RETURNS : 1);
                    } else {
                        // Queue full - keep request for next cycle
                        break; // Try again next cycle
                    }
                }
            }
            break;
        case Status::Flush:
            if (_finished_reqs >= _num_reqs) {
                // Update Flush statistics
                _flush_total_cycles += (_total_cycles - _instruction_start_cycle);
                _flush_count++;
                _total_flush_requests += _flush_requests;
                _status = Status::Off;
            }
            for (int ch = 0; ch < dram_channels; ch++) {
                if (_sent_reqs < _num_reqs) {
                    // Create and push write request for flushing data from buffer
                    auto req = std::make_shared<MemoryAccess>();
                    req->req_type = MemoryAccessType::WRITE;
                    req->dram_address = _addr_it;
                    req->is_ndp = true;
                    req->size = ATOMIC_BYTES;
                    req->request = true;
                    req->start_cycle = 0;
                    req->finish_cycle = 0;
                    if (push_memory_request(req)) {
                        _addr_it += ATOMIC_BYTES;
                        _sent_reqs += (duplex_enabled_ ? DUPLEX_RETURNS : 1);
                        _flush_requests += (duplex_enabled_ ? DUPLEX_RETURNS : 1);
                    } else {
                        // Queue full - keep request for next cycle
                        break; // Try again next cycle
                    }
                }
            }
            break;
        default:
            break;
    }
}

// Memory request queue management functions
bool NDP_ctrl::push_memory_request(std::shared_ptr<MemoryAccess> request, uint32_t index) {
    if (_memory_request_queue.size() < MAX_MEMORY_REQUEST_QUEUE_SIZE) {
        _memory_request_queue.push(request);
        // Also add to pending queue to track outstanding requests
        _memory_request_pending_queue.push(request);
        return true;  // Success
    } else {
        return false; // Caller should retry later
    }
}

bool NDP_ctrl::has_memory_request(uint32_t index) const {
    return !_memory_request_queue.empty();
}

void NDP_ctrl::pop_memory_request(uint32_t index) {
    if (!_memory_request_queue.empty()) {
        _memory_request_queue.pop();
    }
}

MemoryAccess* NDP_ctrl::top_memory_request(uint32_t index) const {
    if (!_memory_request_queue.empty()) {
        return _memory_request_queue.front().get();
    }
    return nullptr;
}

std::shared_ptr<MemoryAccess> NDP_ctrl::top_memory_request_shared(uint32_t index) const {
    if (!_memory_request_queue.empty()) {
        return _memory_request_queue.front();
    }
    return nullptr;
}

bool NDP_ctrl::running() const {
    return _status != Status::Off;
}

bool NDP_ctrl::is_memory_request_queue_full() const {
    return _memory_request_queue.size() >= MAX_MEMORY_REQUEST_QUEUE_SIZE;
}

void NDP_ctrl::print_instruction_stats() {
    std::cout << "\n=== NDP_ctrl[" << _id << "] Instruction Statistics (Total Cycles: " << _total_cycles << ") ===" << std::endl;
    
    // Overall statistics
    uint64_t total_requests = _total_gemm_load_requests + _total_gemm_mac_requests + _total_load_requests + _total_flush_requests;
    uint64_t total_responses = _total_gemm_load_responses + _total_gemm_mac_responses + _total_load_responses + _total_flush_responses;
    if (total_requests > 0 && _total_cycles > 0) {
        double req_bw_util = (total_requests * ATOMIC_BYTES) / (double)_total_cycles; // GB/s
        double resp_bw_util = (total_responses * ATOMIC_BYTES) / (double)_total_cycles; // GB/s
        std::cout << "Overall: Total Requests=" << total_requests << ", Total Responses=" << total_responses
                  << ", Req BW Util=" << std::fixed << std::setprecision(3) << req_bw_util << " GB/s"
                  << ", Resp BW Util=" << std::fixed << std::setprecision(3) << resp_bw_util << " GB/s" << std::endl;
    }
    std::cout << std::endl;
    
    // GEMM Load statistics
    if (_gemm_load_count > 0) {
        double avg_cycles = (double)_gemm_load_total_cycles / _gemm_load_count;
        double req_bw_util = (_total_gemm_load_requests * ATOMIC_BYTES) / (double)_gemm_load_total_cycles; // GB/s
        double resp_bw_util = (_total_gemm_load_responses * ATOMIC_BYTES) / (double)_gemm_load_total_cycles; // GB/s
        std::cout << "GEMM_Load: Count=" << _gemm_load_count 
                  << ", Total Cycles=" << _gemm_load_total_cycles 
                  << ", Avg Cycles=" << (uint64_t)avg_cycles
                  << ", Requests=" << _total_gemm_load_requests << ", Responses=" << _total_gemm_load_responses
                  << ", Req BW=" << std::fixed << std::setprecision(3) << req_bw_util << " GB/s"
                  << ", Resp BW=" << std::fixed << std::setprecision(3) << resp_bw_util << " GB/s" << std::endl;
    }
    
    // GEMM Mac statistics
    if (_gemm_mac_count > 0) {
        double avg_cycles = (double)_gemm_mac_total_cycles / _gemm_mac_count;
        double req_bw_util = (_total_gemm_mac_requests * ATOMIC_BYTES) / (double)_gemm_mac_total_cycles; // GB/s
        double resp_bw_util = (_total_gemm_mac_responses * ATOMIC_BYTES) / (double)_gemm_mac_total_cycles; // GB/s
        std::cout << "GEMM_Mac: Count=" << _gemm_mac_count 
                  << ", Total Cycles=" << _gemm_mac_total_cycles 
                  << ", Avg Cycles=" << (uint64_t)avg_cycles
                  << ", Requests=" << _total_gemm_mac_requests << ", Responses=" << _total_gemm_mac_responses
                  << ", Req BW=" << std::fixed << std::setprecision(3) << req_bw_util << " GB/s"
                  << ", Resp BW=" << std::fixed << std::setprecision(3) << resp_bw_util << " GB/s" << std::endl;
    }
    
    if (_layernorm_count > 0) {
        std::cout << "Layernorm: Count=" << _layernorm_count 
                  << ", Total Cycles=" << _layernorm_total_cycles 
                  << ", Avg Cycles=" << (_layernorm_total_cycles / _layernorm_count) << std::endl;
    }
    
    if (_softmax_count > 0) {
        std::cout << "Softmax: Count=" << _softmax_count 
                  << ", Total Cycles=" << _softmax_total_cycles 
                  << ", Avg Cycles=" << (_softmax_total_cycles / _softmax_count) << std::endl;
    }
    
    if (_gelu_count > 0) {
        std::cout << "GELU: Count=" << _gelu_count 
                  << ", Total Cycles=" << _gelu_total_cycles 
                  << ", Avg Cycles=" << (_gelu_total_cycles / _gelu_count) << std::endl;
    }
    
    if (_add_count > 0) {
        std::cout << "Add: Count=" << _add_count 
                  << ", Total Cycles=" << _add_total_cycles 
                  << ", Avg Cycles=" << (_add_total_cycles / _add_count) << std::endl;
    }
    
    if (_load_count > 0) {
        double req_bw_util = (_total_load_requests * ATOMIC_BYTES) / (double)_load_total_cycles; // GB/s
        double resp_bw_util = (_total_load_responses * ATOMIC_BYTES) / (double)_load_total_cycles; // GB/s
        std::cout << "Load: Count=" << _load_count 
                  << ", Total Cycles=" << _load_total_cycles 
                  << ", Avg Cycles=" << (_load_total_cycles / _load_count)
                  << ", Requests=" << _total_load_requests << ", Responses=" << _total_load_responses
                  << ", Req BW=" << std::fixed << std::setprecision(3) << req_bw_util << " GB/s"
                  << ", Resp BW=" << std::fixed << std::setprecision(3) << resp_bw_util << " GB/s" << std::endl;
    }
    
    if (_flush_count > 0) {
        double req_bw_util = (_total_flush_requests * ATOMIC_BYTES) / (double)_flush_total_cycles; // GB/s
        double resp_bw_util = (_total_flush_responses * ATOMIC_BYTES) / (double)_flush_total_cycles; // GB/s
        std::cout << "Flush: Count=" << _flush_count 
                  << ", Total Cycles=" << _flush_total_cycles 
                  << ", Avg Cycles=" << (_flush_total_cycles / _flush_count)
                  << ", Requests=" << _total_flush_requests << ", Responses=" << _total_flush_responses
                  << ", Req BW=" << std::fixed << std::setprecision(3) << req_bw_util << " GB/s"
                  << ", Resp BW=" << std::fixed << std::setprecision(3) << resp_bw_util << " GB/s" << std::endl;
    }
    
    // Off status statistics
    uint64_t current_off_cycles = _off_total_cycles;
    if (_status == Status::Off && _off_start_cycle > 0) {
        current_off_cycles += (_total_cycles - _off_start_cycle);
    }
    
    if (current_off_cycles > 0) {
        double off_percentage = (current_off_cycles * 100.0) / _total_cycles;
        std::cout << "Off Status: Total Cycles=" << current_off_cycles
                  << ", Percentage=" << std::fixed << std::setprecision(2) << off_percentage << "%"
                  << " (idle cycles)" << std::endl;
    }
    
    std::cout << "===============================================\n" << std::endl;
}

