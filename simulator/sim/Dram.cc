#include "sim/Dram.h"

DRAM::DRAM(SimulationConfig config)
    : _mem(std::make_unique<dramsim3::NewtonSim>(config.dram_config_path, config.log_dir)) {

    _total_processed_requests.resize(config.dram_channels);
    _processed_requests.resize(config.dram_channels);

    for (int ch = 0; ch < config.dram_channels; ch++) {
        _total_processed_requests[ch] = 0;
        _processed_requests[ch] = 0;
    }

    _config = config;
    _cycles = 0;
    _burst_cycle = _mem->GetBurstCycles(); // double data rate
}

bool DRAM::running() { return false; }

void DRAM::cycle() {

    _mem->ClockTick();

    _cycles++;
    int interval = 1000000;
    if (_cycles % interval == 0) {
        for (int ch = 0; ch < _config.dram_channels; ch++) {
            _total_processed_requests[ch] += _processed_requests[ch];
            _processed_requests[ch] = 0;
        }
    }
}

bool DRAM::is_full(uint32_t cid, MemoryAccess *request) {
    bool full = !_mem->WillAcceptTransaction(request->dram_address, int(request->req_type));

    return full;
}

bool DRAM::push(uint32_t cid, MemoryAccess *request) {

    uint32_t mem_ch = get_channel_id(request);
    assert(mem_ch == cid);

    const addr_type atomic_bytes = _mem->GetBurstLength() * _mem->GetBusBits() / 8;
    // const addr_type atomic_bytes =
    //     _mem->GetBurstLength() * _mem->GetBusBits() / 8;
    const addr_type target_addr = request->dram_address;
    // align address
    const addr_type start_addr = target_addr - (target_addr % atomic_bytes);


    int count = 0;
    bool accepted = _mem->AddTransaction(target_addr, int(request->req_type), request);
    if (accepted) {
        request->request = false;
        _mem_req_cnt++;
    }
    return accepted;
}

bool DRAM::is_empty(uint32_t cid) {
    return _mem->IsEmpty(cid);
}

MemoryAccess *DRAM::top(uint32_t cid) {
    assert(!is_empty(cid));
    return (MemoryAccess *)_mem->Top(cid);
}

void DRAM::pop(uint32_t cid) {
    assert(!is_empty(cid));
    _mem->Pop(cid);
    _processed_requests[cid]++;
}

uint32_t DRAM::get_channel_id(MemoryAccess *access) {
    return _mem->GetChannel(access->dram_address);
}

void DRAM::print_stat() {
    uint32_t total_reqs = 0;
    for (int ch = 0; ch < _config.dram_channels; ch++)
    {
        uint32_t ch_reqs = _total_processed_requests[ch] + _processed_requests[ch];
        total_reqs += ch_reqs;
    }
    float util = ((float)total_reqs * _burst_cycle / _config.dram_channels) / _cycles * 100;
    std::cout << "DRAM total: reqs=" << total_reqs << " cycles=" << _cycles
              << " burst_cycle=" << _burst_cycle << " avg_util=" << util << "%" << std::endl;
    _mem->PrintStats();
}