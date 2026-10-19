#ifndef DRAM_H
#define DRAM_H

#include <queue>
#include <utility>

#include "common/Common.h"
#include "newtonsim/NewtonSim.h"

class DRAM {
    public:
        DRAM(SimulationConfig config);
    
        bool running();
        void cycle();
        bool is_full(uint32_t cid, MemoryAccess *request);
        bool push(uint32_t cid, MemoryAccess *request);
        bool is_empty(uint32_t cid);
        MemoryAccess *top(uint32_t cid);
        void pop(uint32_t cid);
        uint32_t get_channel_id(MemoryAccess *request);
        void print_stat();
        addr_type get_addr_align() { return _addr_align; }

        std::unique_ptr<dramsim3::NewtonSim> _mem;
        std::vector<uint64_t> _total_processed_requests;
        std::vector<uint64_t> _processed_requests;
        int _mem_req_cnt = 0;
        int _burst_cycle;
        
    protected:
        // Todo. discuss where to define simulation configs
        SimulationConfig _config;
        uint32_t _n_ch;
        cycle_type _cycles;
        addr_type _addr_align;
};
#endif