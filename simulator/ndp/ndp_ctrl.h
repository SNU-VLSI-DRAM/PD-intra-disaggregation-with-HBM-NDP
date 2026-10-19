#pragma once

#include <list>
#include <memory>
#include <vector>
#include <queue>
#include "ndp/PU.h"
#include "common/Common.h"

// Instruction signatures for each Opcode
// GEMM: preloaded, batch_size, hidden_dim, out_ch, addr_in, addr_w, addr_out
// Layernorm: vector_size
// Softmax: vector_size
// Gelu: vector_size
// Add: vector_size
// FlushOutBuffer: address, data_size
// LoadBuffer: address, data_size
struct Instruction {
    Opcode opcode;

    // GEMM
    bool preloaded = false; // whether inputs are preloaded or not
    uint32_t batch_size = 0;
    uint32_t hidden_dim = 0;
    uint32_t out_ch = 0;
    addr_type addr_in = 0;
    addr_type addr_w = 0;
    addr_type addr_out = 0;

    // Layernorm, Softmax, Gelu, Add
    uint32_t vector_size = 0;

    // FlushOutBuffer, LoadBuffer
    addr_type address = 0;
    uint32_t data_size = 0;

    // GPUComm
    bool is_gpu_comm = false; // whether it's GPU communication or not
    cycle_type comm_cycles = 0; // communication cycles for GPUComm

    Instruction(Opcode op) : opcode(op) {}
};

class NDP_ctrl {
public:
    int get_buf_size() const { return _buf_size; }
    enum class Status {
        Off,
        GEMM_Load,
        GEMM_Mac,
        Layernorm,
        Softmax,
        GELU,
        Add,
        Flush,
        Load
    };
    uint64_t current_step = 0;
    bool gpu_comm_waiting = false;
    cycle_type gpu_comm_cycles = 0;
    cycle_type ndp_decoding_interval = 0;
    cycle_type prev_ndp_decoding_interval = 0;
    cycle_type last_step_cycle = 0;
    cycle_type average_decoding_cycles = 0;

    NDP_ctrl(uint32_t id, int buf_size, int vu_width, int pu_lanes, int pu_buf_size, int lane_width = 1024);
    void set_duplex(bool enabled) { duplex_enabled_ = enabled; }
    void set_duplex_bg_pos(uint32_t bg_pos) { duplex_bg_pos_ = bg_pos; }
    bool issue_instruction(const Instruction& inst);  // Returns true if instruction was issued successfully
    void push_memory_response(MemoryAccess* resp);
    void cycle();

    bool running() const;
    bool has_memory_request(uint32_t index) const;
    void pop_memory_request(uint32_t index);
    MemoryAccess* top_memory_request(uint32_t index) const;
    std::shared_ptr<MemoryAccess> top_memory_request_shared(uint32_t index) const;
    bool push_memory_request(std::shared_ptr<MemoryAccess> request, uint32_t index = 0);  // Returns false if queue is full
    bool is_memory_request_queue_full() const;

protected:
    const uint32_t _id;
    // const SimulationConfig _config; // Removed due to incomplete type

    Status _status = Status::Off;

    // Tile management
    std::deque<std::shared_ptr<Instruction>> _tiles;
    std::queue<std::shared_ptr<Instruction>> _finished_tiles;

    // Memory request queue
    static constexpr size_t MAX_MEMORY_REQUEST_QUEUE_SIZE = 256;
    std::queue<std::shared_ptr<MemoryAccess>> _memory_request_queue;
    std::queue<std::shared_ptr<MemoryAccess>> _memory_request_pending_queue;
    uint32_t _waiting_write_reqs = 0;

    // GEMM variables
    int _batch_size = 0;
    int _hidden_dim = 0;
    int _out_ch = 0;
    addr_type _addr_in = 0;
    addr_type _addr_w = 0;
    addr_type _addr_out = 0;

    // Buffer
    int _buf_size = 0;

    // FSM internal
    int _num_reqs_gemm_load = 0;
    int _num_reqs = 0;
    int _sent_reqs = 0;
    int _finished_reqs = 0;
    addr_type _addr_in_it = 0;
    addr_type _addr_w_it = 0;
    addr_type _addr_w_it_tb = 0;
    addr_type _addr_it = 0;
    int _data_it = 0;
    int _data_size = 0;
    int _remaining_cycles = 0;

    // Instruction cycle statistics
    uint64_t _total_cycles = 0;
    uint64_t _instruction_start_cycle = 0;
    
    // GEMM separate Load/Mac statistics
    uint64_t _gemm_load_total_cycles = 0;
    uint64_t _gemm_mac_total_cycles = 0;
    uint64_t _gemm_load_requests = 0;
    uint64_t _gemm_mac_requests = 0;
    uint32_t _gemm_load_count = 0;
    uint32_t _gemm_mac_count = 0;
    uint64_t _gemm_load_start_cycle = 0;
    uint64_t _gemm_mac_start_cycle = 0;
    
    // Other instruction statistics
    uint64_t _layernorm_total_cycles = 0;
    uint64_t _softmax_total_cycles = 0;
    uint64_t _gelu_total_cycles = 0;
    uint64_t _add_total_cycles = 0;
    uint64_t _load_total_cycles = 0;
    uint64_t _flush_total_cycles = 0;
    uint64_t _load_requests = 0;
    uint64_t _flush_requests = 0;
    uint32_t _layernorm_count = 0;
    uint32_t _softmax_count = 0;
    uint32_t _gelu_count = 0;
    uint32_t _add_count = 0;
    uint32_t _load_count = 0;
    uint32_t _flush_count = 0;
    
    uint64_t _last_print_cycle = 0;
    static constexpr uint64_t PRINT_INTERVAL = 10000;
    
    // Cumulative statistics
    uint64_t _total_gemm_load_requests = 0;
    uint64_t _total_gemm_mac_requests = 0;
    uint64_t _total_load_requests = 0;
    uint64_t _total_flush_requests = 0;
    
    // Response statistics
    uint64_t _total_gemm_load_responses = 0;
    uint64_t _total_gemm_mac_responses = 0;
    uint64_t _total_load_responses = 0;
    uint64_t _total_flush_responses = 0;
    
    // Off status statistics
    uint64_t _off_total_cycles = 0;
    uint64_t _off_start_cycle = 0;

    void print_instruction_stats();

    // VU, PU
    int VU_width; // in elements
    PU pu;

    // Duplex
    bool duplex_enabled_ = false;
    // When duplex is enabled, one DUPLEX_READ is modeled as operating on 4 bankgroups
    // “simultaneously”. To make that meaningful, we must generate addresses that differ
    // only in *bankgroup bits*.
    //
    // We currently assume a 2-bit bankgroup field (4 bankgroups). We avoid relying on a
    // fixed byte stride (e.g., +4KB), since mapping is config-dependent.
    static constexpr uint32_t DUPLEX_BG_BITS = 2;
    static constexpr addr_type DUPLEX_BG_MASK = (static_cast<addr_type>(1) << DUPLEX_BG_BITS) - 1;
    // Conservative defaults for HBM-style configs where bankgroup bits are often around
    // the 12-bit region. These can be overridden later by wiring actual mapping in.
    static constexpr uint32_t DEFAULT_BG_POS = 12;
    uint32_t duplex_bg_pos_ = DEFAULT_BG_POS;

    inline addr_type skip_bankgroup_carry(addr_type base, addr_type atomic_bytes, uint32_t bg_pos) const {
        // We want the “normal” stream to advance in ATOMIC_BYTES, but whenever
        // that increment would naturally carry into the bankgroup field, we
        // *skip over* the entire bankgroup field so that duplex can own BG
        // selection (0..3) without being disturbed by the base iterator.
        const addr_type bg_lsb = (static_cast<addr_type>(1) << bg_pos);
        // boundary where low bits [0..bg_pos-1] wrap
        if (((base & (bg_lsb - 1)) + atomic_bytes) >= bg_lsb) {
            return base + bg_lsb;
        }
        return base;
    }
};










