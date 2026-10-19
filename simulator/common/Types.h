#pragma once

// Primitive types and the data structures that cross the GPU / NDP / DRAM
// boundary. Anything with a knob in it belongs in config/, not here.

#include <cstdint>
#include <memory>
#include <string>

#define SPAD_BASE 0x10000000
#define ACCUM_SPAD_BASE 0x20000000
#define GARBAGE_ADDR 0xFFFFFFFFFFFFFFF
#define KB *1024

#define PAGE_SIZE 4096
#define ADDR_ALIGN 64

template <typename T>
using Ptr = std::shared_ptr<T>;

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

// Must match src/common.h TransactionType ordering.
enum class MemoryAccessType {
    READ,
    DUPLEX_READ,
    WRITE,
    GWRITE,
    COMP,
    READRES,
    P_HEADER,
    COMPS_READRES,
    PWRITE,
    SIZE
};

enum class Opcode {
    GEMM,
    Layernorm,
    Softmax,
    Gelu,
    Add,
    FlushOutBuffer,
    LoadBuffer,
    GPUComm,
    FinishStep
};

typedef struct MemoryAccess {
    addr_type dram_address;
    bool is_ndp;
    uint64_t size;
    MemoryAccessType req_type;
    // PIM header metadata used when req_type == P_HEADER
    int num_comps = 1;
    int num_readres = 1;
    bool p_header_for_gwrite = false;
    bool request;
    cycle_type start_cycle;
    cycle_type finish_cycle;
    int gpu_tag = 0; // 0=default/single GPU, 1=prefill GPU, 2=decode GPU (bullet mode)
} MemoryAccess;

// One step of the NDP instruction stream, as emitted by NDPLayerBuilder and
// lowered to instructions by NDP_scheduler.
struct Layer {
    std::string layer_name;
    uint32_t batch_size = 0;
    uint32_t hidden_dim = 0;
    uint32_t out_ch = 0;
    uint32_t addr_in = 0;
    uint32_t addr_w = 0;
    uint32_t addr_out = 0;
    uint32_t vector_size = 0;
    uint32_t address = 0;
    uint32_t data_size = 0;
    // Link traversals this layer is charged for (0 = no link latency).
    // "Comm" layers use the intra-package NDP-to-NDP latency, "AllReduce" and
    // "PipelineComm" layers use the inter-GPU NVLink latency. A collective that
    // the builder emits as several layers tags only ONE of them so the latency
    // is paid once for the whole collective.
    uint32_t comm_hops = 0;
};
