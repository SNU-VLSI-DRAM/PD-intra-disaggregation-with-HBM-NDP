#pragma once

#include <cstdint>

#include "config/InterconnectModel.h"

// Hardware parameters of a single NDP unit plus the address regions it owns.
struct NDPConfig {
    // ---- Processing unit ----
    int num_lanes_of_PU = 16;
    int lane_width = 1024;
    int buffer_size = 4096;
    int vu_width = 1024;
    int timing_buffer_size = 1024*16;

    // ---- Interconnect (mirrors the matching SimulationConfig fields) ----
    // The NDP is clocked at 1 cycle == 1 ns, so every latency here is in ns.
    long double interconnect_bw = 1800.0L; // GB/s (== bytes/ns)
    // Latency of one inter-GPU NVLink5 hop (NVSwitch traversal in an NVL72
    // domain). Charged to GPU-level collectives: AllReduce, PipelineComm.
    long double nvlink_latency_ns = 800.0L;
    // Latency of one NDP-to-NDP hop inside the GPU package. Charged to the
    // sequence-parallel exchange between the 8 NDP units ("Comm" layers).
    long double sp_latency_ns = 400.0L;
    // Latency model for GPU-level collectives. See collective_latency_hops().
    int collective_algo = (int)CollectiveAlgo::OneShot;

    // ---- Address map ----
    uint64_t input_addr = 0x00000000;   // Input data address space
    uint64_t output_addr = 0x20000000;  // Output data address space
    uint64_t weight_addr = 0x30000000;  // Weight data address space
    uint64_t kv_addr = 0x40000000;      // Key-Value cache address space
};
