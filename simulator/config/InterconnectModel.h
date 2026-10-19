#pragma once

#include <cstdint>

// Cost model for the links between devices.
//
// The GPU and the NDP are both clocked at 1 cycle == 1 ns in this simulator
// (DRAM tCK = 0.5 ns and both devices tick every 2 DRAM cycles), so every
// latency here is expressed directly in nanoseconds and used as a cycle count.
//
// Communication is charged with the usual alpha-beta model:
//
//     T = hops * link_latency + volume / bandwidth
//
// The bandwidth (beta) term was always present; collective_latency_hops()
// supplies the hop count for the latency (alpha) term.

// Collective algorithms, matching SimulationConfig::collective_algo and
// NDPConfig::collective_algo.
enum class CollectiveAlgo {
    OneShot = 0,  // a single link traversal per collective (default)
    TwoShot = 1,  // reduce-scatter + all-gather, two serialized traversals
    Ring    = 2,  // 2*(ranks-1) serialized steps for an all-reduce
};

// Number of serialized link traversals charged to one collective that spans
// `ranks` participants.
//
//   OneShot: the whole collective costs a single link traversal. This is the
//       right model for a flat 1-hop fabric that reduces in the network, i.e.
//       NVLink SHARP (NVLS) on a GB200 NVL72: every rank is one NVSwitch hop
//       away, the switch performs the reduction and multicasts the result back,
//       so the serialized latency does not grow with `ranks`.
//   TwoShot: reduce-scatter followed by all-gather with the reduction done on
//       the GPUs. Still 1 hop per phase on a flat fabric, but the two phases
//       are serialized, so an all-reduce pays 2 traversals.
//   Ring: the bandwidth-optimal algorithm NCCL falls back to when NVLS is
//       unavailable. An all-reduce is 2*(ranks-1) serialized steps, so it is
//       strongly latency-bound for the small messages decode produces.
//
// Non-reducing collectives (all-gather, all-to-all, point-to-point pipeline
// sends) are charged half as many hops as an all-reduce under Ring, and a
// single hop otherwise, because they are one-directional.
inline uint32_t collective_latency_hops(int algo, int ranks, bool is_allreduce) {
    if (ranks <= 1) return 0;
    switch (static_cast<CollectiveAlgo>(algo)) {
        case CollectiveAlgo::Ring:
            return is_allreduce ? 2u * (uint32_t)(ranks - 1) : (uint32_t)(ranks - 1);
        case CollectiveAlgo::TwoShot:
            return is_allreduce ? 2u : 1u;
        case CollectiveAlgo::OneShot:
        default:
            return 1u;
    }
}
