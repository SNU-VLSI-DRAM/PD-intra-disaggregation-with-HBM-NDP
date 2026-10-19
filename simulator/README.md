# Simulator

A cycle-level model of an LLM serving node: a GPU model, an NDP (near-data
processing) model sitting under the same HBM, and a DRAMSim3-backed memory
system, driven from one configuration.

## Layout

```
simulator/
├── main.cc        entry point: parse, assemble config, build layers, run
├── common/
│   ├── Types.h    primitive types, MemoryAccess, Layer, Opcode
│   └── Common.h   umbrella header (prefer the narrow ones in new code)
├── config/
│   ├── CliOptions.*        command-line parsing (named flags)
│   ├── ModelConfig.h       one model's dimensions
│   ├── ModelZoo.*          the model table
│   ├── NDPConfig.h         NDP hardware parameters
│   ├── NdpConfigFile.*     reads the NDP hardware JSON
│   ├── SimConfig.h         SimulationConfig: everything the models read
│   ├── InterconnectModel.h collective cost model (see below)
│   └── ConfigBuilder.*     address map + config assembly
├── gpu/
│   ├── GPU.h
│   ├── gpu_model.cc        construction: config -> derived rates
│   ├── gpu_layers.cc       model -> ordered layer list (prefill/decode, dense/MoE)
│   ├── gpu_tiling.cc       layer -> L2-sized tiles with address ranges
│   └── gpu_exec.cc         launch, memory traffic, retirement
├── ndp/
│   ├── PU.*                the vector/MAC datapath
│   ├── ndp_ctrl.*          one NDP unit's control FSM
│   ├── NDP_scheduler.*     layers -> instructions, issue
│   └── ndp_layer_builder.* model -> NDP layer schedule
└── sim/
    ├── Simulator.*         the top-level loop, GPU/NDP/DRAM arbitration
    └── Dram.*              DRAMSim3 wrapper
```

## Clock domains

`sim_cycles` runs at the DRAM clock (`tCK = 0.5 ns`). The GPU and the NDP both
tick every 2 DRAM cycles, so **1 GPU cycle = 1 NDP cycle = 1 ns**. Every latency
knob is therefore written in nanoseconds and used directly as a cycle count.

## Communication cost model

Collectives are charged with the alpha-beta model:

```
T = hops * link_latency + volume / bandwidth
```

The hop count comes from `collective_latency_hops()` in
`config/InterconnectModel.h` and is shared by the GPU and the NDP:

| `--collective` | hops for an all-reduce | models                                        |
|----------------|------------------------|-----------------------------------------------|
| `one_shot`     | 1                      | flat 1-hop fabric with in-switch reduction, i.e. NVLink SHARP (NVLS) on a GB200 NVL72, where every GPU is one NVSwitch hop away and the switch reduces and multicasts. Default. |
| `two_shot`     | 2                      | reduce-scatter + all-gather, reduction on the GPUs |
| `ring`         | `2*(tp-1)`             | NCCL's ring fallback; latency-dominated at decode message sizes |

Two link latencies exist:

- `--nvlink-latency` (default **800 ns**): one inter-GPU NVLink5 hop. Charged to
  TP all-reduces, all-to-all, reduce-scatter and pipeline sends.
- `--sp-latency` (default **400 ns**): one NDP-to-NDP hop inside a GPU package.
  Charged once per decoder layer, to the sequence-parallel exchange that
  combines the partial attention outputs held by the 8 NDP units.

A collective that a builder emits as several steps tags only the first step with
the hop count, so the latency is paid once per collective, not once per step.

## Running

```
./ndp_sim --model Qwen-235B --phase prefill --tp 8 --sp 8 \
          --batch 16 --context-len 16384 --decode-steps 1024 \
          --nvlink-latency 800 --sp-latency 400 --collective one_shot
```

`--help` lists every option with its default. The simulator reads
`../configs/*.ini` for DRAM timings, so run it from the build directory.

### How the work is divided

By default the GPU runs everything and, with `--ndp`, the NDP decodes while the
GPU prefills. Three flags change that split:

| flag | effect |
|---|---|
| `--ndp-attn-moe-only` | the NDP runs only attention and the MoE experts, leaving the rest of the decoder to the GPU |
| `--no-gpu` | the GPU is not simulated and the NDP decides when the run ends |
| `--duplex` | the NDP reaches memory over a 256 B bus: the DRAM model becomes `HBM4_8Gb_x128_duplex.ini`, the NDP issues `DUPLEX_READ`, and its PU is reshaped from 16 x 1024 to 4 x 4096 lanes to consume the wider returns. It does not change *what* the NDP runs. `--duplex-4x-compute` additionally gives the PU 4x the lanes, 16 x 4096 |

So `--ndp --duplex` is the whole decoder layer over a duplex bus with a GPU
prefilling alongside, while `--duplex --ndp-attn-moe-only --no-gpu` measures the
NDP's attention and expert half on its own.

### PIM

`--pim` routes the GPU's attention to PIM commands.
`--pim-parallel-level` (default 16) sets how many GEMV units per channel one
COMP drives, and so how much data it covers:
`bytes_per_comp = atomic_bytes * level`. It is the main lever on how optimistic
the PIM model is. Nothing else in the simulator reads it.
