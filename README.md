# PD intra-disaggregation with HBM-NDP

A cycle-level model of an LLM serving node where a GPU and near-data processing
units under the same HBM split the work of a decoder layer, plus the queueing
model that turns its latencies into serving throughput.

- `simulator/` — the node model: a GPU, 8 NDP units per package, and the
  memory system they share. See [simulator/README.md](simulator/README.md)
  for the layout, the clock domains, the communication cost model and the flags
  that decide how the work is divided.
- `qps_estimator/` — a queueing estimator that takes the per-layer cycle counts
  the simulator produces (`qps_estimator/cycle_info/`) and reports sustainable
  request rates. See [qps_estimator/README.md](qps_estimator/README.md).
- `scripts/run_test.sh` — reproduces one workload point for each model: batch 16
  at a 16K context, tp=8 pp=1, with the GPU prefilling alongside the NDP and
  then decoding on its own. It runs for hours.

Build and run:

```
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make ndp_sim -j

./simulator/ndp_sim --model LLAMA2-70B --phase decode --no-ndp \
    --tp 8 --pp 1 --batch 4 --context-len 2048 --decode-steps 1
```

Run it from `build/`: the simulator opens its DRAM timings through
`../configs/*.ini`. That takes a minute or so.

`../scripts/run_test.sh` reproduces a full workload point and takes hours.

`./simulator/ndp_sim --help` lists every option with its default.

## Built on

The memory system in this repository is **NewtonSim**, taken from the NeuPIMs
project
([casys-kaist/NeuPIMs, extern/NewtonSim](https://github.com/casys-kaist/NeuPIMs/tree/main/extern/NewtonSim)).
NewtonSim adds Newton processing-in-memory commands to
[DRAMsim3](https://github.com/umd-memsys/DRAMsim3). What comes from it:

- `src/NewtonSim.cc`, `src/newton_controller.{cc,h}`,
  `src/newton_command_queue.{cc,h}`, `include/newtonsim/NewtonSim.h` — the
  Newton command path
- `src/neupims_controller.{cc,h}`, `src/neupims_command_queue.{cc,h}` — the
  NeuPIMs controller, which the `--pim` mode drives
- `configs/*_neupims.ini` — the DRAM models those use

What this repository adds is the GPU and NDP models under `simulator/` and the
queueing model under `qps_estimator/`.

The upstream work:

- G. Heo, S. Lee, J. Cho, H. Choi, S. Lee, H. Ham, G. Kim, D. Mahajan and
  J. Park, "NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM
  Inferencing," in Proceedings of the 29th ACM International Conference on
  Architectural Support for Programming Languages and Operating Systems
  (ASPLOS), 2024.
  [ACM DL](https://dl.acm.org/doi/10.1145/3620666.3651380)
- S. Li, Z. Yang, D. Reddy, A. Srivastava and B. Jacob, "DRAMsim3: a
  Cycle-accurate, Thermal-Capable DRAM Simulator," IEEE Computer Architecture
  Letters, 2020. [Link](https://ieeexplore.ieee.org/document/8999595)

DRAMsim3's own documentation -- its config-file format, trace drivers, thermal
module and gem5 integration -- lives in
[its repository](https://github.com/umd-memsys/DRAMsim3), and covers `src/`,
`configs/`, `tests/` and `ext/` here.

## DRAM configurations

`configs/` holds the DRAM timing files. The ones this work uses are

| file | used by |
|---|---|
| `HBM4_8Gb_x128.ini` | the default |
| `HBM4_8Gb_x128_duplex.ini` | `--duplex`, a 256 B bus instead of 64 B |
| `HBM4_8Gb_x128_neupims.ini` | `--pim`, which adds `pim_type = SINGLE` |
| `HBM3e_8Gb_x128.ini` | `--bullet` and `--bullet-hw`, at 16 channels |

The simulator picks one from the mode flags rather than from a command-line
path, so a mode and its channel count stay in step. See
`simulator/config/ConfigBuilder.cc`.

## Code structure

### `simulator/` — the node model

| path | what it is |
|---|---|
| `main.cc` | parse the command line, assemble the configuration, build the NDP schedule, run |
| `common/Types.h` | primitive types and the structures that cross the GPU / NDP / DRAM boundary: `MemoryAccess`, `Layer`, `Opcode` |
| `common/Common.h` | umbrella include; new code should prefer the narrower headers |
| `config/CliOptions.{h,cc}` | the command line: what each flag means, and the combinations that are rejected |
| `config/SimConfig.h` | `SimulationConfig`, everything the models read, grouped by concern |
| `config/InterconnectModel.h` | the collective cost model shared by both devices: one-shot, two-shot and ring hop counts |
| `config/ModelConfig.h`, `config/ModelZoo.{h,cc}` | one model's dimensions, and the table of models |
| `config/NDPConfig.h`, `config/NdpConfigFile.{h,cc}` | the NDP's hardware parameters, and the JSON reader for them |
| `config/ConfigBuilder.{h,cc}` | turns the parsed command line into the two config structs, and lays out the address map |
| `gpu/GPU.h` | the GPU model's interface: tiles, layer descriptors, the cycle loop |
| `gpu/gpu_model.cc` | construction: config into the derived rates the model runs on |
| `gpu/gpu_layers.cc` | a model into the ordered layer list one decoder layer runs, for prefill, chunked prefill and decode, dense and MoE |
| `gpu/gpu_tiling.cc` | picks an L2-sized `(Mb, Nb, Kb)` blocking per layer and expands it into tiles with their address ranges |
| `gpu/gpu_exec.cc` | the execution loop: launch tiles, issue their memory traffic, apply the communication cost model, retire on DRAM responses |
| `ndp/PU.{h,cc}` | one NDP unit's datapath: lanes and the timing buffer that gates issue |
| `ndp/ndp_ctrl.{h,cc}` | one NDP unit's control FSM, and its memory request and response handling |
| `ndp/NDP_scheduler.{h,cc}` | lowers layers into instructions, tiles the GEMMs, and issues them |
| `ndp/ndp_layer_builder.{h,cc}` | a model into the NDP's layer schedule; `attn_moe_only` narrows it to attention and the MoE experts |
| `sim/Simulator.{h,cc}` | the top-level loop: clocks the devices, arbitrates their DRAM traffic, decides when the run ends |
| `sim/Dram.{h,cc}` | the wrapper around the memory system |
| `ndp_config.json` | the NDP hardware parameters read at startup |
| `CMakeLists.txt` | the build, with sources grouped the way the tree is |

### `qps_estimator/` — serving throughput

| path | what it is |
|---|---|
| `scripts/queueing_estimator.py` | discrete-event estimator for Poisson arrivals with dynamic prefill batching; reports TTFT, TBT, TPOT and end-to-end latency |
| `scripts/sweep_qps_queueing_estimator.py` | runs the estimator across a QPS range and one or more calibration files, and writes a summary CSV |
| `scripts/plot_sweep_summary.py` | plots percentile curves from that CSV |
| `cycle_info/<model>/<config>.json` | the per-layer cycle counts the estimator takes as input, one file per arrangement |

### `src/`, `include/`, `configs/`, `tests/`, `ext/` — the memory system

NewtonSim, and DRAMsim3 underneath it: the DRAM timing model, its controllers
and command queues, the Newton and NeuPIMs processing-in-memory command paths,
and the `.ini` files that describe each part. The simulator reaches it through
`sim/Dram.{h,cc}` only. See [Built on](#built-on).

