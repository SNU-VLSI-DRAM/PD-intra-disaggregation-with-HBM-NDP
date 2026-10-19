#include "ndp/NDP_scheduler.h"
#include <algorithm>
#include <iostream>

NDP_scheduler::NDP_scheduler(NDP_ctrl& ndp_ctrl, const NDPConfig& config) 
    : ndp(ndp_ctrl), ndp_config(config), ndp_done(false) {
}

void NDP_scheduler::load_layers(const std::vector<Layer>& layers) {
    instructions.clear();
    bool use_input_addr = true;  // Alternating between input and output address spaces
    
        
    for (const auto& layer : layers) {
        if (layer.layer_name == "GEMM" || layer.layer_name == "Score" || layer.layer_name == "Context" || layer.layer_name == "Flash_attn") {
            Instruction inst(Opcode::GEMM); // All GEMM-type layers use GEMM opcode
            inst.batch_size = layer.batch_size;
            inst.hidden_dim = layer.hidden_dim;
            inst.out_ch = layer.out_ch;

            if (layer.layer_name == "Flash_attn") inst.out_ch *= 2; // we read K and V at once. but outputs are not doubled.
            
            // Use base addresses (weight address is already set in layer from ndp_layer_builder)
            inst.addr_in = use_input_addr ? ndp_config.input_addr : ndp_config.output_addr;
            inst.addr_w = layer.addr_w; // Use weight address from layer (set by ndp_layer_builder)
            inst.addr_out = use_input_addr ? ndp_config.output_addr : ndp_config.input_addr;
            inst.vector_size = layer.vector_size;
            inst.address = layer.address;
            inst.data_size = layer.data_size;
            
            auto tiles = tile_gemm_instruction(inst, ndp.get_buf_size(), ndp_config.num_lanes_of_PU, ndp_config.lane_width, layer.layer_name == "Flash_attn");
            instructions.insert(instructions.end(), tiles.begin(), tiles.end());
            
            // Alternate address space for next operation
            use_input_addr = !use_input_addr;
        } else if (layer.layer_name == "Layernorm" || layer.layer_name == "GELU" || layer.layer_name == "Softmax") {
            add_tiled_compute_instructions(layer, use_input_addr);
        } else if (layer.layer_name == "FinishStep") {
            Opcode op = Opcode::FinishStep; // Use FinishStep opcode for FinishStep layer
            Instruction inst(op);
            inst.vector_size = layer.vector_size;
            instructions.push_back(inst);
        } else if (layer.layer_name == "Comm") {
            // NDP-to-NDP exchange inside one GPU package (sequence parallelism).
            // A layer tagged with comm_hops pays the intra-package link latency
            // (sp_latency_ns, 400 ns by default) once. The builder tags only the
            // post-attention exchange, so one decoder layer pays it once.
            const bool charge_sp_latency = (layer.comm_hops > 0);
            if (charge_sp_latency) {
                Instruction lat_inst(Opcode::GPUComm);
                lat_inst.comm_cycles =
                    (cycle_type)((long double)layer.comm_hops * ndp_config.sp_latency_ns);
                instructions.push_back(lat_inst);
            }
            Opcode op = Opcode::LoadBuffer; // Use LoadBuffer opcode for Comm layer
            Instruction inst(op);
            inst.vector_size = layer.vector_size;
            // Tagging the buffer traffic as communication lets it issue while the
            // link latency is still being paid, i.e. the DRAM access overlaps the
            // wire time instead of serializing behind it. Same convention the
            // GPU-level AllReduce path below uses.
            inst.is_gpu_comm = charge_sp_latency;
            if (!use_input_addr) {
                inst.address = ndp_config.input_addr;
            } else {
                inst.address = ndp_config.output_addr;
            }
            instructions.push_back(inst);
            Opcode op2 = Opcode::FlushOutBuffer; // Use FlushOutBuffer opcode for Comm layer
            Instruction inst2(op2);
            inst2.vector_size = layer.vector_size;
            inst2.is_gpu_comm = charge_sp_latency;
            if (use_input_addr) {
                inst2.address = ndp_config.input_addr;
            } else {
                inst2.address = ndp_config.output_addr;
            }
            instructions.push_back(inst2);
            
            //// Use base addresses for other operations
            //inst5.addr_in = use_input_addr ? ndp_config.input_addr : ndp_config.output_addr;
            //inst5.addr_w = 0; // No weight for non-GEMM operations
            //inst5.addr_out = use_input_addr ? ndp_config.output_addr : ndp_config.input_addr;
            
        } else if (layer.layer_name == "AllReduce") {
            // GPU-level all-reduce communication - treated as communication overhead
            Opcode op = Opcode::GPUComm;
            Instruction inst(op);
            // alpha-beta model: T = hops * NVLink hop latency + volume / bandwidth.
            // comm_hops is non-zero on only ONE step of a multi-step all-reduce,
            // so the 800 ns NVLink5 hop is charged once per collective rather
            // than once per step. On an NVL72-style fabric every peer GPU is a
            // single NVSwitch hop away, so one traversal is the whole cost.
            inst.comm_cycles = (cycle_type)((2 * layer.vector_size / 8) * 2 / ndp_config.interconnect_bw) // heuristics TODO: refine
                             + (cycle_type)((long double)layer.comm_hops * ndp_config.nvlink_latency_ns);
            instructions.push_back(inst);

            Opcode op2 = Opcode::LoadBuffer; 
            Instruction inst2(op2);
            inst2.vector_size = layer.vector_size / 8; // heuristics TODO: refine
            inst2.address = use_input_addr ? ndp_config.output_addr : ndp_config.input_addr;
            inst2.is_gpu_comm = true;
            instructions.push_back(inst2);

            Opcode op3 = Opcode::FlushOutBuffer;
            Instruction inst3(op3);
            inst3.vector_size = layer.vector_size / 8; // heuristics TODO: refine
            inst3.address = use_input_addr ? ndp_config.input_addr : ndp_config.output_addr;
            inst3.is_gpu_comm = true;
            instructions.push_back(inst3);


            //// Use base addresses for other operations
            //inst6.addr_in = use_input_addr ? ndp_config.input_addr : ndp_config.output_addr;
            //inst6.addr_w = 0; // No weight for non-GEMM operations
            //inst6.addr_out = use_input_addr ? ndp_config.output_addr : ndp_config.input_addr;
            
        } else if (layer.layer_name == "PipelineComm") {
            // GPU-level all-reduce communication - treated as communication overhead
            Opcode op = Opcode::GPUComm;
            Instruction inst(op);
            // Point-to-point send to the next pipeline stage: a single NVLink
            // hop regardless of the collective algorithm, plus the transfer time.
            inst.comm_cycles = (cycle_type)((layer.vector_size / 8 + layer.vector_size) * 2 / ndp_config.interconnect_bw) // heuristics TODO: refine
                             + (cycle_type)((long double)layer.comm_hops * ndp_config.nvlink_latency_ns);
            instructions.push_back(inst);

            Opcode op2 = Opcode::LoadBuffer;
            Instruction inst2(op2);
            inst2.vector_size = layer.vector_size / 8; // heuristics TODO: refine
            inst2.address = use_input_addr ? ndp_config.output_addr : ndp_config.input_addr;
            inst2.is_gpu_comm = true;
            instructions.push_back(inst2);

            Opcode op3 = Opcode::FlushOutBuffer;
            Instruction inst3(op3);
            inst3.vector_size = layer.vector_size; // heuristics TODO: refine
            inst3.address = use_input_addr ? ndp_config.input_addr : ndp_config.output_addr;
            inst3.is_gpu_comm = true;
            instructions.push_back(inst3);
        } else {
            // Other operations (Add, etc.)
            Opcode op = Opcode::Add; // Default opcode for other operations
            Instruction inst(op);
            inst.batch_size = layer.batch_size;
            inst.hidden_dim = layer.hidden_dim;
            inst.out_ch = layer.out_ch;
            
            // Use base addresses for other operations
            inst.addr_in = use_input_addr ? ndp_config.input_addr : ndp_config.output_addr;
            inst.addr_w = 0; // No weight for non-GEMM operations
            inst.addr_out = use_input_addr ? ndp_config.output_addr : ndp_config.input_addr;
            
            inst.vector_size = layer.vector_size;
            inst.address = layer.address;
            inst.data_size = layer.data_size;
            instructions.push_back(inst);
            
            // Alternate address space for next operation
            use_input_addr = !use_input_addr;
        }
    }
    current_inst = 0;

    std::cout << "NDP_scheduler: Loaded " << instructions.size() << " instructions" << std::endl;
}

void NDP_scheduler::cycle(cycle_type current_cycle) {
    if (current_inst < instructions.size()) {
        // Only move to next instruction if current one was issued successfully
        if (ndp.issue_instruction(instructions[current_inst])) {
            current_inst++;
        }
    }
    ndp.cycle();

    if (current_inst >= instructions.size() && !ndp.running()) {
        ndp_done = true;
        std::cout << "Average cycles per step: " << float(current_cycle) / (ndp.current_step) << std::endl;
    }
}

bool NDP_scheduler::has_pending_instructions() const {
    return current_inst < instructions.size();
}

bool NDP_scheduler::is_busy() const {
    return ndp.running() || has_pending_instructions();
}

void NDP_scheduler::add_tiled_compute_instructions(const Layer& layer, bool& use_input_addr) {
    // Tile the operation based on buffer size
    uint64_t src_addr = use_input_addr ? ndp_config.input_addr : ndp_config.output_addr;
    uint64_t dst_addr = use_input_addr ? ndp_config.output_addr : ndp_config.input_addr;
    uint32_t total_data_size = layer.vector_size * 2;  // Assuming 16-bit data
    uint32_t buffer_size = ndp.get_buf_size();
    
    // Calculate number of tiles needed
    uint32_t num_tiles = (total_data_size + buffer_size - 1) / buffer_size;
    uint32_t current_addr_offset = 0;
    
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        uint32_t current_tile_size = std::min(buffer_size, total_data_size - tile * buffer_size);
        uint32_t current_vector_elements = current_tile_size / 2;  // Convert back to elements
        
        // Add Load instruction for this tile
        Instruction load_inst(Opcode::LoadBuffer);
        load_inst.address = src_addr + current_addr_offset;
        load_inst.data_size = current_tile_size;
        instructions.push_back(load_inst);
        
        // Add compute instruction for this tile
        Opcode compute_op;
        if (layer.layer_name == "Layernorm") compute_op = Opcode::Layernorm;
        else if (layer.layer_name == "GELU") compute_op = Opcode::Gelu;
        else if (layer.layer_name == "Softmax") compute_op = Opcode::Softmax;
        else compute_op = Opcode::Add; // Default
        
        Instruction compute_inst(compute_op);
        compute_inst.vector_size = current_vector_elements;
        instructions.push_back(compute_inst);
        
        // Add Flush instruction for this tile
        Instruction flush_inst(Opcode::FlushOutBuffer);
        flush_inst.address = dst_addr + current_addr_offset;
        flush_inst.data_size = current_tile_size;
        instructions.push_back(flush_inst);
        
        current_addr_offset += current_tile_size;
    }
    
    // Alternate address space for next operation
    use_input_addr = !use_input_addr;
}

    
//     // Simple tiling logic - this should be enhanced based on your specific requirements
//     int max_batch_per_tile = std::min(static_cast<int>(inst.batch_size), 16);  // Max 16 batch per tile
//     int max_hidden_per_tile = std::min(static_cast<int>(inst.hidden_dim), lane_width);
//     int max_out_per_tile = std::min(static_cast<int>(inst.out_ch), buf_size / 4);  // Assuming 4 bytes per element
    
                
//                 // Update addresses based on tile position
//                 tile_inst.addr_in = inst.addr_in + (b_start * inst.hidden_dim + h_start) * 2;  // 2 bytes per element
//                 tile_inst.addr_w = inst.addr_w + (h_start * inst.out_ch + o_start) * 2;
//                 tile_inst.addr_out = inst.addr_out + (b_start * inst.out_ch + o_start) * 2;
                
    
std::vector<Instruction> NDP_scheduler::tile_gemm_instruction(const Instruction& inst, int buf_size, int num_lanes_of_PU, int lane_width, bool Flash_attn) {
	std::vector<Instruction> tiled;

    int tile_batch_size = std::min(static_cast<int>(num_lanes_of_PU), static_cast<int>(inst.batch_size)); 
    int tile_hidden_dim = std::min(static_cast<int>(lane_width), static_cast<int>(inst.hidden_dim)); 
    int tile_out_ch = std::min(static_cast<int>(buf_size / 4 / tile_batch_size), static_cast<int>(inst.out_ch));
    int tile_out_ch_ps; 
    if (Flash_attn) {
        // we set tile_out_ch_ps given ndp buffer size
        tile_out_ch_ps = std::min(static_cast<int>(buf_size / 4 / (tile_batch_size * (lane_width / 128))), static_cast<int>(inst.out_ch / 2)); // buffer maximum or output channel size divided by 2 
        // we use tile_out_ch for weights, but tile_out_ch_ps for output.
        tile_out_ch = tile_out_ch_ps * 2; // because we read K and V at once
    }
	for (int oc = 0; oc < inst.out_ch; ) {
		for (int bs = 0; bs < inst.batch_size; ) {
			for (int hd = 0; hd < inst.hidden_dim; ) {
				Instruction tile = inst;
				tile.out_ch = std::min(tile_out_ch, static_cast<int>(inst.out_ch - oc));
				tile.hidden_dim = std::min(tile_hidden_dim, static_cast<int>(inst.hidden_dim - hd));
				tile.batch_size = std::min(tile_batch_size, static_cast<int>(inst.batch_size - bs));
				// Calculate addresses based on tile offsets
				tile.addr_in = inst.addr_in + (bs * inst.hidden_dim + hd * tile_batch_size) * 2;
				tile.addr_w = inst.addr_w + (oc * inst.hidden_dim + hd * tile_out_ch) * 2;
				tile.addr_out = inst.addr_out + (bs * tile_out_ch + oc * inst.batch_size) * 4;


				tiled.push_back(tile);

				hd += tile_hidden_dim;
			}
            // if (Flash_attn) {
            //     Instruction softmax_inst(Opcode::Softmax); // online softmax after attention
            //     softmax_inst.vector_size = tile_out_ch * tile_batch_size * int(inst.hidden_dim / 128); // float32 output
            //     tiled.push_back(softmax_inst);
            // }
			// Insert flush instruction after each batch size tile (2nd loop)
			Instruction flush_inst(Opcode::FlushOutBuffer);
            if (Flash_attn && oc + tile_out_ch >= inst.out_ch) {
                flush_inst.address = inst.addr_out + (bs * inst.hidden_dim) * 4;
                flush_inst.data_size = std::min(tile_batch_size, static_cast<int>(inst.batch_size - bs)) * tile_hidden_dim * 4; // assuming float32 output

            }
            else {
                flush_inst.address = inst.addr_out + (bs * tile_out_ch + oc * inst.batch_size) * 4;
                flush_inst.data_size = tile_out_ch * tile_batch_size * 4; // assuming float32 output
            }
			tiled.push_back(flush_inst);
			bs += tile_batch_size;
		}
		oc += tile_out_ch;
	}
	return tiled;
}