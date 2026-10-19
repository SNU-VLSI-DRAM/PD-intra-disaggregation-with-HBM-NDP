cmake_minimum_required(VERSION 3.15)

# Simulator CMake configuration
# Provides a library (ndp_sim_core) and an executable (ndp_sim)
# Depends on the top-level dramsim3 shared library already defined in parent.

# Sources grouped the way the tree is laid out:
#   config/  configuration: CLI, model table, JSON reader, address map
#   gpu/     the GPU model, split into construction / layers / tiling / execution
#   ndp/     the NDP controller, its scheduler and its layer builder
#   sim/     the top-level simulator loop and the DRAM wrapper
set(SIMULATOR_CONFIG_SOURCES
    config/CliOptions.cc
    config/ConfigBuilder.cc
    config/ModelZoo.cc
    config/NdpConfigFile.cc
)

set(SIMULATOR_GPU_SOURCES
    gpu/gpu_model.cc
    gpu/gpu_layers.cc
    gpu/gpu_tiling.cc
    gpu/gpu_exec.cc
)

set(SIMULATOR_NDP_SOURCES
    ndp/PU.cc
    ndp/ndp_ctrl.cc
    ndp/NDP_scheduler.cc
    ndp/ndp_layer_builder.cc
)

set(SIMULATOR_SIM_SOURCES
    sim/Simulator.cc
    sim/Dram.cc
)

set(SIMULATOR_SOURCES
    ${SIMULATOR_CONFIG_SOURCES}
    ${SIMULATOR_GPU_SOURCES}
    ${SIMULATOR_NDP_SOURCES}
    ${SIMULATOR_SIM_SOURCES}
)

add_library(ndp_sim_core STATIC ${SIMULATOR_SOURCES})

# Include paths
# - simulator (this dir)
# - parent include directories
# - ext headers for args/json/fmt
# Parent scope already has targets: inih, format, json, args

set_target_properties(ndp_sim_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ndp_sim_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/ext/headers
        ${CMAKE_SOURCE_DIR}/ext/fmt/include
)

# Link against dramsim3 and header-only helpers
# (dramsim3 already links inih and format internally, but we add INTERFACE libs for clarity)

target_link_libraries(ndp_sim_core
    PUBLIC
        dramsim3
        args
        json
        format
)

# Executable
add_executable(ndp_sim main.cc)

set_target_properties(ndp_sim PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(ndp_sim
    PRIVATE
        ndp_sim_core
)

target_include_directories(ndp_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optional: install rules (can be extended later)
install(TARGETS ndp_sim RUNTIME DESTINATION bin)
install(TARGETS ndp_sim_core ARCHIVE DESTINATION lib)
