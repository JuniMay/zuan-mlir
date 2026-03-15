cmake_minimum_required(VERSION 3.20.0)
project(dyno LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

include(CTest)

option(ENABLE_RISCV_GNU_TOOLCHAIN "Build the RISC-V GNU toolchain" OFF)
option(ENABLE_BENCHMARKS "Build benchmark targets" OFF)
option(ENABLE_TRITON_BENCHMARKS "Build Triton-based benchmark targets" OFF)
option(ENABLE_RUNTIME_REGRESSIONS "Enable regression testing" OFF)
option(DYNO_BENCHMARK_DUMP_IR "Dump MLIR after each pass" OFF)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

find_package(MLIR REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

#-------------------------------------------------------------------------------
# Set up the source tree
#-------------------------------------------------------------------------------

set(DYNO_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(DYNO_BINARY_DIR ${PROJECT_BINARY_DIR})

message(STATUS "Dyno source dir: ${DYNO_SOURCE_DIR}")
message(STATUS "Dyno binary dir: ${DYNO_BINARY_DIR}")

list(APPEND CMAKE_MODULE_PATH "${DYNO_SOURCE_DIR}/cmake")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${DYNO_SOURCE_DIR}/include)
include_directories(${DYNO_BINARY_DIR}/include)

include_directories(${DYNO_SOURCE_DIR}/include/Dialect)
include_directories(${DYNO_BINARY_DIR}/include/Dialect)

include_directories(${DYNO_SOURCE_DIR}/runtime)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
if (BUILD_TESTING)
  add_subdirectory(test)
  if (ENABLE_RUNTIME_REGRESSIONS)
    add_subdirectory(regression)
  endif()
endif()

#-------------------------------------------------------------------------------
# RISC-V Toolchain
#-------------------------------------------------------------------------------

include(ExternalProject)

set(RISCV_GNU_TOOLCHAIN_DIR "${DYNO_SOURCE_DIR}/third_party/riscv-gnu-toolchain")
set(RISCV_GNU_TOOLCHAIN_BUILD_DIR "${DYNO_BINARY_DIR}/third_party/riscv-gnu-toolchain")

if (ENABLE_RISCV_GNU_TOOLCHAIN)
  ExternalProject_Add(
    riscv-gnu-toolchain
    SOURCE_DIR        ${RISCV_GNU_TOOLCHAIN_DIR}
    PREFIX            ${RISCV_GNU_TOOLCHAIN_BUILD_DIR}
    # LLVM is also enabled here to support OpenMP.
    CONFIGURE_COMMAND ${RISCV_GNU_TOOLCHAIN_DIR}/configure --enable-llvm --prefix=${RISCV_GNU_TOOLCHAIN_BUILD_DIR}
    BUILD_COMMAND     make clean && make -j8 linux build-sim SIM=qemu
    BUILD_IN_SOURCE   TRUE
    INSTALL_COMMAND   ""
  )
endif()

#-------------------------------------------------------------------------------
# Benchmarks
#-------------------------------------------------------------------------------

if (ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
