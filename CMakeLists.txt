cmake_minimum_required(VERSION 3.20.0)
project(zuan LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

find_package(MLIR REQUIRED CONFIG)

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

set(ZUAN_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(ZUAN_BINARY_DIR ${PROJECT_BINARY_DIR})

message(STATUS "Zuan source dir: ${ZUAN_SOURCE_DIR}")
message(STATUS "Zuan binary dir: ${ZUAN_BINARY_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${ZUAN_SOURCE_DIR}/include)
include_directories(${ZUAN_BINARY_DIR}/include)

include_directories(${ZUAN_SOURCE_DIR}/include/Dialect)
include_directories(${ZUAN_BINARY_DIR}/include/Dialect)

include_directories(${ZUAN_SOURCE_DIR}/runtime)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

#-------------------------------------------------------------------------------
# RISC-V Toolchain
#-------------------------------------------------------------------------------

include(ExternalProject)

if (ENABLE_RISCV_GNU_TOOLCHAIN)
  set(RISCV_GNU_TOOLCHAIN_DIR "${ZUAN_SOURCE_DIR}/third_party/riscv-gnu-toolchain")
  set(RISCV_GNU_TOOLCHAIN_BUILD_DIR "${ZUAN_BINARY_DIR}/third_party/riscv-gnu-toolchain")
  ExternalProject_Add(
    riscv-gnu-toolchain
    SOURCE_DIR        ${RISCV_GNU_TOOLCHAIN_DIR}
    PREFIX            ${RISCV_GNU_TOOLCHAIN_BUILD_DIR}
    CONFIGURE_COMMAND ${RISCV_GNU_TOOLCHAIN_DIR}/configure --prefix=${RISCV_GNU_TOOLCHAIN_BUILD_DIR}
    BUILD_COMMAND     make clean && make linux build-qemu -j8
    BUILD_IN_SOURCE   TRUE
    INSTALL_COMMAND   ""
  )
endif()
