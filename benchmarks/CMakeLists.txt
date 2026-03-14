separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

#-------------------------------------------------------------------------------
# GoogleBenchmark Library
#-------------------------------------------------------------------------------

message(STATUS "Configuring GoogleBenchmark Library")

include(ExternalProject)

ExternalProject_Add(
  project_googlebenchmark

  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG "v1.9.1"
  GIT_SHALLOW 1
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/vendor/benchmark
  TIMEOUT 10
  BUILD_BYPRODUCTS <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/vendor/benchmark
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DBENCHMARK_ENABLE_TESTING=OFF
    -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
    -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -Wno-c2y-extensions"
  UPDATE_COMMAND ""
  TEST_COMMAND ""
)

ExternalProject_Get_Property(project_googlebenchmark INSTALL_DIR)

file(MAKE_DIRECTORY ${INSTALL_DIR}/include)
add_library(GoogleBenchmark STATIC IMPORTED)
target_include_directories(GoogleBenchmark INTERFACE ${INSTALL_DIR}/include)
set_property(TARGET GoogleBenchmark PROPERTY IMPORTED_LOCATION
  "${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}")

add_dependencies(GoogleBenchmark project_googlebenchmark)

find_package(Threads)
target_link_libraries(GoogleBenchmark INTERFACE Threads::Threads)

#-------------------------------------------------------------------------------
# OpenMP
#-------------------------------------------------------------------------------

find_package(OpenMP REQUIRED)

#-------------------------------------------------------------------------------
# Configure the benchmarks
#-------------------------------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(ZUAN_OPT ${ZUAN_BINARY_DIR}/bin/zuan-opt)
set(ZUAN_TRANSLATE ${ZUAN_BINARY_DIR}/bin/zuan-translate)
set(ZUAN_OPT_CMD ${ZUAN_OPT})
set(ZUAN_TRANSLATE_CMD ${ZUAN_TRANSLATE})
if (ZUAN_BENCHMARK_DUMP_IR)
  set(ZUAN_BENCHMARK_DUMP_IR_FLAG -mlir-print-ir-after-all)
else()
  unset(ZUAN_BENCHMARK_DUMP_IR_FLAG)
endif()
set(ZUAN_QEMU_LD_PREFIX "$ENV{QEMU_LD_PREFIX}" CACHE PATH "QEMU_LD_PREFIX used for cross-compiled benchmark helper tools")

if(CMAKE_CROSSCOMPILING AND CMAKE_CROSSCOMPILING_EMULATOR)
  if (NOT ZUAN_QEMU_LD_PREFIX STREQUAL "")
    set(ZUAN_OPT_CMD ${CMAKE_COMMAND} -E env QEMU_LD_PREFIX=${ZUAN_QEMU_LD_PREFIX} ${CMAKE_CROSSCOMPILING_EMULATOR} ${ZUAN_OPT})
    set(ZUAN_TRANSLATE_CMD ${CMAKE_COMMAND} -E env QEMU_LD_PREFIX=${ZUAN_QEMU_LD_PREFIX} ${CMAKE_CROSSCOMPILING_EMULATOR} ${ZUAN_TRANSLATE})
  else()
    set(ZUAN_OPT_CMD ${CMAKE_CROSSCOMPILING_EMULATOR} ${ZUAN_OPT})
    set(ZUAN_TRANSLATE_CMD ${CMAKE_CROSSCOMPILING_EMULATOR} ${ZUAN_TRANSLATE})
  endif()
endif()

# Benchmark generation already relies on host-side MLIR tools (`mlir-opt`,
# `mlir-translate`, `clang`). Prefer the host-built Zuan tools as well when
# they are available so custom Zuan/transform pipelines run natively instead of
# through the cross-built binaries under QEMU.
set(ZUAN_HOST_BINARY_DIR ${ZUAN_SOURCE_DIR}/build/bin)
if (EXISTS ${ZUAN_HOST_BINARY_DIR}/zuan-opt)
  set(ZUAN_OPT_CMD ${ZUAN_HOST_BINARY_DIR}/zuan-opt)
endif()
if (EXISTS ${ZUAN_HOST_BINARY_DIR}/zuan-translate)
  set(ZUAN_TRANSLATE_CMD ${ZUAN_HOST_BINARY_DIR}/zuan-translate)
endif()

include(ZuanTriton)

add_subdirectory(zuan)
add_subdirectory(linalg)

if (ENABLE_TRITON_BENCHMARKS)
  zuan_validate_triton_benchmarks()
  add_subdirectory(triton)
endif()
