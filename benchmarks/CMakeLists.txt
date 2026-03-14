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
  BUILD_BYPRODUCTS
    <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}
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
set_property(
  TARGET GoogleBenchmark PROPERTY IMPORTED_LOCATION
  "${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}")

add_dependencies(GoogleBenchmark project_googlebenchmark)

find_package(Threads)
target_link_libraries(GoogleBenchmark INTERFACE Threads::Threads)

#-------------------------------------------------------------------------------
# Configure the benchmarks
#-------------------------------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if (DYNO_BENCHMARK_DUMP_IR)
  set(DYNO_BENCHMARK_DUMP_IR_FLAG -mlir-print-ir-after-all)
else()
  unset(DYNO_BENCHMARK_DUMP_IR_FLAG)
endif()

include(DynoHostTools)
dyno_configure_host_tools()

include(DynoBenchmarkKernels)
include(DynoTriton)

add_subdirectory(dyno)
add_subdirectory(linalg)

if (ENABLE_TRITON_BENCHMARKS)
  find_package(OpenMP REQUIRED)
  dyno_validate_triton_benchmarks()
  add_subdirectory(triton)
endif()
