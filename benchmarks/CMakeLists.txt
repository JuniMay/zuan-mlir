#-------------------------------------------------------------------------------
# Benchmark dependencies
#-------------------------------------------------------------------------------

include(DynoGoogleBenchmark)
dyno_configure_googlebenchmark()

#-------------------------------------------------------------------------------
# Shared kernel build configuration
#-------------------------------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include(DynoKernelBuild)
dyno_configure_kernel_build()

include(DynoBenchmarkKernels)
include(DynoTriton)

#-------------------------------------------------------------------------------
# Benchmark suites
#-------------------------------------------------------------------------------

add_subdirectory(dyno)
add_subdirectory(linalg)

if (ENABLE_TRITON_BENCHMARKS)
  find_package(OpenMP REQUIRED)
  dyno_validate_triton_benchmarks()
  add_subdirectory(triton)
endif()
