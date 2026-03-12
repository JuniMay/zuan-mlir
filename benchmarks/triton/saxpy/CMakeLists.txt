add_executable(triton-saxpy-benchmark launcher.cpp)

add_triton_benchmark(saxpy "--BLOCK_SIZE;512")
target_link_libraries(
  triton-saxpy-benchmark

  PRIVATE
  OpenMP::OpenMP_CXX
  GoogleBenchmark
  triton_saxpy_kernels
  mlir_c_runner_utils
)
