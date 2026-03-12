add_executable(triton-vector-mul-benchmark launcher.cpp)

add_triton_benchmark(vector_mul "--BLOCK_SIZE;512")
target_link_libraries(
  triton-vector-mul-benchmark

  PRIVATE
  OpenMP::OpenMP_CXX
  GoogleBenchmark
  triton_vector_mul_kernels
  mlir_c_runner_utils
)
