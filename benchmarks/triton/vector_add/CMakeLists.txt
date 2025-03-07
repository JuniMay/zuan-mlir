add_executable(triton-vector-add-benchmark launcher.cpp)

add_triton_benchmark(vector_add "--BLOCK_SIZE;512")
target_link_libraries(
  triton-vector-add-benchmark
  
  PRIVATE
  OpenMP::OpenMP_CXX
  GoogleBenchmark
  triton_vector_add_kernels
  mlir_c_runner_utils
)
