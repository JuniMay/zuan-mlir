add_executable(triton-matmul-benchmark launcher.cpp)

add_triton_benchmark(matmul "--BLOCK_SIZE_M;64;--BLOCK_SIZE_N;64;--BLOCK_SIZE_K;128;--GROUP_SIZE_M;8")
target_link_libraries(
  triton-matmul-benchmark
  
  PRIVATE
  OpenMP::OpenMP_CXX
  GoogleBenchmark
  triton_vector_add_kernels
  mlir_c_runner_utils
)
