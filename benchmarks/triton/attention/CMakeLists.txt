add_executable(triton-attention-benchmark launcher.cpp)

add_triton_benchmark(attention "--BLOCK_N;128")
target_link_libraries(
  triton-attention-benchmark

  PRIVATE
  OpenMP::OpenMP_CXX
  GoogleBenchmark
  triton_attention_kernels
  mlir_c_runner_utils
)
