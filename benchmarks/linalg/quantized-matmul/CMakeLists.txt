add_executable(linalg-quantized-matmul-benchmark launcher.cpp)
target_link_libraries(linalg-quantized-matmul-benchmark PRIVATE GoogleBenchmark)

# add_linalg_benchmark(quantized-matmul)
add_linalg_zuan_kernel(quantized-matmul 8 2)
add_linalg_autovec_kernel(quantized-matmul 16)

target_link_libraries(
  linalg-quantized-matmul-benchmark 
  PRIVATE
  linalg_quantized-matmul_zuan_8_2
  linalg_quantized-matmul_autovec_16
)
