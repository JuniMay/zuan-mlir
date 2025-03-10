add_executable(linalg-quantized-matmul-benchmark launcher.cpp)
target_link_libraries(linalg-quantized-matmul-benchmark PRIVATE GoogleBenchmark)

add_linalg_benchmark(quantized-matmul)

target_link_libraries(linalg-quantized-matmul-benchmark PRIVATE linalg_quantized-matmul_kernels)
