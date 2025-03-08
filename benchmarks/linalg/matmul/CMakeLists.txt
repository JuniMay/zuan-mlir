add_executable(linalg-matmul-benchmark launcher.cpp)
target_link_libraries(linalg-matmul-benchmark PRIVATE GoogleBenchmark)

add_linalg_benchmark(matmul)

target_link_libraries(linalg-matmul-benchmark PRIVATE linalg_matmul_kernels)
