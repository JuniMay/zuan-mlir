add_executable(linalg-matmul-transpose-a-benchmark launcher.cpp)
target_link_libraries(linalg-matmul-transpose-a-benchmark PRIVATE GoogleBenchmark)

add_linalg_benchmark(matmul-transpose-a)

target_link_libraries(linalg-matmul-transpose-a-benchmark PRIVATE linalg_matmul-transpose-a_kernels)
