add_executable(linalg-matmul-transpose-b-benchmark launcher.cpp)
target_link_libraries(linalg-matmul-transpose-b-benchmark PRIVATE GoogleBenchmark)

add_linalg_benchmark(matmul-transpose-b)

target_link_libraries(linalg-matmul-transpose-b-benchmark PRIVATE linalg_matmul-transpose-b_kernels)
