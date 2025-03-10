add_executable(linalg-matmul-fp16-benchmark launcher.cpp)
target_link_libraries(linalg-matmul-fp16-benchmark PRIVATE GoogleBenchmark)

add_linalg_benchmark(matmul-fp16)

target_link_libraries(linalg-matmul-fp16-benchmark PRIVATE linalg_matmul-fp16_kernels)
