add_executable(linalg-reduce-benchmark launcher.cpp)
target_link_libraries(linalg-reduce-benchmark PRIVATE GoogleBenchmark)

add_linalg_benchmark(reduce)

target_link_libraries(linalg-reduce-benchmark PRIVATE linalg_reduce_kernels)
