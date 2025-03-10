add_executable(linalg-reduce-2d-benchmark launcher.cpp)
target_link_libraries(linalg-reduce-2d-benchmark PRIVATE GoogleBenchmark)

add_linalg_benchmark(reduce-2d)

target_link_libraries(linalg-reduce-2d-benchmark PRIVATE linalg_reduce-2d_kernels)
