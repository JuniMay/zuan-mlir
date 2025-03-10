add_executable(linalg-rsqrt-benchmark launcher.cpp)
target_link_libraries(linalg-rsqrt-benchmark PRIVATE GoogleBenchmark)

add_linalg_benchmark(rsqrt)

target_link_libraries(linalg-rsqrt-benchmark PRIVATE linalg_rsqrt_kernels)
