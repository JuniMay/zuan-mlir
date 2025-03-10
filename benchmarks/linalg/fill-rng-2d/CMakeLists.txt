add_executable(linalg-fill-rng-2d-benchmark launcher.cpp)
target_link_libraries(linalg-fill-rng-2d-benchmark PRIVATE GoogleBenchmark)

add_linalg_benchmark(fill-rng-2d)

target_link_libraries(linalg-fill-rng-2d-benchmark PRIVATE linalg_fill-rng-2d_kernels)
