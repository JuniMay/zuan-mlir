add_executable(linalg-mmt4d-benchmark launcher.cpp)
target_link_libraries(linalg-mmt4d-benchmark PRIVATE GoogleBenchmark)

add_linalg_benchmark(mmt4d)

target_link_libraries(linalg-mmt4d-benchmark PRIVATE linalg_mmt4d_kernels)
