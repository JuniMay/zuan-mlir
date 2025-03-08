add_executable(linalg-relu-benchmark launcher.cpp)
target_link_libraries(linalg-relu-benchmark PRIVATE GoogleBenchmark)

add_linalg_benchmark(relu)

target_link_libraries(linalg-relu-benchmark PRIVATE linalg_relu_kernels)
