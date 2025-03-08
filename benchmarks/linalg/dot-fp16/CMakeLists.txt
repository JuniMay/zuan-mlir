add_executable(linalg-dot-fp16-benchmark launcher.cpp)
target_link_libraries(linalg-dot-fp16-benchmark PRIVATE GoogleBenchmark)

add_linalg_benchmark(dot-fp16)

target_link_libraries(linalg-dot-fp16-benchmark PRIVATE linalg_dot-fp16_kernels)
