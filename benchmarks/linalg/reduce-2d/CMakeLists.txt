add_executable(linalg-reduce-2d-benchmark launcher.cpp)
target_link_libraries(linalg-reduce-2d-benchmark PRIVATE GoogleBenchmark)

# add_linalg_benchmark(reduce-2d)
add_linalg_zuan_kernel(reduce-2d 16 2)
add_linalg_autovec_kernel(reduce-2d 16)

target_link_libraries(
  linalg-reduce-2d-benchmark
  PRIVATE
  linalg_reduce-2d_zuan_16_2
  linalg_reduce-2d_autovec_16
)
