add_executable(linalg-rsqrt-benchmark launcher.cpp)
target_link_libraries(linalg-rsqrt-benchmark PRIVATE GoogleBenchmark)

# add_linalg_benchmark(rsqrt)
add_linalg_zuan_kernel(rsqrt 16 2)
add_linalg_autovec_kernel(rsqrt 16)

target_link_libraries(
  linalg-rsqrt-benchmark
  PRIVATE
  linalg_rsqrt_zuan_16_2
  linalg_rsqrt_autovec_16
)
