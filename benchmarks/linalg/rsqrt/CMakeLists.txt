add_executable(linalg-rsqrt-benchmark launcher.cpp)
target_link_libraries(linalg-rsqrt-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(rsqrt 16 2)
add_linalg_zuan_kernel(rsqrt 16 4)

add_linalg_autovec_kernel(rsqrt 8)
add_linalg_autovec_kernel(rsqrt 16)
add_linalg_autovec_kernel(rsqrt 32)
add_linalg_autovec_kernel(rsqrt 64)

target_link_libraries(
  linalg-rsqrt-benchmark
  PRIVATE
  linalg_rsqrt_zuan_16_2
  linalg_rsqrt_zuan_16_4
  
  linalg_rsqrt_autovec_8
  linalg_rsqrt_autovec_16
  linalg_rsqrt_autovec_32
  linalg_rsqrt_autovec_64
)
