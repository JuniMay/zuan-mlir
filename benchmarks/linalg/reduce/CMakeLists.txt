add_executable(linalg-reduce-benchmark launcher.cpp)
target_link_libraries(linalg-reduce-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(reduce 16 2)
add_linalg_autovec_kernel(reduce 8)
add_linalg_autovec_kernel(reduce 16)
add_linalg_autovec_kernel(reduce 32)
add_linalg_autovec_kernel(reduce 64)

target_link_libraries(
  linalg-reduce-benchmark
  PRIVATE
  linalg_reduce_zuan_16_2
  linalg_reduce_autovec_8
  linalg_reduce_autovec_16
  linalg_reduce_autovec_32
  linalg_reduce_autovec_64
)
