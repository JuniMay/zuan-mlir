add_executable(linalg-fill-rng-2d-benchmark launcher.cpp)
target_link_libraries(linalg-fill-rng-2d-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(fill-rng-2d 8 2)
add_linalg_autovec_kernel(fill-rng-2d 16)

target_link_libraries(
  linalg-fill-rng-2d-benchmark
  PRIVATE
  linalg_fill-rng-2d_zuan_8_2
  linalg_fill-rng-2d_autovec_16
)
