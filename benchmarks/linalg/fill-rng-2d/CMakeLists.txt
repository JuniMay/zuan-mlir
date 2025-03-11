add_executable(linalg-fill-rng-2d-benchmark launcher.cpp)
target_link_libraries(linalg-fill-rng-2d-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(fill-rng-2d 8 1)
add_linalg_zuan_kernel(fill-rng-2d 8 2)
add_linalg_autovec_kernel(fill-rng-2d 8)
add_linalg_autovec_kernel(fill-rng-2d 16)
add_linalg_autovec_kernel(fill-rng-2d 32)
add_linalg_autovec_kernel(fill-rng-2d 64)

target_link_libraries(
  linalg-fill-rng-2d-benchmark
  PRIVATE
  linalg_fill-rng-2d_zuan_8_1
  linalg_fill-rng-2d_zuan_8_2
  linalg_fill-rng-2d_autovec_8
  linalg_fill-rng-2d_autovec_16
  linalg_fill-rng-2d_autovec_32
  linalg_fill-rng-2d_autovec_64
)
