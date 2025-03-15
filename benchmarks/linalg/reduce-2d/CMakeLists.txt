add_executable(linalg-reduce-2d-benchmark launcher.cpp)
target_link_libraries(linalg-reduce-2d-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(reduce-2d 8 2)
add_linalg_zuan_kernel(reduce-2d 8 4)
add_linalg_zuan_kernel(reduce-2d 8 8)
add_linalg_zuan_kernel(reduce-2d 16 1)
add_linalg_zuan_kernel(reduce-2d 16 2)
add_linalg_zuan_kernel(reduce-2d 16 4)

add_linalg_transform_kernel(reduce-2d 8 2)
add_linalg_transform_kernel(reduce-2d 8 4)
add_linalg_transform_kernel(reduce-2d 8 8)
add_linalg_transform_kernel(reduce-2d 16 1)
add_linalg_transform_kernel(reduce-2d 16 2)
add_linalg_transform_kernel(reduce-2d 16 4)

add_linalg_autovec_kernel(reduce-2d 8)
add_linalg_autovec_kernel(reduce-2d 16)
add_linalg_autovec_kernel(reduce-2d 32)
add_linalg_autovec_kernel(reduce-2d 64)

target_link_libraries(
  linalg-reduce-2d-benchmark
  PRIVATE
  linalg_reduce-2d_zuan_8_2
  linalg_reduce-2d_zuan_8_4
  linalg_reduce-2d_zuan_8_8
  linalg_reduce-2d_zuan_16_1
  linalg_reduce-2d_zuan_16_2
  linalg_reduce-2d_zuan_16_4

  linalg_reduce-2d_transform_8_2
  linalg_reduce-2d_transform_8_4
  linalg_reduce-2d_transform_8_8
  linalg_reduce-2d_transform_16_1
  linalg_reduce-2d_transform_16_2
  linalg_reduce-2d_transform_16_4

  linalg_reduce-2d_autovec_8
  linalg_reduce-2d_autovec_16
  linalg_reduce-2d_autovec_32
  linalg_reduce-2d_autovec_64
)
