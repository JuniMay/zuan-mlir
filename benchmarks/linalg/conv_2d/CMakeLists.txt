add_executable(linalg-conv_2d-benchmark launcher.cpp)
target_link_libraries(linalg-conv_2d-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(conv_2d 8 1)
add_linalg_zuan_kernel(conv_2d 8 2)
add_linalg_zuan_kernel(conv_2d 8 4)
add_linalg_zuan_kernel(conv_2d 4 2)
add_linalg_zuan_kernel(conv_2d 4 4)

add_linalg_autovec_kernel(conv_2d 8)
add_linalg_autovec_kernel(conv_2d 16)
add_linalg_autovec_kernel(conv_2d 32)
add_linalg_autovec_kernel(conv_2d 64)

target_link_libraries(
  linalg-conv_2d-benchmark
  PRIVATE
  linalg_conv_2d_zuan_8_1
  linalg_conv_2d_zuan_8_2
  linalg_conv_2d_zuan_8_4
  linalg_conv_2d_zuan_4_2
  linalg_conv_2d_zuan_4_4

  linalg_conv_2d_autovec_8
  linalg_conv_2d_autovec_16
  linalg_conv_2d_autovec_32
  linalg_conv_2d_autovec_64
)

