add_executable(linalg-relu-benchmark launcher.cpp)
target_link_libraries(linalg-relu-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(relu 16 1)
add_linalg_zuan_kernel(relu 16 2)
add_linalg_zuan_kernel(relu 16 4)
# add_linalg_zuan_kernel(relu 8 1)
# add_linalg_zuan_kernel(relu 8 2)
# add_linalg_zuan_kernel(relu 8 4)
# add_linalg_zuan_kernel(relu 4 1)
# add_linalg_zuan_kernel(relu 4 2)
# add_linalg_zuan_kernel(relu 4 4)
# add_linalg_zuan_kernel(relu 4 8)

add_linalg_autovec_kernel(relu 8)
add_linalg_autovec_kernel(relu 16)
add_linalg_autovec_kernel(relu 32)
add_linalg_autovec_kernel(relu 64)

target_link_libraries(
  linalg-relu-benchmark
  PRIVATE
  # linalg_relu_zuan_4_8
  # linalg_relu_zuan_4_4
  # linalg_relu_zuan_4_2
  # linalg_relu_zuan_4_1
  # linalg_relu_zuan_8_1
  # linalg_relu_zuan_8_2
  # linalg_relu_zuan_8_4
  linalg_relu_zuan_16_1
  linalg_relu_zuan_16_2
  linalg_relu_zuan_16_4
  linalg_relu_autovec_8
  linalg_relu_autovec_16
  linalg_relu_autovec_32
  linalg_relu_autovec_64
)
