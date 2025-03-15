add_executable(linalg-exp-benchmark launcher.cpp)
target_link_libraries(linalg-exp-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(exp 16 1)
add_linalg_zuan_kernel(exp 16 2)
add_linalg_zuan_kernel(exp 16 4)
add_linalg_zuan_kernel(exp 8 1)
add_linalg_zuan_kernel(exp 8 2)
add_linalg_zuan_kernel(exp 8 4)

add_linalg_transform_kernel(exp 16 1)
add_linalg_transform_kernel(exp 16 2)
add_linalg_transform_kernel(exp 16 4)
add_linalg_transform_kernel(exp 8 1)
add_linalg_transform_kernel(exp 8 2)
add_linalg_transform_kernel(exp 8 4)

add_linalg_autovec_kernel(exp 8)
add_linalg_autovec_kernel(exp 16)
add_linalg_autovec_kernel(exp 32)
add_linalg_autovec_kernel(exp 64)

target_link_libraries(
  linalg-exp-benchmark
  PRIVATE
  linalg_exp_zuan_8_1
  linalg_exp_zuan_8_2
  linalg_exp_zuan_8_4
  linalg_exp_zuan_16_1
  linalg_exp_zuan_16_2
  linalg_exp_zuan_16_4

  linalg_exp_transform_8_1
  linalg_exp_transform_8_2
  linalg_exp_transform_8_4
  linalg_exp_transform_16_1
  linalg_exp_transform_16_2
  linalg_exp_transform_16_4


  linalg_exp_autovec_8
  linalg_exp_autovec_16
  linalg_exp_autovec_32
  linalg_exp_autovec_64
)
