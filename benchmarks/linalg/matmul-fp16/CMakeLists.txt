add_executable(linalg-matmul-fp16-benchmark launcher.cpp)
target_link_libraries(linalg-matmul-fp16-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(matmul-fp16 8 4)
add_linalg_zuan_kernel(matmul-fp16 8 8)
add_linalg_zuan_kernel(matmul-fp16 16 2)
add_linalg_zuan_kernel(matmul-fp16 16 4)
add_linalg_zuan_kernel(matmul-fp16 32 2)

add_linalg_transform_kernel(matmul-fp16 8 4)
add_linalg_transform_kernel(matmul-fp16 8 8)
add_linalg_transform_kernel(matmul-fp16 16 2)
add_linalg_transform_kernel(matmul-fp16 16 4)
add_linalg_transform_kernel(matmul-fp16 32 2)

add_linalg_autovec_kernel(matmul-fp16 8)
add_linalg_autovec_kernel(matmul-fp16 16)
add_linalg_autovec_kernel(matmul-fp16 32)
add_linalg_autovec_kernel(matmul-fp16 64)

target_link_libraries(
  linalg-matmul-fp16-benchmark
  PRIVATE
  linalg_matmul-fp16_zuan_8_4
  linalg_matmul-fp16_zuan_8_8
  linalg_matmul-fp16_zuan_16_2
  linalg_matmul-fp16_zuan_16_4
  linalg_matmul-fp16_zuan_32_2

  linalg_matmul-fp16_transform_8_4
  linalg_matmul-fp16_transform_8_8
  linalg_matmul-fp16_transform_16_2
  linalg_matmul-fp16_transform_16_4
  linalg_matmul-fp16_transform_32_2

  linalg_matmul-fp16_autovec_8
  linalg_matmul-fp16_autovec_16
  linalg_matmul-fp16_autovec_32
  linalg_matmul-fp16_autovec_64
)
