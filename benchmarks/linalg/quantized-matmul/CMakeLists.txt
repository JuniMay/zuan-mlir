add_executable(linalg-quantized-matmul-benchmark launcher.cpp)
target_link_libraries(linalg-quantized-matmul-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(quantized-matmul 8 4)
add_linalg_zuan_kernel(quantized-matmul 16 2)

# XXX: quantized-matmul failed to be vectorized.
# add_linalg_transform_kernel(quantized-matmul 8 4)
# add_linalg_transform_kernel(quantized-matmul 16 2)

add_linalg_autovec_kernel(quantized-matmul 8)
add_linalg_autovec_kernel(quantized-matmul 16)
add_linalg_autovec_kernel(quantized-matmul 32)
add_linalg_autovec_kernel(quantized-matmul 64)


target_link_libraries(
  linalg-quantized-matmul-benchmark
  PRIVATE
  linalg_quantized-matmul_zuan_8_4
  linalg_quantized-matmul_zuan_16_2

  # linalg_quantized-matmul_transform_8_4
  # linalg_quantized-matmul_transform_16_2

  linalg_quantized-matmul_autovec_8
  linalg_quantized-matmul_autovec_16
  linalg_quantized-matmul_autovec_32
  linalg_quantized-matmul_autovec_64
)
