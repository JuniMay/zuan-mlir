add_executable(linalg-matmul-transpose-a-benchmark launcher.cpp)
target_link_libraries(linalg-matmul-transpose-a-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(matmul-transpose-a 8 4)
add_linalg_zuan_kernel(matmul-transpose-a 16 2)

add_linalg_transform_kernel(matmul-transpose-a 8 4)
add_linalg_transform_kernel(matmul-transpose-a 16 2)

add_linalg_autovec_kernel(matmul-transpose-a 8)
add_linalg_autovec_kernel(matmul-transpose-a 16)
add_linalg_autovec_kernel(matmul-transpose-a 32)
add_linalg_autovec_kernel(matmul-transpose-a 64)

target_link_libraries(
  linalg-matmul-transpose-a-benchmark
  PRIVATE
  linalg_matmul-transpose-a_zuan_16_2
  linalg_matmul-transpose-a_zuan_8_4

  linalg_matmul-transpose-a_transform_16_2
  linalg_matmul-transpose-a_transform_8_4

  linalg_matmul-transpose-a_autovec_8
  linalg_matmul-transpose-a_autovec_16
  linalg_matmul-transpose-a_autovec_32
  linalg_matmul-transpose-a_autovec_64
)
