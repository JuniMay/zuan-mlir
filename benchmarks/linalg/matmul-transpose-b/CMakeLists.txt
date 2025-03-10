add_executable(linalg-matmul-transpose-b-benchmark launcher.cpp)
target_link_libraries(linalg-matmul-transpose-b-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(matmul-transpose-b 8 4)
add_linalg_zuan_kernel(matmul-transpose-b 16 2)

add_linalg_autovec_kernel(matmul-transpose-b 8)
add_linalg_autovec_kernel(matmul-transpose-b 16)
add_linalg_autovec_kernel(matmul-transpose-b 32)
add_linalg_autovec_kernel(matmul-transpose-b 64)

target_link_libraries(
  linalg-matmul-transpose-b-benchmark
  PRIVATE
  linalg_matmul-transpose-b_zuan_16_2
  linalg_matmul-transpose-b_zuan_8_4
  linalg_matmul-transpose-b_autovec_8
  linalg_matmul-transpose-b_autovec_16
  linalg_matmul-transpose-b_autovec_32
  linalg_matmul-transpose-b_autovec_64
)
