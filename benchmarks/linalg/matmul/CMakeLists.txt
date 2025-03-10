add_executable(linalg-matmul-benchmark launcher.cpp)
target_link_libraries(linalg-matmul-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(matmul 8 4)
add_linalg_zuan_kernel(matmul 16 2)

add_linalg_autovec_kernel(matmul 8)
add_linalg_autovec_kernel(matmul 16)
add_linalg_autovec_kernel(matmul 32)
add_linalg_autovec_kernel(matmul 64)

target_link_libraries(
  linalg-matmul-benchmark
  PRIVATE
  linalg_matmul_zuan_16_2
  linalg_matmul_zuan_8_4
  linalg_matmul_autovec_8
  linalg_matmul_autovec_16
  linalg_matmul_autovec_32
  linalg_matmul_autovec_64
)
