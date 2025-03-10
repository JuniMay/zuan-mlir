add_executable(linalg-mmt4d-benchmark launcher.cpp)
target_link_libraries(linalg-mmt4d-benchmark PRIVATE GoogleBenchmark)

# add_linalg_benchmark(mmt4d)
add_linalg_zuan_kernel(mmt4d 8 4)
add_linalg_zuan_kernel(mmt4d 16 2)
add_linalg_autovec_kernel(mmt4d 8)
add_linalg_autovec_kernel(mmt4d 16)
add_linalg_autovec_kernel(mmt4d 32)
add_linalg_autovec_kernel(mmt4d 64)

target_link_libraries(
  linalg-mmt4d-benchmark
  PRIVATE
  
  linalg_mmt4d_zuan_8_4
  linalg_mmt4d_zuan_16_2

  linalg_mmt4d_autovec_8
  linalg_mmt4d_autovec_16
  linalg_mmt4d_autovec_32
  linalg_mmt4d_autovec_64
)
