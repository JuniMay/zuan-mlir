add_executable(linalg-mmt4d-benchmark launcher.cpp)
target_link_libraries(linalg-mmt4d-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(mmt4d 8 4)
add_linalg_zuan_kernel(mmt4d 16 2)

# XXX: mmt4d failed to be vectorized, using non-scalable vector here.
# vscale = 256 / 64 = 4 on spacemit K1.
add_linalg_transform_kernel(mmt4d 32 4)
add_linalg_transform_kernel(mmt4d 64 2)

add_linalg_autovec_kernel(mmt4d 8)
add_linalg_autovec_kernel(mmt4d 16)
add_linalg_autovec_kernel(mmt4d 32)
add_linalg_autovec_kernel(mmt4d 64)

target_link_libraries(
  linalg-mmt4d-benchmark
  PRIVATE
  
  linalg_mmt4d_zuan_8_4
  linalg_mmt4d_zuan_16_2

  linalg_mmt4d_transform_32_4
  linalg_mmt4d_transform_64_2

  linalg_mmt4d_autovec_8
  linalg_mmt4d_autovec_16
  linalg_mmt4d_autovec_32
  linalg_mmt4d_autovec_64
)
