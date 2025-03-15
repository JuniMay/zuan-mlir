add_executable(linalg-dot-fp16-benchmark launcher.cpp)
target_link_libraries(linalg-dot-fp16-benchmark PRIVATE GoogleBenchmark)

add_linalg_zuan_kernel(dot-fp16 16 2)

add_linalg_transform_kernel(dot-fp16 16 1)

add_linalg_autovec_kernel(dot-fp16 8)
add_linalg_autovec_kernel(dot-fp16 16)
add_linalg_autovec_kernel(dot-fp16 32)
add_linalg_autovec_kernel(dot-fp16 64)

target_link_libraries(
  linalg-dot-fp16-benchmark
  PRIVATE
  linalg_dot-fp16_zuan_16_2
  
  linalg_dot-fp16_transform_16_1

  linalg_dot-fp16_autovec_8
  linalg_dot-fp16_autovec_16
  linalg_dot-fp16_autovec_32
  linalg_dot-fp16_autovec_64
)
