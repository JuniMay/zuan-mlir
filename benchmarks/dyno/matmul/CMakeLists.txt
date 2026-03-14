add_executable(dyno-matmul-benchmark launcher.cpp naive.cpp)
target_link_libraries(dyno-matmul-benchmark PRIVATE GoogleBenchmark)

set_source_files_properties(naive.cpp PROPERTIES COMPILE_FLAGS "-O3")

dyno_add_direct_dyno_kernel(
  TARGET dyno-matmul-kernel
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir
  VF 16
  UF 2
  EXTRA_FLAGS -O3 -g ${CLANG_FLAGS_LIST}
)

target_link_libraries(dyno-matmul-benchmark PRIVATE dyno-matmul-kernel)
