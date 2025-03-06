add_library(
  BenchmarkCommon

  OBJECT
  MemRefUtils.cpp 
)

target_include_directories(
  BenchmarkCommon

  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
