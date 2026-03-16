add_linalg_runtime_regression(
  NAME runtime-linalg-reduce-parallel
  MLIR_FILE ${DYNO_SOURCE_DIR}/benchmarks/linalg/reduce/reduce.mlir
  DRIVER ${CMAKE_CURRENT_SOURCE_DIR}/driver.cpp
  DYNO_VF 16
  DYNO_UF 1
  DYNO_REDUCTION_MODE parallel
  DYNO_FP_POLICY relaxed
  EPSILON 0.0001
)
