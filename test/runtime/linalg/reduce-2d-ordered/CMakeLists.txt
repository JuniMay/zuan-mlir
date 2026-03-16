add_linalg_runtime_regression(
  NAME runtime-linalg-reduce-2d-ordered
  MLIR_FILE ${DYNO_SOURCE_DIR}/benchmarks/linalg/reduce-2d/reduce-2d.mlir
  DRIVER ${CMAKE_CURRENT_SOURCE_DIR}/driver.cpp
  DYNO_VF 16
  DYNO_UF 2
  DYNO_REDUCTION_MODE sequential
  DYNO_FP_POLICY strict
  EPSILON 0.0001
)
