add_linalg_runtime_regression(
  NAME runtime-linalg-relu
  MLIR_FILE ${CMAKE_CURRENT_SOURCE_DIR}/kernel.mlir
  DRIVER ${CMAKE_CURRENT_SOURCE_DIR}/driver.cpp
  DYNO_VF 16
  DYNO_UF 2
  DYNO_REDUCTION_MODE sequential
  DYNO_FP_POLICY strict
  EPSILON 0.0
)
