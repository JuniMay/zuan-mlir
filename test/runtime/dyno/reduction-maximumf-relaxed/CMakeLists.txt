add_dyno_runtime_regression(
  NAME runtime-dyno-reduction-maximumf-relaxed
  MLIR_FILE ${CMAKE_CURRENT_SOURCE_DIR}/kernel.mlir
  DRIVER ${CMAKE_CURRENT_SOURCE_DIR}/driver.cpp
  DYNO_VF 16
  DYNO_UF 1
  DYNO_REDUCTION_MODE parallel
  DYNO_FP_POLICY relaxed
  EPSILON 0.0
)
