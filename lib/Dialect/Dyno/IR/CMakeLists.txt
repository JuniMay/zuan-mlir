add_mlir_dialect_library(
  DynoDialect

  DynoDialect.cpp
  DynoTypes.cpp
  DynoOps.cpp

  DEPENDS
  MLIRDynoOpsIncGen
  MLIRDynoAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  DynoUtils
)
