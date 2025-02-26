add_mlir_dialect_library(
  ZuanDialect

  ZuanDialect.cpp
  ZuanTypes.cpp
  ZuanOps.cpp

  DEPENDS
  MLIRZuanOpsIncGen
  MLIRZuanAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
