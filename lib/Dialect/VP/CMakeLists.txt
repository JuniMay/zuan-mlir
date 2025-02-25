add_mlir_dialect_library(
  VPDialect

  VPDialect.cpp
  VPOps.cpp

  DEPENDS
  MLIRVPOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
