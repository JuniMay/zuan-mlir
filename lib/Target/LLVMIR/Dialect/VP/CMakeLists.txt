add_mlir_translation_library(
  VPToLLVMIRTranslation

  VPToLLVMIRTranslation.cpp

  DEPENDS
  MLIRVPOpsConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  VPDialect
)