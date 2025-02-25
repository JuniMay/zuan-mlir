# This defines the `MLIRVPOpsIncGen` target in cmake.
add_mlir_dialect(VPOps vp)
add_mlir_doc(VPOps VPOps Dialects/ -gen-op-doc)

# LLVMIR Conversion Library
set(LLVM_TARGET_DEFINITIONS VPOps.td)
mlir_tablegen(VPOpsConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRVPOpsConversionsIncGen)
