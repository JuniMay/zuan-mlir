# This defines the `MLIRDynoOpsIncGen` target in cmake.
add_mlir_dialect(DynoOps dyno)
add_mlir_doc(DynoOps DynoOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS DynoAttributes.td)
mlir_tablegen(DynoEnums.h.inc -gen-enum-decls)
mlir_tablegen(DynoEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(DynoAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(DynoAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRDynoAttributesIncGen)
