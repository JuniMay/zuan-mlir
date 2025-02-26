# This defines the `MLIRZuanOpsIncGen` target in cmake.
add_mlir_dialect(ZuanOps zuan)
add_mlir_doc(ZuanOps ZuanOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS ZuanAttributes.td)
mlir_tablegen(ZuanEnums.h.inc -gen-enum-decls)
mlir_tablegen(ZuanEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ZuanAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ZuanAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRZuanAttributesIncGen)