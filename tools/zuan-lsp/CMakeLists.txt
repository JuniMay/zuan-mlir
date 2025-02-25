get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(
  LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRLspServerLib
)
add_llvm_executable(zuan-lsp zuan-lsp.cpp)

llvm_update_compile_flags(zuan-lsp)
target_link_libraries(zuan-lsp PRIVATE ${LIBS})

mlir_check_all_link_libraries(zuan-lsp)
