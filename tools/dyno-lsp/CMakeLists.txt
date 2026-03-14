get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(
  LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRLspServerLib
)
add_llvm_executable(dyno-lsp dyno-lsp.cpp)

llvm_update_compile_flags(dyno-lsp)
target_link_libraries(dyno-lsp PRIVATE ${LIBS})

mlir_check_all_link_libraries(dyno-lsp)
