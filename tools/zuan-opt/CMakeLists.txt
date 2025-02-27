get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(
  LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  VPToLLVMConversion
  LinalgToZuanConversion
)

add_llvm_executable(zuan-opt zuan-opt.cpp)

llvm_update_compile_flags(zuan-opt)
target_link_libraries(zuan-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(zuan-opt)
