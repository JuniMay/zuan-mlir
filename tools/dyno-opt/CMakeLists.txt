get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(
  LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses

  VPToLLVMConversion
  LinalgToDynoConversion
  LowerDyno
  DynoToVPConversion
  DynoTransforms
)

add_llvm_executable(dyno-opt dyno-opt.cpp)

llvm_update_compile_flags(dyno-opt)
target_link_libraries(dyno-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(dyno-opt)
