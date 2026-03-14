get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(
  LIBS

  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
  VPToLLVMIRTranslation
)
add_llvm_executable(dyno-translate dyno-translate.cpp)

llvm_update_compile_flags(dyno-translate)
target_link_libraries(dyno-translate PRIVATE ${LIBS})

mlir_check_all_link_libraries(dyno-translate)
