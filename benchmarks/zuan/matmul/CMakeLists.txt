add_executable(zuan-matmul-benchmark launcher.cpp naive.cpp)
target_link_libraries(zuan-matmul-benchmark PRIVATE GoogleBenchmark)

set_source_files_properties(naive.cpp PROPERTIES COMPILE_FLAGS "-O3")

separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

set(mlir_file ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir)
set(ll_file ${CMAKE_CURRENT_BINARY_DIR}/matmul.ll)
set(obj_file ${CMAKE_CURRENT_BINARY_DIR}/matmul.o)
set(asm_file ${CMAKE_CURRENT_BINARY_DIR}/matmul.s)

add_custom_command(
  OUTPUT ${obj_file}
  COMMAND ${ZUAN_BINARY_DIR}/bin/zuan-opt -lower-zuan="target-rank=2" -o ${CMAKE_CURRENT_BINARY_DIR}/stage0.mlir ${mlir_file}
  COMMAND ${ZUAN_BINARY_DIR}/bin/zuan-opt -zuan-stripmining="vf=16;uf=2;scalable=true" -o ${CMAKE_CURRENT_BINARY_DIR}/stage1.mlir ${CMAKE_CURRENT_BINARY_DIR}/stage0.mlir 
  COMMAND ${ZUAN_BINARY_DIR}/bin/zuan-opt -convert-zuan-to-vp="vf=16;scalable=true" -o ${CMAKE_CURRENT_BINARY_DIR}/stage2.mlir ${CMAKE_CURRENT_BINARY_DIR}/stage1.mlir
  COMMAND ${ZUAN_BINARY_DIR}/bin/zuan-opt -convert-vp-to-llvm -o ${CMAKE_CURRENT_BINARY_DIR}/stage3.mlir ${CMAKE_CURRENT_BINARY_DIR}/stage2.mlir
  COMMAND ${ZUAN_BINARY_DIR}/bin/zuan-opt
    -expand-strided-metadata 
    -lower-affine 
    -convert-scf-to-cf 
    -convert-cf-to-llvm 
    -convert-vector-to-llvm 
    -convert-arith-to-llvm 
    -convert-math-to-llvm
    -convert-math-to-libm
    -convert-vector-to-llvm
    -llvm-request-c-wrappers
    -finalize-memref-to-llvm 
    -convert-func-to-llvm
    -reconcile-unrealized-casts 
    -canonicalize 
    -cse -o ${CMAKE_CURRENT_BINARY_DIR}/stage4.mlir ${CMAKE_CURRENT_BINARY_DIR}/stage3.mlir
  COMMAND ${ZUAN_BINARY_DIR}/bin/zuan-translate ${CMAKE_CURRENT_BINARY_DIR}/stage4.mlir -zuan-to-llvmir -o ${ll_file}
  COMMAND ${ZUAN_SOURCE_DIR}/llvm-project/build/bin/clang -march=rv64gcv_zvfh -O3 -g ${CLANG_FLAGS_LIST} -c -save-temps ${ll_file} -o ${obj_file}
  DEPENDS ${mlir_file} zuan-opt
)

add_library(zuan-matmul-kernel STATIC ${obj_file})
set_target_properties(zuan-matmul-kernel PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(zuan-matmul-benchmark PRIVATE zuan-matmul-kernel)
