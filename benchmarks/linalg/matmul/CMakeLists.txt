add_executable(linalg-matmul-benchmark launcher.cpp)
target_link_libraries(linalg-matmul-benchmark PRIVATE GoogleBenchmark)

separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

set(mlir_file ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir)

set(scalar_ll_file ${CMAKE_CURRENT_BINARY_DIR}/matmul_scalar.ll)
set(scalar_obj_file ${CMAKE_CURRENT_BINARY_DIR}/matmul_scalar.o)
add_custom_command(
  OUTPUT ${scalar_obj_file}
  COMMAND
    cat ${mlir_file} |
    sed "s/kernel/kernel_scalar/" |
    ${ZUAN_BINARY_DIR}/bin/zuan-opt
      -convert-linalg-to-affine-loops
      -lower-affine
      -convert-vector-to-scf
      -convert-scf-to-cf
      -convert-cf-to-llvm 
      -convert-vector-to-llvm
      -expand-strided-metadata
      -convert-math-to-llvm
      -convert-math-to-libm
      -convert-arith-to-llvm
      -llvm-request-c-wrappers
      -finalize-memref-to-llvm
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${ZUAN_BINARY_DIR}/bin/zuan-translate -zuan-to-llvmir -o ${scalar_ll_file}
  COMMAND
  ${ZUAN_SOURCE_DIR}/llvm-project/build/bin/clang -O3 ${CLANG_FLAGS_LIST} ${scalar_ll_file} -c -save-temps -o ${scalar_obj_file}
  DEPENDS ${mlir_file} zuan-opt
)

set(zuan_ll_file ${CMAKE_CURRENT_BINARY_DIR}/matmul_zuan.ll)
set(zuan_obj_file ${CMAKE_CURRENT_BINARY_DIR}/matmul_zuan.o)

add_custom_command(
  OUTPUT ${zuan_obj_file}
  COMMAND 
    cat ${mlir_file} |
    sed "s/kernel/kernel_zuan/" |
    ${ZUAN_BINARY_DIR}/bin/zuan-opt -convert-linalg-to-zuan -o ${CMAKE_CURRENT_BINARY_DIR}/linalg-to-zuan.mlir
  COMMAND ${ZUAN_BINARY_DIR}/bin/zuan-opt -lower-zuan="target-rank=2" -o ${CMAKE_CURRENT_BINARY_DIR}/stage0.mlir ${CMAKE_CURRENT_BINARY_DIR}/linalg-to-zuan.mlir
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
  COMMAND ${ZUAN_BINARY_DIR}/bin/zuan-translate ${CMAKE_CURRENT_BINARY_DIR}/stage4.mlir -zuan-to-llvmir -o ${zuan_ll_file}
  COMMAND ${ZUAN_SOURCE_DIR}/llvm-project/build/bin/clang -march=rv64gcv_zvfh -O3 -g ${CLANG_FLAGS_LIST} -c -save-temps ${zuan_ll_file} -o ${zuan_obj_file}
  DEPENDS ${mlir_file} zuan-opt
)

add_library(linalg-matmul-kernel STATIC ${zuan_obj_file} ${scalar_obj_file})
set_target_properties(linalg-matmul-kernel PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(linalg-matmul-benchmark PRIVATE linalg-matmul-kernel)
