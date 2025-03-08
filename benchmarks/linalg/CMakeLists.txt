#-------------------------------------------------------------------------------
# Helper CMake Functions
#-------------------------------------------------------------------------------

set(LLVM_BINARY_DIR ${ZUAN_SOURCE_DIR}/llvm-project/build/bin)

# This function defines `linalg_${benchmark_name}_kernels` library target.
function(add_linalg_benchmark benchmark_name)
  set(generated_file_prefix ${CMAKE_CURRENT_BINARY_DIR}/${benchmark_name})
  
  set(mlir_file ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark_name}.mlir)

  set(zuan_mlir ${generated_file_prefix}_zuan.mlir)
  set(zuan_llvm ${generated_file_prefix}_zuan.ll)
  set(zuan_debug_file ${generated_file_prefix}_zuan_debug.mlir)
  set(zuan_obj ${generated_file_prefix}_zuan.o)

  set(autovec_mlir ${generated_file_prefix}_autovec.mlir)
  set(autovec_llvm ${generated_file_prefix}_autovec.ll)
  set(autovec_debug_file ${generated_file_prefix}_autovec_debug.mlir)
  set(autovec_obj ${generated_file_prefix}_autovec.o)

  add_custom_command(
    OUTPUT ${zuan_mlir} ${zuan_llvm} ${zuan_obj}
    COMMAND
      cat ${mlir_file} |
      sed "'s/kernel(/kernel_zuan(/'" |
      ${ZUAN_OPT}
        -convert-linalg-to-zuan
        -lower-zuan="target-rank=2"
        -zuan-stripmining="vf=16;uf=2;scalable=true"
        -canonicalize
        -cse
        -convert-zuan-to-vp="vf=16;scalable=true"
        -convert-vp-to-llvm
        -canonicalize
        -cse
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
        -cse
        -mlir-print-ir-after-all
        -o ${zuan_mlir} 2> ${zuan_debug_file}
    COMMAND
      ${ZUAN_TRANSLATE} ${zuan_mlir} -zuan-to-llvmir -o ${zuan_llvm}
    COMMAND
      ${LLVM_BINARY_DIR}/clang 
        -march=rv64gcv_zvfh -O2 -g ${CLANG_FLAGS_LIST} -c -save-temps ${zuan_llvm} -o ${zuan_obj}
    DEPENDS ${mlir_file} zuan-opt zuan-translate
  )

  add_custom_command(
    OUTPUT ${autovec_mlir} ${autovec_llvm} ${autovec_obj}
    COMMAND
      cat ${mlir_file} |
      sed "'s/kernel(/kernel_autovec(/'" |
      ${LLVM_BINARY_DIR}/mlir-opt
        -convert-linalg-to-affine-loops
        -affine-super-vectorize="virtual-vector-size=16;vectorize-reductions"
        -convert-vector-to-scf="full-unroll"
        -convert-vector-to-llvm
        -lower-affine
        -convert-scf-to-cf
        -convert-cf-to-llvm 
        -convert-ub-to-llvm
        -expand-strided-metadata
        -convert-math-to-llvm
        -convert-math-to-libm
        -convert-arith-to-llvm
        -llvm-request-c-wrappers
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts
        -mlir-print-ir-after-all 2> ${autovec_debug_file} |
      ${LLVM_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o ${autovec_llvm}
    COMMAND
      ${LLVM_BINARY_DIR}/clang -march=rv64gcv_zvfh -O3 ${CLANG_FLAGS_LIST} ${autovec_llvm} -c -save-temps -o ${autovec_obj}
    DEPENDS ${mlir_file} zuan-opt 
  )

  add_library(linalg_${benchmark_name}_kernels STATIC ${zuan_obj} ${autovec_obj})
  set_target_properties(linalg_${benchmark_name}_kernels PROPERTIES LINKER_LANGUAGE CXX)

endfunction()


add_subdirectory(matmul)
add_subdirectory(matmul-transpose-b)
add_subdirectory(matmul-transpose-a)

