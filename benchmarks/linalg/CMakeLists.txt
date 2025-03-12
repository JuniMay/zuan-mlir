#-------------------------------------------------------------------------------
# Helper CMake Functions
#-------------------------------------------------------------------------------

set(LLVM_BINARY_DIR ${ZUAN_SOURCE_DIR}/llvm-project/build/bin)

# `linalg_${benchmark_name}_zuan_${vf}_${uf}`
function(add_linalg_zuan_kernel benchmark_name vf uf)
  set(mlir_file ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark_name}.mlir)
  set(generated_file_prefix ${CMAKE_CURRENT_BINARY_DIR}/${benchmark_name}_zuan_${vf}_${uf})
  
  set(zuan_mlir ${generated_file_prefix}.mlir)
  set(zuan_llvm ${generated_file_prefix}.ll)
  set(zuan_debug_file ${generated_file_prefix}_debug.mlir)
  set(zuan_obj ${generated_file_prefix}.o)

  add_custom_command(
    OUTPUT ${zuan_mlir} ${zuan_llvm} ${zuan_obj}
    COMMAND
      cat ${mlir_file} |
      sed "'s/kernel(/kernel_zuan_${vf}_${uf}(/'" |
      ${ZUAN_OPT}
        -linalg-generalize-named-ops
        -convert-linalg-to-zuan
        -lower-zuan="target-rank=2"
        -zuan-stripmining="vf=${vf};uf=${uf};scalable=true"
        -zuan-strength-reduction
        -canonicalize
        -cse
        -convert-zuan-to-vp="vf=${vf};scalable=true"
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
        -march=rv64gcv_zvfh -O3 -ffast-math -g ${CLANG_FLAGS_LIST} -c -save-temps ${zuan_llvm} -o ${zuan_obj}
    DEPENDS ${mlir_file} zuan-opt zuan-translate
  )

  add_library(linalg_${benchmark_name}_zuan_${vf}_${uf} STATIC ${zuan_obj})
  set_target_properties(linalg_${benchmark_name}_zuan_${vf}_${uf} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

# `linalg_${benchmark_name}_autovec_${virtvecsize}`
function(add_linalg_autovec_kernel benchmark_name virtvecsize)
  set(mlir_file ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark_name}.mlir)
  set(generated_file_prefix ${CMAKE_CURRENT_BINARY_DIR}/${benchmark_name}_autovec_${virtvecsize})
  
  set(autovec_mlir ${generated_file_prefix}.mlir)
  set(autovec_llvm ${generated_file_prefix}.ll)
  set(autovec_debug_file ${generated_file_prefix}_debug.mlir)
  set(autovec_obj ${generated_file_prefix}.o)

  add_custom_command(
    OUTPUT ${autovec_mlir} ${autovec_llvm} ${autovec_obj}
    COMMAND
      cat ${mlir_file} |
      sed "'s/kernel(/kernel_autovec_${virtvecsize}(/'" |
      ${LLVM_BINARY_DIR}/mlir-opt
        -convert-linalg-to-affine-loops
        -affine-super-vectorize="virtual-vector-size=${virtvecsize};vectorize-reductions"
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
      ${LLVM_BINARY_DIR}/clang -march=rv64gcv_zvfh -O3 -ffast-math ${CLANG_FLAGS_LIST} ${autovec_llvm} -c -save-temps -o ${autovec_obj}
    DEPENDS ${mlir_file} zuan-opt
  )

  add_library(linalg_${benchmark_name}_autovec_${virtvecsize} STATIC ${autovec_obj})
  set_target_properties(linalg_${benchmark_name}_autovec_${virtvecsize} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

add_subdirectory(dot-fp16)
add_subdirectory(exp)
add_subdirectory(fill-rng-2d)
add_subdirectory(matmul)
add_subdirectory(matmul-fp16)
add_subdirectory(matmul-transpose-b)
add_subdirectory(matmul-transpose-a)
add_subdirectory(mmt4d)
add_subdirectory(quantized-matmul)
add_subdirectory(reduce)
add_subdirectory(reduce-2d)
add_subdirectory(relu)
add_subdirectory(rsqrt)
