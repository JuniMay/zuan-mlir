#-------------------------------------------------------------------------------
# Helper CMake Functions
#-------------------------------------------------------------------------------

set(LLVM_BINARY_DIR ${ZUAN_SOURCE_DIR}/llvm-project/build/bin)

# `linalg_${benchmark_name}_zuan_${vf}_${uf}`
function(add_linalg_zuan_kernel benchmark_name vf uf)
  if(${ARGC} GREATER 3 AND ARGV3 STREQUAL "TRUE")
    set(estimate_math_flag ";enable-math-estimation=true")
    set(estimate_math_suffix "_est")
  else()
    set(estimate_math_flag ";enable-math-estimation=false")
    set(estimate_math_suffix "")
  endif()

  set(mlir_file ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark_name}.mlir)
  set(generated_file_prefix ${CMAKE_CURRENT_BINARY_DIR}/${benchmark_name}_zuan_${vf}_${uf}${estimate_math_suffix})
  
  set(zuan_mlir ${generated_file_prefix}.mlir)
  set(zuan_llvm ${generated_file_prefix}.ll)
  set(zuan_debug_file ${generated_file_prefix}_debug.mlir)
  set(zuan_obj ${generated_file_prefix}.o)

  add_custom_command(
    OUTPUT ${zuan_mlir} ${zuan_llvm} ${zuan_obj}
    COMMAND
      cat ${mlir_file} |
      sed "'s/kernel(/kernel_zuan_${vf}_${uf}${estimate_math_suffix}(/'" |
      ${ZUAN_OPT}
        -linalg-generalize-named-ops
        -convert-linalg-to-zuan
        -lower-zuan="target-rank=2"
        -zuan-stripmining="vf=${vf};uf=${uf};scalable=true"
        -zuan-strength-reduction
        -canonicalize
        -cse
        -convert-zuan-to-vp="vf=${vf};scalable=true"
        -convert-vp-to-llvm="${estimate_math_flag}"
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

  add_library(linalg_${benchmark_name}_zuan_${vf}_${uf}${estimate_math_suffix} STATIC ${zuan_obj})
  set_target_properties(linalg_${benchmark_name}_zuan_${vf}_${uf}${estimate_math_suffix} PROPERTIES LINKER_LANGUAGE CXX)
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
        -convert-vector-to-scf="full-unroll;target-rank=1"
        -convert-vector-to-llvm="reassociate-fp-reductions"
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
      ${LLVM_BINARY_DIR}/clang
        -march=rv64gcv_zvfh -O3 -ffast-math -mllvm --riscv-v-vector-bits-min=256
        ${CLANG_FLAGS_LIST} ${autovec_llvm} -c -save-temps -o ${autovec_obj}
    DEPENDS ${mlir_file}
  )

  add_library(linalg_${benchmark_name}_autovec_${virtvecsize} STATIC ${autovec_obj})
  set_target_properties(linalg_${benchmark_name}_autovec_${virtvecsize} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

# `linalg_${benchmark_name}_transform_${vf}_${uf}`
function(add_linalg_transform_kernel benchmark_name vf uf)
  set(mlir_file ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark_name}.mlir)

  set(transform_file ${CMAKE_CURRENT_SOURCE_DIR}/transform.txt)
  set(generated_file_prefix ${CMAKE_CURRENT_BINARY_DIR}/${benchmark_name}_transform_${vf}_${uf})

  set(transform_mlir ${generated_file_prefix}.mlir)
  set(transform_llvm ${generated_file_prefix}.ll)
  set(transform_debug_file ${generated_file_prefix}_debug.mlir)
  set(transform_obj ${generated_file_prefix}.o)

  # The file that has `VF` and `UF` to be replaced
  set(transform_sed ${generated_file_prefix}.transform.mlir)
  # Firstly, sed the `VF` and `UF` in the transform.mlir file
  add_custom_command(
    OUTPUT ${transform_sed}
    COMMAND
      cat ${transform_file} |
      sed "'s/VF/${vf}/'" |
      sed "'s/UF/${uf}/'" > ${transform_sed}
    DEPENDS ${transform_file}
  )

  add_custom_command(
    OUTPUT ${transform_mlir} ${transform_llvm} ${transform_obj}
    COMMAND
      cat ${mlir_file} |
      sed "'s/kernel(/kernel_transform_${vf}_${uf}(/'" |
      ${LLVM_BINARY_DIR}/mlir-opt
        -transform-preload-library="transform-library-paths=${transform_sed}"
        -transform-interpreter
        -canonicalize
        -lower-vector-mask
        -convert-vector-to-scf="full-unroll;target-rank=1;lower-scalable"
        -convert-scf-to-cf
        -convert-cf-to-llvm
        -expand-strided-metadata
        -convert-math-to-llvm
        -convert-math-to-libm
        -convert-vector-to-llvm="reassociate-fp-reductions"
        -convert-ub-to-llvm
        -llvm-request-c-wrappers
        -finalize-memref-to-llvm
        -lower-affine
        -convert-arith-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts
        -mlir-print-ir-after-all 2> ${transform_debug_file} |
        ${LLVM_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o ${transform_llvm}
    COMMAND
      ${LLVM_BINARY_DIR}/clang
        -march=rv64gcv_zvfh -O3 -ffast-math -mllvm --riscv-v-vector-bits-min=256
        ${CLANG_FLAGS_LIST} ${transform_llvm} -c -save-temps -o ${transform_obj}
    DEPENDS ${mlir_file} ${transform_sed}
  )

  add_library(linalg_${benchmark_name}_transform_${vf}_${uf} STATIC ${transform_obj})
  set_target_properties(linalg_${benchmark_name}_transform_${vf}_${uf} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

add_subdirectory(conv_2d)
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
