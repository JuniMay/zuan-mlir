#-------------------------------------------------------------------------------
# Helper CMake Functions
#-------------------------------------------------------------------------------

set(LLVM_BINARY_DIR ${ZUAN_SOURCE_DIR}/llvm-project/build/bin)

# This function defines triton_`benchmark_name`_kernels library, which includes
# the generated object files for the Zuan and Triton CPU kernels.
# 
# The `xxx_kernel` names in IRs will be replaced with `kernel_zuan` and
# `kernel_triton_cpu` respectively.
function(add_triton_benchmark benchmark_name python_args)
  set(python_file ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark_name}.py)

  set(generated_file_prefix ${CMAKE_CURRENT_BINARY_DIR}/${benchmark_name})
  set(ttir_file ${generated_file_prefix}.ttir)
  set(mlir_file ${generated_file_prefix}.mlir)
  set(mlir_debug_file ${generated_file_prefix}_debug.mlir)
  set(launcher_header ${generated_file_prefix}.h)

  set(zuan_mlir ${generated_file_prefix}_zuan.mlir)
  set(zuan_llvm ${generated_file_prefix}_zuan.ll)
  set(zuan_debug_file ${generated_file_prefix}_zuan_debug.mlir)
  set(zuan_obj ${generated_file_prefix}_zuan.o)

  set(triton_cpu_mlir ${generated_file_prefix}_cpu.mlir)
  set(triton_cpu_llvm ${generated_file_prefix}_cpu.ll)
  set(triton_cpu_obj ${generated_file_prefix}_cpu.o)

  #-----------------------------------------------------------------------------
  # Step 1: Generate the TTIR file
  #-----------------------------------------------------------------------------
  add_custom_command(
    OUTPUT ${ttir_file} ${launcher_header}
    COMMAND 
      PYTHONPATH=$ENV{PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/..
      ${Python3_EXECUTABLE} ${python_file} -o ${ttir_file} ${python_args}
    DEPENDS ${python_file}
  )

  #-----------------------------------------------------------------------------
  # Step 2: Convert TTIR to Linalg and then to Zuan
  #-----------------------------------------------------------------------------
  add_custom_command(
    OUTPUT ${mlir_file}
    COMMAND 
      ${TRITON_SHARED_OPT}
      # -triton-to-linalg-experimental 
      -triton-to-structured
      -cse
      -canonicalize
      -triton-to-unstructured
      -triton-arith-to-linalg
      -structured-to-memref
      -unstructured-to-memref
      -triton-ptr-to-memref
      -reconcile-unrealized-casts
      -cse
      -canonicalize
      -mlir-print-ir-after-all
      -o ${mlir_file} ${ttir_file} 2> ${mlir_debug_file}
    DEPENDS ${ttir_file}
  )

  add_custom_command(
    OUTPUT ${zuan_mlir} ${zuan_llvm} ${zuan_obj}
    COMMAND
      cat ${mlir_file} |
      sed "'s/kernel(/kernel_zuan(/'" |
      ${ZUAN_OPT}
        -linalg-fuse-elementwise-ops 
        -empty-tensor-to-alloc-tensor 
        -one-shot-bufferize="allow-return-allocs-from-loops=true"
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

  #-----------------------------------------------------------------------------
  # Step 3: Convert TTIR to Triton CPU
  #-----------------------------------------------------------------------------
  add_custom_command(
    OUTPUT ${triton_cpu_mlir} ${triton_cpu_llvm} ${triton_cpu_obj}
    COMMAND
      cat ${ttir_file} |
      sed "'s/kernel(/kernel_triton_cpu(/'" |
      ${TRITON_CPU_OPT}
        -triton-combine
        -canonicalize
        -triton-reorder-broadcast
        -cse
        -loop-invariant-code-motion 
        -symbol-dce
        # TTIR -> TTCIR
        -triton-cpu-scalarize="skip-gather-scatter=true"
        -triton-cpu-convert-memory-ops="use-gather-scatter=true"
        -triton-cpu-convert-ptr-ops
        -triton-cpu-convert-elementwise-ops
        -triton-cpu-convert-elem-manip-ops
        -triton-cpu-convert-dot-op
        -triton-cpu-convert-histogram-op
        -triton-cpu-convert-reduction="use-multidim-reduction-op=false;use-reduction-op=true"
        -triton-cpu-convert-scan
        -triton-cpu-convert-control-flow-op
        -triton-cpu-convert-atomic-ops
        -triton-cpu-convert-debug-ops
        -cse
        -symbol-dce
        -canonicalize
        # TTCIR -> Target TTCIR
        -triton-cpu-canonicalize
        -triton-cpu-optimize-masks
        -canonicalize
        -triton-cpu-convert-dot-generic
        -triton-cpu-add-casts-for-unsupported-ops="promote-bf16-to-fp32=false;convert-mixed-precision-matmul=true;promote-lib-math-to-fp32=true"
        -triton-cpu-decompose-fp-conversions="decompose-bf16-conversions=false;decompose-fp8-conversions=true"
        -cse
        -symbol-dce
        -canonicalize
        # Target TTCIR -> LLVM
        -triton-cpu-lower-multi-reduction
        -expand-strided-metadata
        -convert-vector-to-scf="full-unroll=true;target-rank=1;lower-tensors=false"
        -lower-affine
        -convert-scf-to-cf
        -convert-index-to-llvm
        -triton-cpu-func-op-to-llvm
        -triton-cpu-get-program-id-op-to-llvm
        -triton-cpu-memory-op-to-llvm
        -triton-cpu-atomic-ops-to-llvm
        -triton-cpu-debug-ops-to-llvm 
        -triton-cpu-math-to-vec-lib="lib=sleef"
        -convert-math-to-llvm
        -convert-math-to-libm
        -convert-vector-to-llvm
        -finalize-memref-to-llvm
        -convert-arith-to-llvm
        -convert-func-to-llvm
        -convert-ub-to-llvm
        -canonicalize
        -cse
        -symbol-dce -o ${triton_cpu_mlir}
    COMMAND
      ${LLVM_BINARY_DIR}/mlir-translate ${triton_cpu_mlir} -mlir-to-llvmir -o ${triton_cpu_llvm}
    COMMAND
      ${LLVM_BINARY_DIR}/clang 
        -march=rv64gcv_zvfh -O2 -g ${CLANG_FLAGS_LIST} -c -save-temps ${triton_cpu_llvm} -o ${triton_cpu_obj}
    DEPENDS ${mlir_file} zuan-translate
  )

  add_library(triton_${benchmark_name}_kernels STATIC ${zuan_obj} ${triton_cpu_obj})
  set_target_properties(triton_${benchmark_name}_kernels PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

add_subdirectory(matmul)
add_subdirectory(vector_add)
