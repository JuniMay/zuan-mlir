# This function defines triton_`benchmark_name`_kernels library, which includes
# the generated object files for the scalar reference, Dyno, and Triton CPU
# kernels.
function(add_triton_benchmark benchmark_name python_args)
  set(python_file ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark_name}.py)

  set(generated_file_prefix ${CMAKE_CURRENT_BINARY_DIR}/${benchmark_name})
  set(ttir_file ${generated_file_prefix}.ttir)
  set(mlir_file ${generated_file_prefix}.mlir)
  set(launcher_header ${generated_file_prefix}.h)

  set(dyno_mlir ${generated_file_prefix}_dyno.mlir)
  set(dyno_llvm ${generated_file_prefix}_dyno.ll)
  set(dyno_obj ${generated_file_prefix}_dyno.o)

  set(scalar_mlir ${generated_file_prefix}_scalar.mlir)
  set(scalar_llvm ${generated_file_prefix}_scalar.ll)
  set(scalar_obj ${generated_file_prefix}_scalar.o)

  set(triton_cpu_input ${generated_file_prefix}_triton_cpu.ttir)
  set(triton_cpu_mlir ${generated_file_prefix}_cpu.mlir)
  set(triton_cpu_llvm ${generated_file_prefix}_cpu.ll)
  set(triton_cpu_obj ${generated_file_prefix}_cpu.o)

  add_custom_command(
    OUTPUT ${ttir_file} ${launcher_header}
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/..:$ENV{PYTHONPATH}"
      "TRITON_PLUGIN_DIRS=${DYNO_TRITON_SHARED_SOURCE_DIR}"
      "TRITON_SHARED_OPT_PATH=${DYNO_TRITON_SHARED_OPT}"
      "LLVM_BINARY_DIR=${DYNO_TRITON_LLVM_BINARY_DIR}"
      ${DYNO_TRITON_PYTHON_EXECUTABLE}
      ${python_file}
      -o ${ttir_file}
      ${python_args}
    DEPENDS
      ${python_file}
      ${CMAKE_CURRENT_SOURCE_DIR}/../wrappergen.py
      ${DYNO_TRITON_SHARED_OPT}
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${mlir_file}
    COMMAND ${DYNO_TRITON_SHARED_OPT}
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
      ${DYNO_BENCHMARK_DUMP_IR_FLAG}
      ${ttir_file}
      -o ${mlir_file}
    DEPENDS ${ttir_file}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  dyno_lower_scalar_mlir_to_llvm_ir(
    INPUT ${mlir_file}
    OUTPUT_MLIR ${scalar_mlir}
    OUTPUT_LLVM ${scalar_llvm}
    KERNEL_NAME kernel_scalar
    BUFFERIZE
    POST_EXPAND_LOWER_AFFINE
    DEPENDS ${mlir_file}
  )
  dyno_compile_llvm_ir_to_object(
    INPUT ${scalar_llvm}
    OUTPUT ${scalar_obj}
    EXTRA_FLAGS -O2 -g ${CLANG_FLAGS_LIST}
  )

  dyno_lower_dyno_mlir_to_llvm_ir(
    INPUT ${mlir_file}
    OUTPUT_MLIR ${dyno_mlir}
    OUTPUT_LLVM ${dyno_llvm}
    KERNEL_NAME kernel_dyno
    VF 16
    UF 2
    BUFFERIZE
    FUSE_ELEMENTWISE
    FOLD_MEMREF_ALIAS_OPS
    PRE_VP_CANONICALIZE
    POST_VP_CANONICALIZE
    DEPENDS ${mlir_file}
  )
  dyno_compile_llvm_ir_to_object(
    INPUT ${dyno_llvm}
    OUTPUT ${dyno_obj}
    EXTRA_FLAGS -O2 -g ${CLANG_FLAGS_LIST}
  )

  dyno_rewrite_mlir_kernel(
    INPUT ${ttir_file}
    OUTPUT ${triton_cpu_input}
    KERNEL_NAME kernel_triton_cpu
    DEPENDS ${ttir_file}
  )

  add_custom_command(
    OUTPUT ${triton_cpu_mlir}
    COMMAND ${DYNO_TRITON_CPU_OPT}
      -triton-combine
      -canonicalize
      -triton-reorder-broadcast
      -cse
      -loop-invariant-code-motion
      -symbol-dce
      -triton-cpu-scalarize=skip-gather-scatter=true
      -triton-cpu-convert-memory-ops=use-gather-scatter=true
      -triton-cpu-convert-ptr-ops
      -triton-cpu-convert-elementwise-ops
      -triton-cpu-convert-elem-manip-ops
      -triton-cpu-convert-dot-op
      -triton-cpu-convert-histogram-op
      "-triton-cpu-convert-reduction=use-multidim-reduction-op=false\\;use-reduction-op=true"
      -triton-cpu-convert-scan
      -triton-cpu-convert-control-flow-op
      -triton-cpu-convert-atomic-ops
      -triton-cpu-convert-debug-ops
      -cse
      -symbol-dce
      -canonicalize
      -triton-cpu-canonicalize
      -triton-cpu-optimize-masks
      -canonicalize
      -triton-cpu-convert-dot-generic
      "-triton-cpu-add-casts-for-unsupported-ops=promote-bf16-to-fp32=false\\;convert-mixed-precision-matmul=true\\;promote-lib-math-to-fp32=true"
      "-triton-cpu-decompose-fp-conversions=decompose-bf16-conversions=false\\;decompose-fp8-conversions=true"
      -cse
      -symbol-dce
      -canonicalize
      -triton-cpu-lower-multi-reduction
      -expand-strided-metadata
      "-convert-vector-to-scf=full-unroll=true\\;target-rank=1\\;lower-tensors=false"
      -lower-affine
      -convert-scf-to-cf
      -convert-index-to-llvm
      -triton-cpu-func-op-to-llvm
      -triton-cpu-get-program-id-op-to-llvm
      -triton-cpu-memory-op-to-llvm
      -triton-cpu-atomic-ops-to-llvm
      -triton-cpu-debug-ops-to-llvm
      -triton-cpu-math-to-vec-lib=lib=sleef
      -convert-math-to-llvm
      -convert-math-to-libm
      -convert-vector-to-llvm
      -finalize-memref-to-llvm
      -convert-arith-to-llvm
      -convert-func-to-llvm
      -convert-ub-to-llvm
      -canonicalize
      -cse
      -symbol-dce
      ${triton_cpu_input}
      -o ${triton_cpu_mlir}
    DEPENDS ${triton_cpu_input} ${DYNO_TRITON_CPU_OPT}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${triton_cpu_llvm}
    COMMAND ${DYNO_TRITON_CPU_LLVM_BINARY_DIR}/mlir-translate
      ${triton_cpu_mlir}
      -mlir-to-llvmir
      -o ${triton_cpu_llvm}
    DEPENDS ${triton_cpu_mlir} ${DYNO_TRITON_CPU_LLVM_BINARY_DIR}/mlir-translate
    VERBATIM
  )

  dyno_compile_llvm_ir_to_object(
    INPUT ${triton_cpu_llvm}
    OUTPUT ${triton_cpu_obj}
    EXTRA_FLAGS -O2 -g ${CLANG_FLAGS_LIST}
    DEPENDS
      ${DYNO_TRITON_CPU_OPT}
      ${DYNO_TRITON_CPU_LLVM_BINARY_DIR}/mlir-translate
  )

  add_library(triton_${benchmark_name}_kernels STATIC
              ${scalar_obj} ${dyno_obj} ${triton_cpu_obj})
  set_target_properties(triton_${benchmark_name}_kernels
                        PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

add_subdirectory(matmul)
add_subdirectory(attention)
add_subdirectory(saxpy)
add_subdirectory(vector_add)
add_subdirectory(vector_mul)
