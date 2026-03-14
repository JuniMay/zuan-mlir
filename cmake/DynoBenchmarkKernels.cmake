include_guard(GLOBAL)

include(CMakeParseArguments)

if (NOT DEFINED DYNO_HOST_MLIR_OPT OR NOT DEFINED DYNO_HOST_DYNO_OPT_CMD)
  message(FATAL_ERROR
    "Include DynoHostTools.cmake and call dyno_configure_host_tools() before "
    "loading DynoBenchmarkKernels.cmake.")
endif()

function(dyno_rewrite_mlir_kernel)
  set(options)
  set(oneValueArgs INPUT OUTPUT KERNEL_NAME)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  add_custom_command(
    OUTPUT ${ARG_OUTPUT}
    COMMAND ${CMAKE_COMMAND}
      -DINPUT=${ARG_INPUT}
      -DOUTPUT=${ARG_OUTPUT}
      "-DOLD=kernel("
      "-DNEW=${ARG_KERNEL_NAME}("
      -P ${DYNO_SOURCE_DIR}/cmake/DynoRewriteText.cmake
    DEPENDS ${ARG_INPUT} ${ARG_DEPENDS}
            ${DYNO_SOURCE_DIR}/cmake/DynoRewriteText.cmake
    VERBATIM
  )
endfunction()

function(dyno_compile_llvm_ir_to_object)
  set(options)
  set(oneValueArgs INPUT OUTPUT)
  set(multiValueArgs DEPENDS EXTRA_FLAGS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  add_custom_command(
    OUTPUT ${ARG_OUTPUT}
    COMMAND ${DYNO_HOST_CLANG}
      -march=rv64gcv_zvfh
      ${ARG_EXTRA_FLAGS}
      -c
      -save-temps
      ${ARG_INPUT}
      -o ${ARG_OUTPUT}
    DEPENDS ${ARG_INPUT} ${ARG_DEPENDS} ${DYNO_HOST_CLANG}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
endfunction()

function(dyno_lower_scalar_mlir_to_llvm_ir)
  set(options BUFFERIZE POST_EXPAND_LOWER_AFFINE REQUEST_C_WRAPPERS)
  set(oneValueArgs INPUT OUTPUT_MLIR OUTPUT_LLVM KERNEL_NAME)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  set(_dyno_scalar_input "${ARG_INPUT}")
  set(_dyno_scalar_deps ${ARG_DEPENDS})
  if (ARG_KERNEL_NAME)
    set(_dyno_scalar_input "${ARG_OUTPUT_MLIR}.entry.mlir")
    dyno_rewrite_mlir_kernel(
      INPUT ${ARG_INPUT}
      OUTPUT ${_dyno_scalar_input}
      KERNEL_NAME ${ARG_KERNEL_NAME}
      DEPENDS ${ARG_DEPENDS}
    )
    set(_dyno_scalar_deps ${_dyno_scalar_input})
  endif()

  set(_dyno_scalar_prefix_args)
  if (ARG_BUFFERIZE)
    list(APPEND _dyno_scalar_prefix_args
      -empty-tensor-to-alloc-tensor
      "-one-shot-bufferize=allow-return-allocs-from-loops=true")
  endif()

  set(_dyno_scalar_suffix_args)
  if (ARG_POST_EXPAND_LOWER_AFFINE)
    list(APPEND _dyno_scalar_suffix_args -lower-affine)
  endif()
  if (ARG_REQUEST_C_WRAPPERS)
    list(APPEND _dyno_scalar_suffix_args -llvm-request-c-wrappers)
  endif()

  add_custom_command(
    OUTPUT ${ARG_OUTPUT_MLIR}
    COMMAND ${DYNO_HOST_MLIR_OPT}
      ${_dyno_scalar_prefix_args}
      -linalg-generalize-named-ops
      -convert-linalg-to-affine-loops
      -canonicalize
      -cse
      -lower-affine
      -convert-scf-to-cf
      -convert-cf-to-llvm
      -convert-ub-to-llvm
      -expand-strided-metadata
      ${_dyno_scalar_suffix_args}
      -convert-math-to-llvm
      -convert-math-to-libm
      -convert-vector-to-llvm
      -convert-arith-to-llvm
      -finalize-memref-to-llvm
      -convert-func-to-llvm
      -reconcile-unrealized-casts
      -canonicalize
      -cse
      ${DYNO_BENCHMARK_DUMP_IR_FLAG}
      ${_dyno_scalar_input}
      -o ${ARG_OUTPUT_MLIR}
    DEPENDS ${_dyno_scalar_deps} ${DYNO_HOST_MLIR_OPT}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${ARG_OUTPUT_LLVM}
    COMMAND ${DYNO_HOST_MLIR_TRANSLATE}
      ${ARG_OUTPUT_MLIR}
      -mlir-to-llvmir
      -o ${ARG_OUTPUT_LLVM}
    DEPENDS ${ARG_OUTPUT_MLIR} ${DYNO_HOST_MLIR_TRANSLATE}
    VERBATIM
  )
endfunction()

function(dyno_lower_dyno_mlir_to_llvm_ir)
  set(options
      BUFFERIZE
      FUSE_ELEMENTWISE
      FOLD_MEMREF_ALIAS_OPS
      INPUT_IS_DYNO
      POST_VP_CANONICALIZE
      PRE_VP_CANONICALIZE
      REQUEST_C_WRAPPERS
      STRENGTH_REDUCTION
      ENABLE_MATH_ESTIMATION)
  set(oneValueArgs INPUT OUTPUT_MLIR OUTPUT_LLVM KERNEL_NAME VF UF)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  set(_dyno_input "${ARG_INPUT}")
  set(_dyno_deps ${ARG_DEPENDS})
  if (ARG_KERNEL_NAME)
    set(_dyno_input "${ARG_OUTPUT_MLIR}.entry.mlir")
    dyno_rewrite_mlir_kernel(
      INPUT ${ARG_INPUT}
      OUTPUT ${_dyno_input}
      KERNEL_NAME ${ARG_KERNEL_NAME}
      DEPENDS ${ARG_DEPENDS}
    )
    set(_dyno_deps ${_dyno_input})
  endif()

  set(_dyno_prefix_args)
  if (ARG_FUSE_ELEMENTWISE)
    list(APPEND _dyno_prefix_args -linalg-fuse-elementwise-ops)
  endif()
  if (ARG_BUFFERIZE)
    list(APPEND _dyno_prefix_args
      -empty-tensor-to-alloc-tensor
      "-one-shot-bufferize=allow-return-allocs-from-loops=true")
  endif()
  if (NOT ARG_INPUT_IS_DYNO)
    list(APPEND _dyno_prefix_args -convert-linalg-to-dyno)
  endif()

  set(_dyno_middle_args
    "-lower-dyno=target-rank=2"
    "-dyno-stripmining=vf=${ARG_VF}\\;uf=${ARG_UF}\\;scalable=true")
  if (ARG_STRENGTH_REDUCTION)
    list(APPEND _dyno_middle_args -dyno-strength-reduction)
  endif()
  if (ARG_PRE_VP_CANONICALIZE)
    list(APPEND _dyno_middle_args -canonicalize -cse)
  endif()
  list(APPEND _dyno_middle_args "-convert-dyno-to-vp=vf=${ARG_VF}\\;scalable=true")
  if (ARG_FOLD_MEMREF_ALIAS_OPS)
    list(APPEND _dyno_middle_args -fold-memref-alias-ops)
  endif()

  if (ARG_ENABLE_MATH_ESTIMATION)
    set(_dyno_vp_to_llvm "-convert-vp-to-llvm=enable-math-estimation=true")
  else()
    set(_dyno_vp_to_llvm "-convert-vp-to-llvm=enable-math-estimation=false")
  endif()

  set(_dyno_suffix_args)
  if (ARG_POST_VP_CANONICALIZE)
    list(APPEND _dyno_suffix_args -canonicalize -cse)
  endif()
  list(APPEND _dyno_suffix_args
    -expand-strided-metadata
    -lower-affine
    -convert-scf-to-cf
    -convert-cf-to-llvm
    -convert-vector-to-llvm
    -convert-arith-to-llvm
    -convert-math-to-llvm
    -convert-math-to-libm
    -convert-vector-to-llvm)
  if (ARG_REQUEST_C_WRAPPERS)
    list(APPEND _dyno_suffix_args -llvm-request-c-wrappers)
  endif()
  list(APPEND _dyno_suffix_args
    -finalize-memref-to-llvm
    -convert-func-to-llvm
    -reconcile-unrealized-casts
    -canonicalize
    -cse)

  add_custom_command(
    OUTPUT ${ARG_OUTPUT_MLIR}
    COMMAND ${DYNO_HOST_DYNO_OPT_CMD}
      ${_dyno_prefix_args}
      ${_dyno_middle_args}
      ${_dyno_vp_to_llvm}
      ${_dyno_suffix_args}
      ${DYNO_BENCHMARK_DUMP_IR_FLAG}
      ${_dyno_input}
      -o ${ARG_OUTPUT_MLIR}
    DEPENDS ${_dyno_deps} dyno-opt
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${ARG_OUTPUT_LLVM}
    COMMAND ${DYNO_HOST_DYNO_TRANSLATE_CMD}
      ${ARG_OUTPUT_MLIR}
      -dyno-to-llvmir
      -o ${ARG_OUTPUT_LLVM}
    DEPENDS ${ARG_OUTPUT_MLIR} dyno-translate
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
endfunction()

function(dyno_lower_autovec_mlir_to_llvm_ir)
  set(options)
  set(oneValueArgs INPUT OUTPUT_MLIR OUTPUT_LLVM KERNEL_NAME VIRTUAL_VECTOR_SIZE)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  set(_dyno_autovec_input "${ARG_INPUT}")
  set(_dyno_autovec_deps ${ARG_DEPENDS})
  if (ARG_KERNEL_NAME)
    set(_dyno_autovec_input "${ARG_OUTPUT_MLIR}.entry.mlir")
    dyno_rewrite_mlir_kernel(
      INPUT ${ARG_INPUT}
      OUTPUT ${_dyno_autovec_input}
      KERNEL_NAME ${ARG_KERNEL_NAME}
      DEPENDS ${ARG_DEPENDS}
    )
    set(_dyno_autovec_deps ${_dyno_autovec_input})
  endif()

  add_custom_command(
    OUTPUT ${ARG_OUTPUT_MLIR}
    COMMAND ${DYNO_HOST_MLIR_OPT}
      -convert-linalg-to-affine-loops
      "-affine-super-vectorize=virtual-vector-size=${ARG_VIRTUAL_VECTOR_SIZE}\\;vectorize-reductions"
      "-convert-vector-to-scf=full-unroll\\;target-rank=1"
      "-convert-vector-to-llvm=reassociate-fp-reductions"
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
      ${DYNO_BENCHMARK_DUMP_IR_FLAG}
      ${_dyno_autovec_input}
      -o ${ARG_OUTPUT_MLIR}
    DEPENDS ${_dyno_autovec_deps} ${DYNO_HOST_MLIR_OPT}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${ARG_OUTPUT_LLVM}
    COMMAND ${DYNO_HOST_MLIR_TRANSLATE}
      ${ARG_OUTPUT_MLIR}
      -mlir-to-llvmir
      -o ${ARG_OUTPUT_LLVM}
    DEPENDS ${ARG_OUTPUT_MLIR} ${DYNO_HOST_MLIR_TRANSLATE}
    VERBATIM
  )
endfunction()

function(dyno_lower_transform_mlir_to_llvm_ir)
  set(options)
  set(oneValueArgs
      INPUT
      OUTPUT_MLIR
      OUTPUT_LLVM
      KERNEL_NAME
      TRANSFORM_FILE
      VF
      UF)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  set(_dyno_transform_generated "${ARG_OUTPUT_MLIR}.transform.mlir")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               "${ARG_TRANSFORM_FILE}")
  file(READ "${ARG_TRANSFORM_FILE}" _dyno_transform_contents)
  string(REPLACE "VF" "${ARG_VF}" _dyno_transform_contents
                 "${_dyno_transform_contents}")
  string(REPLACE "UF" "${ARG_UF}" _dyno_transform_contents
                 "${_dyno_transform_contents}")
  file(WRITE "${_dyno_transform_generated}" "${_dyno_transform_contents}")

  set(_dyno_transform_input "${ARG_INPUT}")
  set(_dyno_transform_deps ${ARG_DEPENDS} ${_dyno_transform_generated})
  if (ARG_KERNEL_NAME)
    set(_dyno_transform_input "${ARG_OUTPUT_MLIR}.entry.mlir")
    dyno_rewrite_mlir_kernel(
      INPUT ${ARG_INPUT}
      OUTPUT ${_dyno_transform_input}
      KERNEL_NAME ${ARG_KERNEL_NAME}
      DEPENDS ${ARG_DEPENDS}
    )
    set(_dyno_transform_deps ${_dyno_transform_input} ${_dyno_transform_generated})
  endif()

  add_custom_command(
    OUTPUT ${ARG_OUTPUT_MLIR}
    COMMAND ${DYNO_HOST_MLIR_OPT}
      "-transform-preload-library=transform-library-paths=${_dyno_transform_generated}"
      -transform-interpreter
      -canonicalize
      -lower-vector-mask
      "-convert-vector-to-scf=full-unroll\\;target-rank=1\\;lower-scalable"
      -convert-scf-to-cf
      -convert-cf-to-llvm
      -expand-strided-metadata
      -convert-math-to-llvm
      -convert-math-to-libm
      "-convert-vector-to-llvm=reassociate-fp-reductions"
      -convert-ub-to-llvm
      -llvm-request-c-wrappers
      -finalize-memref-to-llvm
      -lower-affine
      -convert-arith-to-llvm
      -convert-func-to-llvm
      -reconcile-unrealized-casts
      ${DYNO_BENCHMARK_DUMP_IR_FLAG}
      ${_dyno_transform_input}
      -o ${ARG_OUTPUT_MLIR}
    DEPENDS ${_dyno_transform_deps} ${DYNO_HOST_MLIR_OPT}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${ARG_OUTPUT_LLVM}
    COMMAND ${DYNO_HOST_MLIR_TRANSLATE}
      ${ARG_OUTPUT_MLIR}
      -mlir-to-llvmir
      -o ${ARG_OUTPUT_LLVM}
    DEPENDS ${ARG_OUTPUT_MLIR} ${DYNO_HOST_MLIR_TRANSLATE}
    VERBATIM
  )
endfunction()

function(add_linalg_dyno_kernel benchmark_name vf uf)
  if (${ARGC} GREATER 3 AND ARGV3 STREQUAL "TRUE")
    set(_dyno_estimate_suffix "_est")
    set(_dyno_estimate_math ENABLE_MATH_ESTIMATION)
  else()
    set(_dyno_estimate_suffix "")
    unset(_dyno_estimate_math)
  endif()

  set(_dyno_mlir_file ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark_name}.mlir)
  set(_dyno_prefix
      ${CMAKE_CURRENT_BINARY_DIR}/${benchmark_name}_dyno_${vf}_${uf}${_dyno_estimate_suffix})
  set(_dyno_mlir ${_dyno_prefix}.mlir)
  set(_dyno_llvm ${_dyno_prefix}.ll)
  set(_dyno_obj ${_dyno_prefix}.o)

  dyno_lower_dyno_mlir_to_llvm_ir(
    INPUT ${_dyno_mlir_file}
    OUTPUT_MLIR ${_dyno_mlir}
    OUTPUT_LLVM ${_dyno_llvm}
    KERNEL_NAME kernel_dyno_${vf}_${uf}${_dyno_estimate_suffix}
    VF ${vf}
    UF ${uf}
    FOLD_MEMREF_ALIAS_OPS
    PRE_VP_CANONICALIZE
    POST_VP_CANONICALIZE
    REQUEST_C_WRAPPERS
    STRENGTH_REDUCTION
    ${_dyno_estimate_math}
  )
  dyno_compile_llvm_ir_to_object(
    INPUT ${_dyno_llvm}
    OUTPUT ${_dyno_obj}
    EXTRA_FLAGS -O3 -ffast-math -g ${CLANG_FLAGS_LIST}
  )

  add_library(linalg_${benchmark_name}_dyno_${vf}_${uf}${_dyno_estimate_suffix}
              STATIC ${_dyno_obj})
  set_target_properties(
    linalg_${benchmark_name}_dyno_${vf}_${uf}${_dyno_estimate_suffix}
    PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

function(add_linalg_scalar_kernel benchmark_name)
  set(_dyno_mlir_file ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark_name}.mlir)
  set(_dyno_prefix ${CMAKE_CURRENT_BINARY_DIR}/${benchmark_name}_scalar)
  set(_dyno_mlir ${_dyno_prefix}.mlir)
  set(_dyno_llvm ${_dyno_prefix}.ll)
  set(_dyno_obj ${_dyno_prefix}.o)

  dyno_lower_scalar_mlir_to_llvm_ir(
    INPUT ${_dyno_mlir_file}
    OUTPUT_MLIR ${_dyno_mlir}
    OUTPUT_LLVM ${_dyno_llvm}
    KERNEL_NAME kernel_scalar
    REQUEST_C_WRAPPERS
  )
  dyno_compile_llvm_ir_to_object(
    INPUT ${_dyno_llvm}
    OUTPUT ${_dyno_obj}
    EXTRA_FLAGS -O3 -ffast-math ${CLANG_FLAGS_LIST}
  )

  add_library(linalg_${benchmark_name}_scalar STATIC ${_dyno_obj})
  set_target_properties(linalg_${benchmark_name}_scalar
                        PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

function(add_linalg_autovec_kernel benchmark_name virtvecsize)
  set(_dyno_mlir_file ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark_name}.mlir)
  set(_dyno_prefix
      ${CMAKE_CURRENT_BINARY_DIR}/${benchmark_name}_autovec_${virtvecsize})
  set(_dyno_mlir ${_dyno_prefix}.mlir)
  set(_dyno_llvm ${_dyno_prefix}.ll)
  set(_dyno_obj ${_dyno_prefix}.o)

  dyno_lower_autovec_mlir_to_llvm_ir(
    INPUT ${_dyno_mlir_file}
    OUTPUT_MLIR ${_dyno_mlir}
    OUTPUT_LLVM ${_dyno_llvm}
    KERNEL_NAME kernel_autovec_${virtvecsize}
    VIRTUAL_VECTOR_SIZE ${virtvecsize}
  )
  dyno_compile_llvm_ir_to_object(
    INPUT ${_dyno_llvm}
    OUTPUT ${_dyno_obj}
    EXTRA_FLAGS
      -O3
      -ffast-math
      -mllvm
      --riscv-v-vector-bits-min=256
      ${CLANG_FLAGS_LIST}
  )

  add_library(linalg_${benchmark_name}_autovec_${virtvecsize} STATIC
              ${_dyno_obj})
  set_target_properties(linalg_${benchmark_name}_autovec_${virtvecsize}
                        PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

function(add_linalg_transform_kernel benchmark_name vf uf)
  set(_dyno_mlir_file ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark_name}.mlir)
  set(_dyno_prefix
      ${CMAKE_CURRENT_BINARY_DIR}/${benchmark_name}_transform_${vf}_${uf})
  set(_dyno_mlir ${_dyno_prefix}.mlir)
  set(_dyno_llvm ${_dyno_prefix}.ll)
  set(_dyno_obj ${_dyno_prefix}.o)

  dyno_lower_transform_mlir_to_llvm_ir(
    INPUT ${_dyno_mlir_file}
    OUTPUT_MLIR ${_dyno_mlir}
    OUTPUT_LLVM ${_dyno_llvm}
    KERNEL_NAME kernel_transform_${vf}_${uf}
    TRANSFORM_FILE ${CMAKE_CURRENT_SOURCE_DIR}/transform.txt
    VF ${vf}
    UF ${uf}
  )
  dyno_compile_llvm_ir_to_object(
    INPUT ${_dyno_llvm}
    OUTPUT ${_dyno_obj}
    EXTRA_FLAGS
      -O3
      -ffast-math
      -mllvm
      --riscv-v-vector-bits-min=256
      ${CLANG_FLAGS_LIST}
  )

  add_library(linalg_${benchmark_name}_transform_${vf}_${uf} STATIC
              ${_dyno_obj})
  set_target_properties(linalg_${benchmark_name}_transform_${vf}_${uf}
                        PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

function(dyno_add_direct_dyno_kernel)
  set(options)
  set(oneValueArgs TARGET INPUT VF UF)
  set(multiValueArgs EXTRA_FLAGS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  set(_dyno_prefix ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET})
  set(_dyno_mlir ${_dyno_prefix}.mlir)
  set(_dyno_llvm ${_dyno_prefix}.ll)
  set(_dyno_obj ${_dyno_prefix}.o)

  dyno_lower_dyno_mlir_to_llvm_ir(
    INPUT ${ARG_INPUT}
    OUTPUT_MLIR ${_dyno_mlir}
    OUTPUT_LLVM ${_dyno_llvm}
    INPUT_IS_DYNO
    VF ${ARG_VF}
    UF ${ARG_UF}
    REQUEST_C_WRAPPERS
  )
  dyno_compile_llvm_ir_to_object(
    INPUT ${_dyno_llvm}
    OUTPUT ${_dyno_obj}
    EXTRA_FLAGS ${ARG_EXTRA_FLAGS}
  )

  add_library(${ARG_TARGET} STATIC ${_dyno_obj})
  set_target_properties(${ARG_TARGET} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()
