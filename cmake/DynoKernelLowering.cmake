include_guard(GLOBAL)

include(CMakeParseArguments)
include(DynoKernelBuild)

#-------------------------------------------------------------------------------
# Shared build preconditions
#-------------------------------------------------------------------------------

# Verify that the common MLIR/LLVM host tools needed by every lowering path are
# available in the configured kernel-build state.
function(_dyno_require_llvm_lowering_tools)
  _dyno_require_kernel_build_configured()
  foreach(_tool_var IN ITEMS DYNO_HOST_MLIR_OPT DYNO_HOST_MLIR_TRANSLATE
                             DYNO_HOST_CLANG)
    if (NOT DEFINED ${_tool_var} OR "${${_tool_var}}" STREQUAL "")
      message(FATAL_ERROR
        "Dyno kernel build tools are not configured. "
        "Call dyno_configure_kernel_build() before lowering kernels.")
    endif()
  endforeach()
endfunction()

# Verify that the Dyno-specific host tools are also available.
#
# This extends `_dyno_require_llvm_lowering_tools()` for lowering paths that
# invoke `dyno-opt` or `dyno-translate`.
function(_dyno_require_dyno_lowering_tools)
  _dyno_require_llvm_lowering_tools()
  foreach(_tool_var IN ITEMS DYNO_HOST_DYNO_OPT_CMD DYNO_HOST_DYNO_TRANSLATE_CMD)
    if (NOT DEFINED ${_tool_var} OR "${${_tool_var}}" STREQUAL "")
      message(FATAL_ERROR
        "Dyno host tools are not configured. "
        "Call dyno_configure_kernel_build() before lowering Dyno kernels.")
    endif()
  endforeach()
endfunction()

#-------------------------------------------------------------------------------
# Shared file-generation helpers
#-------------------------------------------------------------------------------

# Configure optional pass-by-pass IR dumping for one lowered MLIR artifact.
#
# Arguments:
# - `out_setup_var`: receives setup commands that clear and recreate the dump dir
# - `out_args_var`: receives the extra MLIR pass-manager dump flags
# - `output_mlir`: final lowered MLIR file whose sibling `.ir/` directory is used
#
# When `DYNO_BENCHMARK_DUMP_IR` is off, both outputs are empty.
function(dyno_configure_ir_dump out_setup_var out_args_var output_mlir)
  if (DYNO_BENCHMARK_DUMP_IR)
    set(_dyno_dump_dir "${output_mlir}.ir")
    set(_dyno_dump_setup
      COMMAND ${CMAKE_COMMAND} -E rm -rf ${_dyno_dump_dir}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${_dyno_dump_dir})
    set(_dyno_dump_args
      -mlir-print-ir-after-all
      "-mlir-print-ir-tree-dir=${_dyno_dump_dir}")
  else()
    unset(_dyno_dump_setup)
    unset(_dyno_dump_args)
  endif()

  set(${out_setup_var} ${_dyno_dump_setup} PARENT_SCOPE)
  set(${out_args_var} ${_dyno_dump_args} PARENT_SCOPE)
endfunction()

# Rewrite the textual kernel entry point in an MLIR file.
#
# Arguments:
# - `INPUT`: source MLIR file
# - `OUTPUT`: rewritten MLIR file
# - `KERNEL_NAME`: replacement symbol name for the `kernel(` entry point
# - `DEPENDS`: extra dependencies for the generated output
#
# This is a thin wrapper around `cmake/DynoRewriteText.cmake` and is used to
# manufacture stable wrapper symbol names such as `kernel_scalar`.
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

# Prepare the MLIR input path used by a lowering pipeline.
#
# Arguments:
# - `INPUT`: original MLIR source file
# - `OUTPUT_MLIR`: final lowered output path; used to derive the rewritten entry
#   filename when `KERNEL_NAME` is provided
# - `KERNEL_NAME`: optional replacement entry-point name
# - `DEPENDS`: extra file dependencies
#
# Outputs:
# - `out_input_var`: the actual MLIR file the lowering command should consume
# - `out_deps_var`: the corresponding dependency list
#
# If no kernel rename is requested, this is effectively a pass-through.
function(_dyno_prepare_mlir_kernel_input out_input_var out_deps_var)
  set(options)
  set(oneValueArgs INPUT OUTPUT_MLIR KERNEL_NAME)
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

  set(${out_input_var} "${_dyno_input}" PARENT_SCOPE)
  set(${out_deps_var} ${_dyno_deps} PARENT_SCOPE)
endfunction()

# Create the translation step from LLVM-dialect MLIR into textual LLVM IR.
#
# Arguments:
# - `INPUT`: LLVM-dialect MLIR file
# - `OUTPUT`: textual LLVM IR file to generate
# - `TRANSLATE_TOOL`: translation executable
# - `TRANSLATION_FLAG`: one translation mode flag such as `-mlir-to-llvmir`
# - `DEPENDS`: extra dependencies, typically the translator executable itself
function(_dyno_translate_mlir_to_llvm_ir)
  set(options)
  set(oneValueArgs INPUT OUTPUT TRANSLATE_TOOL TRANSLATION_FLAG)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  add_custom_command(
    OUTPUT ${ARG_OUTPUT}
    COMMAND ${ARG_TRANSLATE_TOOL}
      ${ARG_INPUT}
      ${ARG_TRANSLATION_FLAG}
      -o ${ARG_OUTPUT}
    DEPENDS ${ARG_INPUT} ${ARG_DEPENDS}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
endfunction()

#-------------------------------------------------------------------------------
# Shared lowering and compilation helpers
#-------------------------------------------------------------------------------

# Compile textual LLVM IR into a target object file using the configured host
# `clang`.
#
# Arguments:
# - `INPUT`: LLVM IR `.ll` file
# - `OUTPUT`: target object `.o`
# - `EXTRA_FLAGS`: optimization/debug flags specific to the caller
# - `DEPENDS`: extra dependencies for the custom command
#
# The helper always adds the shared target/sysroot flags plus the project's
# RISC-V vector ISA baseline.
function(dyno_compile_llvm_ir_to_object)
  set(options)
  set(oneValueArgs INPUT OUTPUT)
  set(multiValueArgs DEPENDS EXTRA_FLAGS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  _dyno_require_llvm_lowering_tools()
  dyno_get_kernel_compile_flags(_dyno_target_clang_flags)

  add_custom_command(
    OUTPUT ${ARG_OUTPUT}
    COMMAND ${DYNO_HOST_CLANG}
      -march=rv64gcv_zvfh
      ${_dyno_target_clang_flags}
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

# Lower a scalar Linalg kernel directly to LLVM IR without going through Dyno.
#
# Key options:
# - `BUFFERIZE`: insert the one-shot bufferization prefix
# - `POST_EXPAND_LOWER_AFFINE`: re-run `-lower-affine` after expansion
# - `REQUEST_C_WRAPPERS`: emit C ABI wrappers for launcher code
# - `KERNEL_NAME`: optional entry-point rewrite
#
# Outputs:
# - `OUTPUT_MLIR`: LLVM-dialect MLIR after the full lowering pipeline
# - `OUTPUT_LLVM`: textual LLVM IR translated from `OUTPUT_MLIR`
function(dyno_lower_scalar_mlir_to_llvm_ir)
  set(options BUFFERIZE POST_EXPAND_LOWER_AFFINE REQUEST_C_WRAPPERS)
  set(oneValueArgs INPUT OUTPUT_MLIR OUTPUT_LLVM KERNEL_NAME)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  _dyno_require_llvm_lowering_tools()

  _dyno_prepare_mlir_kernel_input(
    _dyno_scalar_input
    _dyno_scalar_deps
    INPUT ${ARG_INPUT}
    OUTPUT_MLIR ${ARG_OUTPUT_MLIR}
    KERNEL_NAME ${ARG_KERNEL_NAME}
    DEPENDS ${ARG_DEPENDS}
  )

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

  dyno_configure_ir_dump(_dyno_dump_setup _dyno_dump_args
                         "${ARG_OUTPUT_MLIR}")

  add_custom_command(
    OUTPUT ${ARG_OUTPUT_MLIR}
    ${_dyno_dump_setup}
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
      ${_dyno_dump_args}
      ${_dyno_scalar_input}
      -o ${ARG_OUTPUT_MLIR}
    DEPENDS ${_dyno_scalar_deps} ${DYNO_HOST_MLIR_OPT}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  _dyno_translate_mlir_to_llvm_ir(
    INPUT ${ARG_OUTPUT_MLIR}
    OUTPUT ${ARG_OUTPUT_LLVM}
    TRANSLATE_TOOL ${DYNO_HOST_MLIR_TRANSLATE}
    TRANSLATION_FLAG -mlir-to-llvmir
    DEPENDS ${DYNO_HOST_MLIR_TRANSLATE}
  )
endfunction()

# Lower a Dyno kernel, or a Linalg kernel routed through Dyno, to LLVM IR.
#
# Key options:
# - `INPUT_IS_DYNO`: skip the front-end `-convert-linalg-to-dyno` step
# - `VF` / `UF`: stripmining configuration
# - `REDUCTION_MODE` / `FP_POLICY`: forwarded into `-dyno-stripmining`
# - `BUFFERIZE`, `FUSE_ELEMENTWISE`, `STRENGTH_REDUCTION`,
#   `PRE_VP_CANONICALIZE`, `POST_VP_CANONICALIZE`,
#   `FOLD_MEMREF_ALIAS_OPS`, `REQUEST_C_WRAPPERS`,
#   `ENABLE_MATH_ESTIMATION`: toggle optional pipeline stages
#
# Outputs:
# - `OUTPUT_MLIR`: LLVM-dialect MLIR after Dyno/VP lowering
# - `OUTPUT_LLVM`: textual LLVM IR translated from `OUTPUT_MLIR`
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
  set(oneValueArgs INPUT OUTPUT_MLIR OUTPUT_LLVM KERNEL_NAME VF UF
                   REDUCTION_MODE FP_POLICY)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  _dyno_require_dyno_lowering_tools()

  _dyno_prepare_mlir_kernel_input(
    _dyno_input
    _dyno_deps
    INPUT ${ARG_INPUT}
    OUTPUT_MLIR ${ARG_OUTPUT_MLIR}
    KERNEL_NAME ${ARG_KERNEL_NAME}
    DEPENDS ${ARG_DEPENDS}
  )

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

  set(_dyno_stripmining_arg
    "-dyno-stripmining=vf=${ARG_VF} uf=${ARG_UF} scalable=true")
  if (ARG_REDUCTION_MODE)
    string(APPEND _dyno_stripmining_arg
           " reduction-mode=${ARG_REDUCTION_MODE}")
  endif()
  if (ARG_FP_POLICY)
    string(APPEND _dyno_stripmining_arg " fp-policy=${ARG_FP_POLICY}")
  endif()

  set(_dyno_middle_args
    "-lower-dyno=target-rank=2"
    "${_dyno_stripmining_arg}")
  if (ARG_STRENGTH_REDUCTION)
    list(APPEND _dyno_middle_args -dyno-strength-reduction)
  endif()
  if (ARG_PRE_VP_CANONICALIZE)
    list(APPEND _dyno_middle_args -canonicalize -cse)
  endif()
  list(APPEND _dyno_middle_args
       "-convert-dyno-to-vp=vf=${ARG_VF} scalable=true")
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

  dyno_configure_ir_dump(_dyno_dump_setup _dyno_dump_args
                         "${ARG_OUTPUT_MLIR}")

  add_custom_command(
    OUTPUT ${ARG_OUTPUT_MLIR}
    ${_dyno_dump_setup}
    COMMAND ${DYNO_HOST_DYNO_OPT_CMD}
      ${_dyno_prefix_args}
      ${_dyno_middle_args}
      ${_dyno_vp_to_llvm}
      ${_dyno_suffix_args}
      ${_dyno_dump_args}
      ${_dyno_input}
      -o ${ARG_OUTPUT_MLIR}
    DEPENDS ${_dyno_deps} dyno-opt
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  _dyno_translate_mlir_to_llvm_ir(
    INPUT ${ARG_OUTPUT_MLIR}
    OUTPUT ${ARG_OUTPUT_LLVM}
    TRANSLATE_TOOL ${DYNO_HOST_DYNO_TRANSLATE_CMD}
    TRANSLATION_FLAG -dyno-to-llvmir
    DEPENDS dyno-translate
  )
endfunction()

# Lower a Linalg kernel through the upstream affine super-vectorizer path.
#
# Key options:
# - `VIRTUAL_VECTOR_SIZE`: value forwarded to `-affine-super-vectorize`
# - `KERNEL_NAME`: optional entry-point rewrite
#
# This path provides the auto-vectorized comparison kernels used by the
# benchmark suite.
function(dyno_lower_autovec_mlir_to_llvm_ir)
  set(options)
  set(oneValueArgs INPUT OUTPUT_MLIR OUTPUT_LLVM KERNEL_NAME VIRTUAL_VECTOR_SIZE)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  _dyno_require_llvm_lowering_tools()

  _dyno_prepare_mlir_kernel_input(
    _dyno_autovec_input
    _dyno_autovec_deps
    INPUT ${ARG_INPUT}
    OUTPUT_MLIR ${ARG_OUTPUT_MLIR}
    KERNEL_NAME ${ARG_KERNEL_NAME}
    DEPENDS ${ARG_DEPENDS}
  )

  dyno_configure_ir_dump(_dyno_dump_setup _dyno_dump_args
                         "${ARG_OUTPUT_MLIR}")

  add_custom_command(
    OUTPUT ${ARG_OUTPUT_MLIR}
    ${_dyno_dump_setup}
    COMMAND ${DYNO_HOST_MLIR_OPT}
      -convert-linalg-to-affine-loops
      "-affine-super-vectorize=virtual-vector-size=${ARG_VIRTUAL_VECTOR_SIZE} vectorize-reductions"
      "-convert-vector-to-scf=full-unroll target-rank=1"
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
      ${_dyno_dump_args}
      ${_dyno_autovec_input}
      -o ${ARG_OUTPUT_MLIR}
    DEPENDS ${_dyno_autovec_deps} ${DYNO_HOST_MLIR_OPT}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  _dyno_translate_mlir_to_llvm_ir(
    INPUT ${ARG_OUTPUT_MLIR}
    OUTPUT ${ARG_OUTPUT_LLVM}
    TRANSLATE_TOOL ${DYNO_HOST_MLIR_TRANSLATE}
    TRANSLATION_FLAG -mlir-to-llvmir
    DEPENDS ${DYNO_HOST_MLIR_TRANSLATE}
  )
endfunction()

# Lower a Linalg kernel through the transform-dialect comparison path.
#
# Key options:
# - `TRANSFORM_FILE`: template transform script containing `VF` / `UF`
#   placeholders
# - `VF` / `UF`: values substituted into the transform template
# - `KERNEL_NAME`: optional entry-point rewrite
#
# The helper materializes a build-local transform module, runs the transform
# interpreter, and then finishes the standard LLVM lowering pipeline.
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

  _dyno_require_llvm_lowering_tools()

  set(_dyno_transform_generated "${ARG_OUTPUT_MLIR}.transform.mlir")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               "${ARG_TRANSFORM_FILE}")
  file(READ "${ARG_TRANSFORM_FILE}" _dyno_transform_contents)
  string(REPLACE "VF" "${ARG_VF}" _dyno_transform_contents
                 "${_dyno_transform_contents}")
  string(REPLACE "UF" "${ARG_UF}" _dyno_transform_contents
                 "${_dyno_transform_contents}")
  file(WRITE "${_dyno_transform_generated}" "${_dyno_transform_contents}")

  _dyno_prepare_mlir_kernel_input(
    _dyno_transform_input
    _dyno_transform_deps
    INPUT ${ARG_INPUT}
    OUTPUT_MLIR ${ARG_OUTPUT_MLIR}
    KERNEL_NAME ${ARG_KERNEL_NAME}
    DEPENDS ${ARG_DEPENDS}
  )
  list(APPEND _dyno_transform_deps ${_dyno_transform_generated})

  dyno_configure_ir_dump(_dyno_dump_setup _dyno_dump_args
                         "${ARG_OUTPUT_MLIR}")

  add_custom_command(
    OUTPUT ${ARG_OUTPUT_MLIR}
    ${_dyno_dump_setup}
    COMMAND ${DYNO_HOST_MLIR_OPT}
      "-transform-preload-library=transform-library-paths=${_dyno_transform_generated}"
      -transform-interpreter
      -canonicalize
      -lower-vector-mask
      "-convert-vector-to-scf=full-unroll target-rank=1 lower-scalable"
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
      ${_dyno_dump_args}
      ${_dyno_transform_input}
      -o ${ARG_OUTPUT_MLIR}
    DEPENDS ${_dyno_transform_deps} ${DYNO_HOST_MLIR_OPT}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  _dyno_translate_mlir_to_llvm_ir(
    INPUT ${ARG_OUTPUT_MLIR}
    OUTPUT ${ARG_OUTPUT_LLVM}
    TRANSLATE_TOOL ${DYNO_HOST_MLIR_TRANSLATE}
    TRANSLATION_FLAG -mlir-to-llvmir
    DEPENDS ${DYNO_HOST_MLIR_TRANSLATE}
  )
endfunction()
