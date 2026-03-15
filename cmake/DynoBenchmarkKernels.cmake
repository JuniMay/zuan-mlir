include_guard(GLOBAL)

include(CMakeParseArguments)
include(DynoKernelLowering)

#-------------------------------------------------------------------------------
# Internal benchmark target helpers
#-------------------------------------------------------------------------------

# Package a generated object file as a static library with a fixed C++ linker
# language. The benchmark launchers consume these libraries uniformly regardless
# of the lowering path that produced the object.
function(_dyno_add_kernel_object_library target_name object_file)
  add_library(${target_name} STATIC ${object_file})
  set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

#-------------------------------------------------------------------------------
# Linalg benchmark kernels
#-------------------------------------------------------------------------------

# Build one Dyno-backed benchmark kernel library from a Linalg source file.
#
# Arguments:
# - `benchmark_name`: base filename of the `.mlir` source in the current dir
# - `vf`, `uf`: stripmining configuration
# - optional fourth positional arg `TRUE`: enable math-estimation lowering and
#   append the `_est` suffix to the generated target name
#
# Output target:
# - `linalg_<benchmark_name>_dyno_<vf>_<uf>[_est]`
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
    EXTRA_FLAGS -O3 -ffast-math -g
  )

  _dyno_add_kernel_object_library(
    linalg_${benchmark_name}_dyno_${vf}_${uf}${_dyno_estimate_suffix}
    ${_dyno_obj}
  )
endfunction()

# Build the scalar baseline kernel library for a benchmark source.
#
# Output target:
# - `linalg_<benchmark_name>_scalar`
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
    EXTRA_FLAGS -O3 -ffast-math
  )

  _dyno_add_kernel_object_library(linalg_${benchmark_name}_scalar ${_dyno_obj})
endfunction()

# Build the auto-vectorized comparison kernel library for a benchmark source.
#
# Arguments:
# - `benchmark_name`: base filename of the `.mlir` source
# - `virtvecsize`: virtual vector size passed into the affine super-vectorizer
#
# Output target:
# - `linalg_<benchmark_name>_autovec_<virtvecsize>`
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
  )

  _dyno_add_kernel_object_library(
    linalg_${benchmark_name}_autovec_${virtvecsize}
    ${_dyno_obj}
  )
endfunction()

# Build the transform-dialect comparison kernel library for a benchmark source.
#
# Arguments:
# - `benchmark_name`: base filename of the `.mlir` source
# - `vf`, `uf`: values substituted into the local `transform.txt` template
#
# Output target:
# - `linalg_<benchmark_name>_transform_<vf>_<uf>`
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
  )

  _dyno_add_kernel_object_library(
    linalg_${benchmark_name}_transform_${vf}_${uf}
    ${_dyno_obj}
  )
endfunction()

#-------------------------------------------------------------------------------
# Direct Dyno benchmark kernels
#-------------------------------------------------------------------------------

# Build a benchmark kernel library from a direct Dyno source file.
#
# Arguments:
# - `TARGET`: final static library target name
# - `INPUT`: Dyno MLIR source file
# - `VF` / `UF`: stripmining configuration
# - `EXTRA_FLAGS`: extra clang flags for object compilation
#
# This is used by benchmark families that do not start from Linalg.
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

  _dyno_add_kernel_object_library(${ARG_TARGET} ${_dyno_obj})
endfunction()
