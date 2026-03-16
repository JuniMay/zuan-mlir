include_guard(GLOBAL)

include(CMakeParseArguments)
include(DynoKernelLowering)

#-------------------------------------------------------------------------------
# Runtime regression helpers
#-------------------------------------------------------------------------------

# Shared implementation for all runtime regression registrations.
#
# Required arguments:
# - `NAME`: executable and CTest name
# - `CATEGORY`: test label family
# - `MLIR_FILE`: kernel source file
# - `DRIVER`: C++ driver source
# - `DYNO_VF` / `DYNO_UF`: stripmining parameters
#
# Optional arguments:
# - `INPUT_IS_DYNO`: treat `MLIR_FILE` as direct Dyno instead of Linalg
# - `DYNO_REDUCTION_MODE` / `DYNO_FP_POLICY`: forwarded into the lowering helper
# - `EPSILON`: compile-time tolerance exposed to the driver
#
# Side effects:
# - builds the lowered kernel object and driver executable
# - registers the executable as a labeled CTest entry
# - appends the executable target to `DYNO_RUNTIME_REGRESSION_TARGETS`
function(_dyno_add_runtime_regression)
  set(options INPUT_IS_DYNO)
  set(oneValueArgs
      NAME
      CATEGORY
      MLIR_FILE
      DRIVER
      DYNO_VF
      DYNO_UF
      DYNO_REDUCTION_MODE
      DYNO_FP_POLICY
      EPSILON)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "" ${ARGN})

  foreach(_required IN ITEMS NAME CATEGORY MLIR_FILE DRIVER DYNO_VF DYNO_UF)
    if (NOT ARG_${_required})
      message(FATAL_ERROR
        "_dyno_add_runtime_regression missing required argument `${_required}`")
    endif()
  endforeach()

  if (NOT ARG_EPSILON)
    set(ARG_EPSILON 0.0)
  endif()

  set(_dyno_prefix "${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}")
  set(_dyno_mlir "${_dyno_prefix}.mlir")
  set(_dyno_llvm "${_dyno_prefix}.ll")
  set(_dyno_obj "${_dyno_prefix}.o")

  set(_dyno_lower_args
    INPUT ${ARG_MLIR_FILE}
    OUTPUT_MLIR ${_dyno_mlir}
    OUTPUT_LLVM ${_dyno_llvm}
    KERNEL_NAME kernel_dyno
    VF ${ARG_DYNO_VF}
    UF ${ARG_DYNO_UF}
    REQUEST_C_WRAPPERS
    PRE_VP_CANONICALIZE
    POST_VP_CANONICALIZE
    FOLD_MEMREF_ALIAS_OPS
    STRENGTH_REDUCTION
  )
  if (ARG_INPUT_IS_DYNO)
    list(APPEND _dyno_lower_args INPUT_IS_DYNO)
  endif()
  if (ARG_DYNO_REDUCTION_MODE)
    list(APPEND _dyno_lower_args REDUCTION_MODE ${ARG_DYNO_REDUCTION_MODE})
  endif()
  if (ARG_DYNO_FP_POLICY)
    list(APPEND _dyno_lower_args FP_POLICY ${ARG_DYNO_FP_POLICY})
  endif()

  dyno_lower_dyno_mlir_to_llvm_ir(${_dyno_lower_args})
  dyno_compile_llvm_ir_to_object(
    INPUT ${_dyno_llvm}
    OUTPUT ${_dyno_obj}
    EXTRA_FLAGS -O2 -g
  )

  add_library(${ARG_NAME}_dyno STATIC ${_dyno_obj})
  set_target_properties(${ARG_NAME}_dyno PROPERTIES LINKER_LANGUAGE CXX)

  add_executable(${ARG_NAME} ${ARG_DRIVER})
  target_include_directories(${ARG_NAME} PRIVATE
    ${DYNO_SOURCE_DIR}/test/runtime/common
  )
  target_compile_definitions(${ARG_NAME} PRIVATE
    DYNO_REGRESSION_NAME="${ARG_NAME}"
    DYNO_REGRESSION_DYNO_VF=${ARG_DYNO_VF}
    DYNO_REGRESSION_DYNO_UF=${ARG_DYNO_UF}
    DYNO_REGRESSION_EPSILON=${ARG_EPSILON}
    DYNO_REGRESSION_REDUCTION_MODE="${ARG_DYNO_REDUCTION_MODE}"
    DYNO_REGRESSION_FP_POLICY="${ARG_DYNO_FP_POLICY}"
  )
  target_link_libraries(${ARG_NAME} PRIVATE ${ARG_NAME}_dyno m)

  dyno_wrap_target_command(_dyno_test_cmd "$<TARGET_FILE:${ARG_NAME}>")
  add_test(NAME ${ARG_NAME} COMMAND ${_dyno_test_cmd})
  set_tests_properties(${ARG_NAME} PROPERTIES
    LABELS "runtime-regression;${ARG_CATEGORY}"
    TIMEOUT 60
  )

  set_property(GLOBAL APPEND PROPERTY DYNO_RUNTIME_REGRESSION_TARGETS
               ${ARG_NAME})
endfunction()

#-------------------------------------------------------------------------------
# Public regression wrappers
#-------------------------------------------------------------------------------

# Register a runtime regression that starts from a Linalg source file.
function(add_linalg_runtime_regression)
  _dyno_add_runtime_regression(CATEGORY linalg-runtime-regression ${ARGN})
endfunction()

# Register a runtime regression that starts from a direct Dyno source file.
function(add_dyno_runtime_regression)
  _dyno_add_runtime_regression(
    CATEGORY dyno-runtime-regression
    INPUT_IS_DYNO
    ${ARGN}
  )
endfunction()
