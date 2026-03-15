include_guard(GLOBAL)

#-------------------------------------------------------------------------------
# Default path configuration
#-------------------------------------------------------------------------------

if (NOT DEFINED DYNO_TRITON_UPSTREAM_SOURCE_DIR)
  set(DYNO_TRITON_UPSTREAM_SOURCE_DIR "${DYNO_SOURCE_DIR}/third_party/triton")
endif()

if (NOT DEFINED DYNO_TRITON_CPU_SOURCE_DIR)
  set(DYNO_TRITON_CPU_SOURCE_DIR "${DYNO_SOURCE_DIR}/third_party/triton-cpu")
endif()

if (NOT DEFINED DYNO_TRITON_SHARED_SOURCE_DIR)
  set(DYNO_TRITON_SHARED_SOURCE_DIR "${DYNO_SOURCE_DIR}/third_party/triton_shared")
endif()

if (NOT DEFINED DYNO_TRITON_ROOT_DIR)
  set(DYNO_TRITON_ROOT_DIR "${DYNO_BINARY_DIR}/third_party/triton")
endif()

set(DYNO_TRITON_SHARED_ROOT_DIR "${DYNO_TRITON_ROOT_DIR}/shared")

set(DYNO_TRITON_SHARED_BUILD_DIR "${DYNO_TRITON_SHARED_ROOT_DIR}/build")

set(DYNO_TRITON_VENV_DIR "${DYNO_TRITON_SHARED_ROOT_DIR}/venv")

if (NOT DEFINED DYNO_TRITON_SHARED_OPT OR DYNO_TRITON_SHARED_OPT STREQUAL "")
  set(DYNO_TRITON_SHARED_OPT
    "${DYNO_TRITON_SHARED_BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt")
endif()

if (NOT DEFINED DYNO_TRITON_LLVM_BINARY_DIR OR DYNO_TRITON_LLVM_BINARY_DIR STREQUAL "")
  set(DYNO_TRITON_LLVM_BINARY_DIR "${DYNO_TRITON_SHARED_ROOT_DIR}/llvm/bin")
endif()

set(DYNO_TRITON_CPU_ROOT_DIR "${DYNO_TRITON_ROOT_DIR}/cpu")

set(DYNO_TRITON_CPU_BUILD_DIR "${DYNO_TRITON_CPU_ROOT_DIR}/build")

if (NOT DEFINED DYNO_TRITON_CPU_OPT OR DYNO_TRITON_CPU_OPT STREQUAL "")
  set(DYNO_TRITON_CPU_OPT "${DYNO_TRITON_CPU_BUILD_DIR}/bin/triton-opt")
endif()

if (NOT DEFINED DYNO_TRITON_CPU_LLVM_BINARY_DIR OR DYNO_TRITON_CPU_LLVM_BINARY_DIR STREQUAL "")
  set(DYNO_TRITON_CPU_LLVM_BINARY_DIR "${DYNO_TRITON_CPU_ROOT_DIR}/llvm/bin")
endif()

#-------------------------------------------------------------------------------
# Default tool discovery
#-------------------------------------------------------------------------------

if (NOT DEFINED DYNO_TRITON_PYTHON_EXECUTABLE
    OR DYNO_TRITON_PYTHON_EXECUTABLE STREQUAL "")
  if (EXISTS "${DYNO_TRITON_VENV_DIR}/bin/python")
    set(_dyno_triton_python "${DYNO_TRITON_VENV_DIR}/bin/python")
  elseif (DEFINED Python3_EXECUTABLE AND EXISTS "${Python3_EXECUTABLE}")
    set(_dyno_triton_python "${Python3_EXECUTABLE}")
  else()
    find_program(_dyno_triton_python NAMES python3 python)
  endif()

  set(DYNO_TRITON_PYTHON_EXECUTABLE "${_dyno_triton_python}")
endif()

#-------------------------------------------------------------------------------
# Public validation
#-------------------------------------------------------------------------------

# Validate that the optional Triton benchmark toolchain is present and usable.
#
# Checks:
# - resolved host MLIR/LLVM tools from the shared kernel-build configuration
# - upstream Triton, Triton CPU, and Triton Shared source checkouts
# - the Python executable used to generate TTIR
# - the Triton Shared and Triton CPU optimizer/translator binaries
#
# This function is called only when `ENABLE_TRITON_BENCHMARKS=ON` so the normal
# build remains independent from the Triton setup.
function(dyno_validate_triton_benchmarks)
  set(_missing)

  foreach(_tool_var IN ITEMS DYNO_HOST_MLIR_OPT DYNO_HOST_MLIR_TRANSLATE
                             DYNO_HOST_CLANG)
    if (NOT DEFINED ${_tool_var} OR NOT EXISTS "${${_tool_var}}")
      list(APPEND _missing
        "missing resolved host tool `${_tool_var}`; set DYNO_HOST_LLVM_TOOLS_DIR if needed")
    endif()
  endforeach()

  if (NOT EXISTS "${DYNO_TRITON_UPSTREAM_SOURCE_DIR}/setup.py")
    list(APPEND _missing
      "missing upstream Triton checkout: ${DYNO_TRITON_UPSTREAM_SOURCE_DIR}")
  endif()

  if (NOT EXISTS "${DYNO_TRITON_CPU_SOURCE_DIR}/setup.py")
    list(APPEND _missing
      "missing Triton CPU checkout: ${DYNO_TRITON_CPU_SOURCE_DIR}")
  endif()

  if (NOT EXISTS "${DYNO_TRITON_SHARED_SOURCE_DIR}/CMakeLists.txt")
    list(APPEND _missing
      "missing Triton Shared checkout: ${DYNO_TRITON_SHARED_SOURCE_DIR}")
  endif()

  if (NOT EXISTS "${DYNO_TRITON_PYTHON_EXECUTABLE}")
    list(APPEND _missing
      "missing Triton Python executable: ${DYNO_TRITON_PYTHON_EXECUTABLE}")
  endif()

  if (NOT EXISTS "${DYNO_TRITON_SHARED_OPT}")
    list(APPEND _missing
      "missing triton-shared-opt: ${DYNO_TRITON_SHARED_OPT}")
  endif()

  if (NOT EXISTS "${DYNO_TRITON_LLVM_BINARY_DIR}/mlir-opt")
    list(APPEND _missing
      "missing Triton Shared LLVM optimizer: ${DYNO_TRITON_LLVM_BINARY_DIR}/mlir-opt")
  endif()

  if (NOT EXISTS "${DYNO_TRITON_LLVM_BINARY_DIR}/mlir-translate")
    list(APPEND _missing
      "missing Triton Shared LLVM translator: ${DYNO_TRITON_LLVM_BINARY_DIR}/mlir-translate")
  endif()

  if (NOT EXISTS "${DYNO_TRITON_LLVM_BINARY_DIR}/opt")
    list(APPEND _missing
      "missing Triton Shared LLVM opt: ${DYNO_TRITON_LLVM_BINARY_DIR}/opt")
  endif()

  if (NOT EXISTS "${DYNO_TRITON_LLVM_BINARY_DIR}/llc")
    list(APPEND _missing
      "missing Triton Shared LLVM llc: ${DYNO_TRITON_LLVM_BINARY_DIR}/llc")
  endif()

  if (NOT EXISTS "${DYNO_TRITON_CPU_OPT}")
    list(APPEND _missing "missing triton-opt: ${DYNO_TRITON_CPU_OPT}")
  endif()

  if (NOT EXISTS "${DYNO_TRITON_CPU_LLVM_BINARY_DIR}/mlir-translate")
    list(APPEND _missing
      "missing Triton CPU LLVM translator: ${DYNO_TRITON_CPU_LLVM_BINARY_DIR}/mlir-translate")
  endif()

  if (_missing)
    string(JOIN "\n  " _formatted_missing ${_missing})
    message(FATAL_ERROR
      "ENABLE_TRITON_BENCHMARKS=ON, but the Triton benchmark toolchain is not ready:\n"
      "  ${_formatted_missing}\n"
      "Run ${DYNO_SOURCE_DIR}/scripts/setup-triton.sh ${DYNO_BINARY_DIR}\n"
      "or override the relevant DYNO_TRITON_* variables with -D.")
  endif()
endfunction()
