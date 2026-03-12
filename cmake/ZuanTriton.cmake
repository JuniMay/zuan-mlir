set(ZUAN_HOST_LLVM_BINARY_DIR
  "${ZUAN_SOURCE_DIR}/llvm-project/build/bin"
  CACHE PATH "Path to the host LLVM tools used by the benchmark lowering pipeline")

set(ZUAN_TRITON_UPSTREAM_SOURCE_DIR
  "${ZUAN_SOURCE_DIR}/third_party/triton"
  CACHE PATH "Path to the upstream Triton source checkout used by Triton Shared")

set(ZUAN_TRITON_CPU_SOURCE_DIR
  "${ZUAN_SOURCE_DIR}/third_party/triton-cpu"
  CACHE PATH "Path to the Triton CPU source checkout")

set(ZUAN_TRITON_SHARED_SOURCE_DIR
  "${ZUAN_SOURCE_DIR}/third_party/triton_shared"
  CACHE PATH "Path to the Triton Shared source checkout")

set(ZUAN_TRITON_ROOT_DIR
  "${ZUAN_BINARY_DIR}/third_party/triton"
  CACHE PATH "Root directory for build-local Triton artifacts")

set(ZUAN_TRITON_SHARED_ROOT_DIR
  "${ZUAN_TRITON_ROOT_DIR}/shared"
  CACHE PATH "Root directory for Triton Shared build artifacts")

set(ZUAN_TRITON_SHARED_BUILD_DIR
  "${ZUAN_TRITON_SHARED_ROOT_DIR}/build"
  CACHE PATH "Path to the upstream Triton build directory used by Triton Shared")

set(ZUAN_TRITON_VENV_DIR
  "${ZUAN_TRITON_SHARED_ROOT_DIR}/venv"
  CACHE PATH "Path to the Triton Python virtual environment")

set(ZUAN_TRITON_SHARED_OPT
  "${ZUAN_TRITON_SHARED_BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
  CACHE FILEPATH "Path to triton-shared-opt")

set(ZUAN_TRITON_LLVM_BINARY_DIR
  "${ZUAN_TRITON_SHARED_ROOT_DIR}/llvm/bin"
  CACHE PATH "Path to the LLVM bin directory used by Triton Shared")

set(ZUAN_TRITON_CPU_ROOT_DIR
  "${ZUAN_TRITON_ROOT_DIR}/cpu"
  CACHE PATH "Root directory for Triton CPU build artifacts")

set(ZUAN_TRITON_CPU_BUILD_DIR
  "${ZUAN_TRITON_CPU_ROOT_DIR}/build"
  CACHE PATH "Path to the Triton CPU build directory")

set(ZUAN_TRITON_CPU_OPT
  "${ZUAN_TRITON_CPU_BUILD_DIR}/bin/triton-opt"
  CACHE FILEPATH "Path to triton-opt")

set(ZUAN_TRITON_CPU_LLVM_BINARY_DIR
  "${ZUAN_TRITON_CPU_ROOT_DIR}/llvm/bin"
  CACHE PATH "Path to the LLVM bin directory used by Triton CPU")

if (NOT DEFINED ZUAN_TRITON_PYTHON_EXECUTABLE OR ZUAN_TRITON_PYTHON_EXECUTABLE STREQUAL "")
  if (EXISTS "${ZUAN_TRITON_VENV_DIR}/bin/python")
    set(_zuan_triton_python "${ZUAN_TRITON_VENV_DIR}/bin/python")
  elseif (DEFINED Python3_EXECUTABLE AND EXISTS "${Python3_EXECUTABLE}")
    set(_zuan_triton_python "${Python3_EXECUTABLE}")
  else()
    find_program(_zuan_triton_python NAMES python3 python)
  endif()

  set(ZUAN_TRITON_PYTHON_EXECUTABLE
    "${_zuan_triton_python}"
    CACHE FILEPATH "Python executable used to generate Triton benchmark TTIR" FORCE)
endif()

function(zuan_validate_triton_benchmarks)
  set(_missing)

  if (NOT EXISTS "${ZUAN_TRITON_UPSTREAM_SOURCE_DIR}/setup.py")
    list(APPEND _missing "missing upstream Triton checkout: ${ZUAN_TRITON_UPSTREAM_SOURCE_DIR}")
  endif()

  if (NOT EXISTS "${ZUAN_TRITON_CPU_SOURCE_DIR}/setup.py")
    list(APPEND _missing "missing Triton CPU checkout: ${ZUAN_TRITON_CPU_SOURCE_DIR}")
  endif()

  if (NOT EXISTS "${ZUAN_TRITON_SHARED_SOURCE_DIR}/CMakeLists.txt")
    list(APPEND _missing "missing Triton Shared checkout: ${ZUAN_TRITON_SHARED_SOURCE_DIR}")
  endif()

  if (NOT EXISTS "${ZUAN_TRITON_PYTHON_EXECUTABLE}")
    list(APPEND _missing "missing Triton Python executable: ${ZUAN_TRITON_PYTHON_EXECUTABLE}")
  endif()

  if (NOT EXISTS "${ZUAN_TRITON_SHARED_OPT}")
    list(APPEND _missing "missing triton-shared-opt: ${ZUAN_TRITON_SHARED_OPT}")
  endif()

  if (NOT EXISTS "${ZUAN_TRITON_LLVM_BINARY_DIR}/mlir-opt")
    list(APPEND _missing "missing Triton Shared LLVM optimizer: ${ZUAN_TRITON_LLVM_BINARY_DIR}/mlir-opt")
  endif()

  if (NOT EXISTS "${ZUAN_TRITON_LLVM_BINARY_DIR}/mlir-translate")
    list(APPEND _missing "missing Triton Shared LLVM translator: ${ZUAN_TRITON_LLVM_BINARY_DIR}/mlir-translate")
  endif()

  if (NOT EXISTS "${ZUAN_TRITON_LLVM_BINARY_DIR}/opt")
    list(APPEND _missing "missing Triton Shared LLVM opt: ${ZUAN_TRITON_LLVM_BINARY_DIR}/opt")
  endif()

  if (NOT EXISTS "${ZUAN_TRITON_LLVM_BINARY_DIR}/llc")
    list(APPEND _missing "missing Triton Shared LLVM llc: ${ZUAN_TRITON_LLVM_BINARY_DIR}/llc")
  endif()

  if (NOT EXISTS "${ZUAN_TRITON_CPU_OPT}")
    list(APPEND _missing "missing triton-opt: ${ZUAN_TRITON_CPU_OPT}")
  endif()

  if (NOT EXISTS "${ZUAN_TRITON_CPU_LLVM_BINARY_DIR}/mlir-translate")
    list(APPEND _missing "missing Triton CPU LLVM translator: ${ZUAN_TRITON_CPU_LLVM_BINARY_DIR}/mlir-translate")
  endif()

  if (NOT EXISTS "${ZUAN_HOST_LLVM_BINARY_DIR}/mlir-translate")
    list(APPEND _missing "missing host LLVM translator: ${ZUAN_HOST_LLVM_BINARY_DIR}/mlir-translate")
  endif()

  if (NOT EXISTS "${ZUAN_HOST_LLVM_BINARY_DIR}/clang")
    list(APPEND _missing "missing host LLVM clang: ${ZUAN_HOST_LLVM_BINARY_DIR}/clang")
  endif()

  if (_missing)
    string(JOIN "\n  " _formatted_missing ${_missing})
    message(FATAL_ERROR
      "ENABLE_TRITON_BENCHMARKS=ON, but the Triton benchmark toolchain is not ready:\n"
      "  ${_formatted_missing}\n"
      "Run ${ZUAN_SOURCE_DIR}/scripts/setup-triton.sh ${ZUAN_BINARY_DIR}\n"
      "or override the ZUAN_TRITON_* cache entries.")
  endif()
endfunction()
